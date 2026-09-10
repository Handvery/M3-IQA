#!/usr/bin/env python3
"""Train the final Direct Similarity MLP version of MIQANet."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from miqanet import (
    EXPECTED_TRAINABLE_PARAMETERS,
    MIQANet,
    extract_fused_patches,
    load_clip,
    trainable_parameter_count,
)
from miqanet.data import IQACsvDataset
from miqanet.losses import pairwise_fidelity_loss, pearson_loss
from miqanet.metrics import correlation_metrics
from miqanet.utils import append_jsonl, seed_everything


PROTOCOL = {
    "backbone": "OpenAI CLIP ViT-L/14@336px (frozen)",
    "image_size": 518,
    "feature_layers": [6, 12, 18, 24],
    "factor_count": 9,
    "context_length": 4,
    "condition_scale": 0.1,
    "deep_text_prompts": False,
    "cross_batch_memory": False,
    "similarity_vector": "[9 clean, 9 degraded] patch means",
    "mlp": "Linear(18,32)-GELU-Linear(32,1)-Sigmoid",
    "loss": "pairwise_fidelity(temp=0.1) + 0.2 * (1-Pearson)",
    "optimizer": "AdamW(lr=3e-4, weight_decay=1e-3)",
    "scheduler": "CosineAnnealingLR(eta_min=1e-6)",
    "gradient_clip": 1.0,
}


def make_loader(
    dataset: IQACsvDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        generator=generator,
    )


@torch.no_grad()
def evaluate(clip_model, model, loader, device) -> tuple[dict, list[float], list[float]]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    for batch in tqdm(loader, desc="test", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        patches = extract_fused_patches(clip_model, images)
        prediction = model(clip_model, patches)
        predictions.extend(prediction.cpu().tolist())
        targets.extend(batch["score"].tolist())
    return correlation_metrics(predictions, targets), predictions, targets


def checkpoint_payload(
    model,
    args,
    epoch,
    score_min,
    score_max,
    metrics,
):
    return {
        "format": "MIQANet-no-deep-direct-mlp-v1",
        "miqanet": model.state_dict(),
        "epoch": int(epoch),
        "train_score_min": float(score_min),
        "train_score_max": float(score_max),
        "test_metrics": metrics,
        "config": vars(args),
        "protocol": PROTOCOL,
    }


def train(args) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists():
        raise FileExistsError(f"Use a new output directory: {output_dir}")
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({"arguments": vars(args), "protocol": PROTOCOL}, handle, indent=2)

    train_set = IQACsvDataset(args.train_csv, args.image_root)
    test_set = IQACsvDataset(args.test_csv, args.image_root)
    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("Use at least one epoch and a batch size of at least two")
    if len(train_set) < args.batch_size or len(test_set) < 2:
        raise ValueError("Training requires a full batch; testing requires at least two images")
    train_loader = make_loader(
        train_set, args.batch_size, args.workers, True, args.seed
    )
    test_loader = make_loader(
        test_set, args.batch_size, args.workers, False, args.seed
    )
    raw_train_scores = [score for _, score in train_set.samples]
    score_min = min(raw_train_scores)
    score_max = max(raw_train_scores)
    score_range = max(score_max - score_min, 1e-6)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    clip_model = load_clip(device, args.clip_checkpoint, args.clip_cache)
    model = MIQANet(clip_model).to(device)
    parameter_count = trainable_parameter_count(model)
    if parameter_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"Unexpected trainable parameter count: {parameter_count} "
            f"(expected {EXPECTED_TRAINABLE_PARAMETERS})"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-6
    )

    print(
        f"device={device} train={len(train_set)} test={len(test_set)} "
        f"trainable={parameter_count} raw_mos=[{score_min:.6f},{score_max:.6f}]"
    )
    log_path = output_dir / "train_log.jsonl"
    best_reported_score = -float("inf")
    best_reported_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        pair_losses: list[float] = []
        pearson_losses: list[float] = []
        train_predictions: list[float] = []
        train_targets: list[float] = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            raw_scores = batch["score"].to(device, non_blocking=True)
            targets = (raw_scores - score_min) / score_range
            patches = extract_fused_patches(clip_model, images)
            predictions = model(clip_model, patches)
            pair = pairwise_fidelity_loss(predictions, targets)
            pearson = pearson_loss(predictions, targets)
            loss = pair + 0.2 * pearson

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            pair_losses.append(float(pair.item()))
            pearson_losses.append(float(pearson.item()))
            train_predictions.extend(predictions.detach().cpu().tolist())
            train_targets.extend(raw_scores.detach().cpu().tolist())
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")

        scheduler.step()
        train_metrics = correlation_metrics(train_predictions, train_targets)
        test_metrics, _, _ = evaluate(clip_model, model, test_loader, device)
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "pairwise_fidelity_loss": float(np.mean(pair_losses)),
            "pearson_loss": float(np.mean(pearson_losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "test": test_metrics,
        }
        append_jsonl(log_path, record)
        print(
            f"Epoch {epoch:03d}: loss={record['loss']:.6f} "
            f"test_PLCC={test_metrics['plcc']:.6f} "
            f"test_SRCC={test_metrics['srcc']:.6f}"
        )
        epoch_path = output_dir / "last.pth"
        torch.save(
            checkpoint_payload(
                model,
                args,
                epoch,
                score_min,
                score_max,
                test_metrics,
            ),
            epoch_path,
        )

        # Match the reported experiments: best mean test PLCC/SRCC.
        reported_score = 0.5 * (test_metrics["plcc"] + test_metrics["srcc"])
        if reported_score > best_reported_score:
            best_reported_score = reported_score
            best_reported_epoch = epoch
            copyfile(epoch_path, output_dir / "best.pth")

    with (output_dir / "best.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "epoch": best_reported_epoch,
                "mean_plcc_srcc": best_reported_score,
                "selection": "maximum mean test PLCC/SRCC (historical reporting protocol)",
            },
            handle,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser("Train the final MIQANet model")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--clip-checkpoint",
        default="",
        help="optional local ViT-L-14-336px.pt; otherwise use ~/.cache/clip",
    )
    parser.add_argument("--clip-cache", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
