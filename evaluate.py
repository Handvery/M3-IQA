#!/usr/bin/env python3
"""Evaluate a slim MIQANet checkpoint or an existing legacy Q-CoPS one."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from miqanet import MIQANet, extract_fused_patches, load_clip
from miqanet.checkpoint import load_model_state
from miqanet.data import IQACsvDataset
from miqanet.metrics import correlation_metrics
from miqanet.utils import seed_everything


@torch.no_grad()
def run(args) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    clip_model = load_clip(device, args.clip_checkpoint, args.clip_cache)
    model = MIQANet(clip_model).to(device)
    payload = load_model_state(model, args.checkpoint)
    model.eval()

    score_min = payload.get("train_score_min") if isinstance(payload, dict) else None
    score_max = payload.get("train_score_max") if isinstance(payload, dict) else None
    if score_min is None or score_max is None:
        raise KeyError("Checkpoint does not contain train_score_min/train_score_max")
    score_min, score_max = float(score_min), float(score_max)

    dataset = IQACsvDataset(args.csv, args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    predictions: list[float] = []
    targets: list[float] = []
    paths: list[str] = []
    for batch in tqdm(loader, desc="evaluate"):
        images = batch["image"].to(device, non_blocking=True)
        patches = extract_fused_patches(clip_model, images)
        prediction = model(clip_model, patches)
        predictions.extend(prediction.cpu().tolist())
        targets.extend(batch["score"].tolist())
        paths.extend(batch["image_path"])

    metrics = correlation_metrics(predictions, targets)
    mos_predictions = score_min + (score_max - score_min) * np.asarray(predictions)
    metrics.update(
        {
            "samples": len(dataset),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "train_score_min": score_min,
            "train_score_max": score_max,
        }
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image_path", "score", "prediction", "prediction_mos_scale"]
        )
        writer.writerows(zip(paths, targets, predictions, mos_predictions.tolist()))
    metrics_json = Path(args.metrics_json)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"predictions={output_csv}")
    print(f"metrics={metrics_json}")


def parse_args():
    parser = argparse.ArgumentParser("Evaluate MIQANet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--clip-checkpoint", default="")
    parser.add_argument("--clip-cache", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
