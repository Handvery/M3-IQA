"""Slim checkpoints plus deterministic migration of legacy Q-CoPS heads."""

from __future__ import annotations

import torch


LEGACY_UNUSED_KEYS = {"criterion_logits", "global_fusion_logits", "hierarchy_blend_logits"}


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as error:
        # Historical files contain NumPy scalar metadata, which PyTorch's
        # restricted loader may reject.  Only use this fallback for a path the
        # caller explicitly supplied.
        if "Weights only load failed" not in str(error):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def checkpoint_state(payload) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a dictionary")
    for key in ("miqanet", "model", "eviiqa_head"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise KeyError("Checkpoint contains no MIQANet state dict")


def load_model_state(model: torch.nn.Module, checkpoint: str):
    payload = safe_torch_load(checkpoint, map_location="cpu")
    source = checkpoint_state(payload)
    if any("compound_prompts" in key for key in source):
        raise ValueError("This checkpoint uses deep prompts and is incompatible with this model")
    target = model.state_dict()
    filtered = {key: value for key, value in source.items() if key not in LEGACY_UNUSED_KEYS}
    missing = sorted(set(target).difference(filtered))
    unexpected = sorted(set(filtered).difference(target))
    mismatched = sorted(
        key for key, value in filtered.items() if key in target and value.shape != target[key].shape
    )
    if missing or mismatched or unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
        )
    model.load_state_dict(filtered, strict=True)
    return payload


__all__ = ["load_model_state", "safe_torch_load"]
