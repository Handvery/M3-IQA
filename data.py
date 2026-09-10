"""CSV dataset and deterministic CLIP preprocessing."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def image_transform(image_size: int = 518):
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=None,
            ),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


class IQACsvDataset(Dataset):
    """Read ``score`` plus ``image_path``/``image_name`` from a CSV file."""

    def __init__(self, csv_file: str, image_root: str = ""):
        self.csv_file = str(csv_file)
        self.image_root = Path(image_root).expanduser() if image_root else None
        self.transform = image_transform()
        self.samples: list[tuple[str, float]] = []
        with open(csv_file, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            path_key = "image_path" if "image_path" in fields else "image_name"
            if path_key not in fields or "score" not in fields:
                raise ValueError(
                    "CSV must contain 'score' and either 'image_path' or 'image_name'; "
                    f"found {reader.fieldnames}"
                )
            for row in reader:
                path = row[path_key].strip()
                if path:
                    score = float(row["score"])
                    if not math.isfinite(score):
                        raise ValueError(f"Non-finite score for {path}")
                    self.samples.append((path, score))
        if not self.samples:
            raise ValueError(f"No samples found in {csv_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def resolved_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            if self.image_root is None:
                raise ValueError(
                    f"Relative image path requires --image-root: {path}"
                )
            candidate = self.image_root / candidate
        return candidate

    def __getitem__(self, index: int):
        path, score = self.samples[index]
        resolved = self.resolved_path(path)
        with Image.open(resolved) as image:
            image = self.transform(image.convert("RGB"))
        return {
            "image": image,
            "score": torch.tensor(score, dtype=torch.float32),
            "image_path": str(resolved),
        }


__all__ = ["IQACsvDataset", "image_transform"]
