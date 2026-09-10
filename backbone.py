"""Minimal modified CLIP runtime used by MIQANet.

Only the native ViT path, intermediate patch-token extraction, and text
encoding of shallow prompts are retained. The module keeps the parameter
names of the OpenAI CLIP checkpoint, so the official ViT-L/14@336px weights
load without conversion.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
from typing import Sequence
import urllib.request

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


MODEL_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/"
    "ViT-L-14-336px.pt"
)
MODEL_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"


class LayerNorm(nn.LayerNorm):
    """LayerNorm with fp32 accumulation, matching the original CLIP code."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return super().forward(x.float()).to(dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Standard frozen CLIP transformer block."""

    def __init__(self, width: int, heads: int, attention_mask: torch.Tensor | None = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(width * 4, width)),
                ]
            )
        )
        self.ln_2 = LayerNorm(width)
        self.attn_mask = attention_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.attn_mask
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.no_grad()
    def forward(self, image: torch.Tensor, feature_layers: Sequence[int]):
        x = self.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros_like(x[:, :1])
        x = torch.cat((cls, x), dim=1)

        old_side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)
        if old_side != new_side:
            patch_position = self.positional_embedding[1:].reshape(
                1, old_side, old_side, x.shape[-1]
            ).permute(0, 3, 1, 2)
            patch_position = F.interpolate(
                patch_position, (new_side, new_side), mode="bilinear"
            )
            patch_position = patch_position.reshape(
                1, x.shape[-1], new_side * new_side
            ).transpose(1, 2)
            self.positional_embedding.data = torch.cat(
                (self.positional_embedding[:1], patch_position[0]), dim=0
            )

        x = self.ln_pre(x + self.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        requested = set(int(layer) for layer in feature_layers)
        patch_tokens = []
        for layer_index, block in enumerate(self.transformer.resblocks, start=1):
            x = block(x)
            if layer_index in requested:
                patch_tokens.append(x.clone())
        if len(patch_tokens) != len(feature_layers):
            raise ValueError(
                f"Requested layers {list(feature_layers)}, obtained {len(patch_tokens)} tensors"
            )
        return [
            self.ln_post(tokens.permute(1, 0, 2))[:, 1:] @ self.proj
            for tokens in patch_tokens
        ]


class ModifiedCLIP(nn.Module):
    """The exact CLIP interfaces needed by MIQANet."""

    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()
        self.context_length = context_length
        self.visual = VisionTransformer(
            image_resolution,
            vision_patch_size,
            vision_width,
            vision_layers,
            vision_width // 64,
            embed_dim,
        )

        self.transformer = nn.Module()
        self.transformer.width = transformer_width
        self.transformer.layers = transformer_layers
        attention_mask = self.build_attention_mask()
        self.transformer.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    transformer_width,
                    transformer_heads,
                    attention_mask,
                )
                for _ in range(transformer_layers)
            ]
        )
        self.transformer.get_cast_dtype = lambda: self.transformer.resblocks[
            0
        ].mlp.c_fc.weight.dtype

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        projection_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attention_std = self.transformer.width**-0.5
        feedforward_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attention_std)
            nn.init.normal_(block.attn.out_proj.weight, std=projection_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=feedforward_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=projection_std)
        nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self) -> torch.Tensor:
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        return mask.triu_(1)

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.conv1.weight.dtype

    def encode_image(
        self, image: torch.Tensor, feature_layers: Sequence[int]
    ) -> list[torch.Tensor]:
        return self.visual(image.to(self.dtype), feature_layers)

    def encode_text(
        self,
        prompts: torch.Tensor,
        tokenized_prompts: torch.Tensor,
    ) -> torch.Tensor:
        x = prompts + self.positional_embedding.to(self.dtype)
        x = x.permute(1, 0, 2)
        for block in self.transformer.resblocks:
            x = block(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(self.dtype)
        row = torch.arange(x.shape[0], device=x.device)
        return x[row, tokenized_prompts.argmax(dim=-1)] @ self.text_projection


def _download_checkpoint(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    root = Path(cache_dir or Path.home() / ".cache" / "clip").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "ViT-L-14-336px.pt"
    if target.is_file():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == MODEL_SHA256:
            return target
        target.unlink()
    with urllib.request.urlopen(MODEL_URL) as source, target.open("wb") as output:
        total = int(source.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="iB", unit_scale=True, desc="CLIP checkpoint") as bar:
            while True:
                chunk = source.read(8192)
                if not chunk:
                    break
                output.write(chunk)
                bar.update(len(chunk))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"CLIP checkpoint SHA256 mismatch: {target}")
    return target


def _load_openai_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    with checkpoint.open("rb") as handle:
        try:
            scripted = torch.jit.load(handle, map_location="cpu").eval()
            return scripted.state_dict()
        except RuntimeError:
            handle.seek(0)
            return torch.load(handle, map_location="cpu", weights_only=True)


def load_clip(
    device: torch.device | str,
    checkpoint: str = "",
    cache_dir: str = "",
) -> ModifiedCLIP:
    """Load the frozen OpenAI ViT-L/14@336px backbone."""

    path = Path(checkpoint).expanduser() if checkpoint else _download_checkpoint(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = _load_openai_state(path)
    if "visual.proj" not in state:
        raise ValueError("MIQANet requires a ViT CLIP checkpoint")

    vision_width = state["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [key for key in state if key.startswith("visual.") and key.endswith(".attn.in_proj_weight")]
    )
    patch_size = state["visual.conv1.weight"].shape[-1]
    grid_size = round((state["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = patch_size * grid_size
    embed_dim = state["text_projection"].shape[1]
    context_length = state["positional_embedding"].shape[0]
    vocab_size = state["token_embedding.weight"].shape[0]
    text_width = state["ln_final.weight"].shape[0]
    text_layers = len(
        {key.split(".")[2] for key in state if key.startswith("transformer.resblocks")}
    )
    model = ModifiedCLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        patch_size,
        context_length,
        vocab_size,
        text_width,
        text_width // 64,
        text_layers,
    )
    state = dict(state)
    for metadata_key in ("input_resolution", "context_length", "vocab_size"):
        state.pop(metadata_key, None)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = ["LayerNorm", "QuickGELU", "ModifiedCLIP", "load_clip"]
