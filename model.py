"""The final MIQANet model: factorized similarities followed by one MLP."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import LayerNorm, ModifiedCLIP, QuickGELU
from .simple_tokenizer import SimpleTokenizer


FACTOR_PROMPTS: Sequence[tuple[str, str, str]] = (
    (
        "blur",
        "a sharp photograph with clearly resolved fine details.",
        "a blurry photograph with smeared and poorly resolved details.",
    ),
    (
        "noise",
        "a clean photograph with smooth regions and little visible noise.",
        "a noisy photograph with visible grain and random sensor noise.",
    ),
    (
        "compression",
        "a cleanly encoded photograph without blocking artifacts.",
        "a heavily compressed photograph with blocking and quantization artifacts.",
    ),
    (
        "resampling",
        "a naturally sampled photograph with stable edges and textures.",
        "a poorly resampled photograph with ringing, aliasing, and damaged edges.",
    ),
    (
        "banding",
        "a photograph with smooth tonal transitions and continuous gradients.",
        "a photograph with posterization, banding, and broken tonal transitions.",
    ),
    (
        "exposure",
        "a well exposed photograph with visible highlight and shadow detail.",
        "a poorly exposed photograph with clipped highlights or crushed shadows.",
    ),
    (
        "contrast",
        "a photograph with natural contrast and a well balanced dynamic range.",
        "a photograph with washed-out or unnaturally harsh contrast.",
    ),
    (
        "color",
        "a photograph with natural colors and accurate white balance.",
        "a photograph with color casts, inaccurate colors, and poor white balance.",
    ),
    (
        "overall",
        "an overall high-quality photograph that is clear and visually pleasing.",
        "an overall low-quality photograph with visible perceptual degradation.",
    ),
)


_TOKENIZER = SimpleTokenizer()


def tokenize(
    texts: str | list[str], context_length: int = 77
) -> torch.IntTensor:
    if isinstance(texts, str):
        texts = [texts]
    start = _TOKENIZER.encoder["<|startoftext|>"]
    end = _TOKENIZER.encoder["<|endoftext|>"]
    encoded = [[start] + _TOKENIZER.encode(text) + [end] for text in texts]
    output = torch.zeros(len(encoded), context_length, dtype=torch.int)
    for index, tokens in enumerate(encoded):
        if len(tokens) > context_length:
            raise RuntimeError(f"Prompt is longer than {context_length} tokens: {texts[index]}")
        output[index, : len(tokens)] = torch.tensor(tokens)
    return output


def _initialize_prompt_module(module: nn.Module) -> None:
    name = module.__class__.__name__
    if "Linear" in name:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif "LayerNorm" in name:
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PrototypeExtractor(nn.Module):
    """Cross-attend factor queries to the fused CLIP patch tokens."""

    def __init__(self, width: int = 768, heads: int = 12):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_1 = LayerNorm(width)
        self.ln_2 = LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(width, width)),
                ]
            )
        )
        self.apply(_initialize_prompt_module)

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        queries = queries.unsqueeze(0).repeat(image_tokens.shape[0], 1, 1)
        normalized_queries = self.ln_1(queries)
        normalized_image = self.ln_1(image_tokens)
        attended = self.attn(
            normalized_queries,
            normalized_image,
            normalized_image,
            need_weights=False,
        )[0]
        queries = queries + attended
        return queries + self.mlp(self.ln_2(queries))


class FactorPromptLearner(nn.Module):
    """Create nine image-conditioned clean/degraded prompt pairs."""

    def __init__(
        self,
        clip_model: ModifiedCLIP,
        context_length: int = 4,
        condition_scale: float = 0.1,
    ):
        super().__init__()
        self.factor_names = tuple(factor[0] for factor in FACTOR_PROMPTS)
        self.num_factors = len(self.factor_names)
        self.context_length = int(context_length)
        self.condition_scale = float(condition_scale)
        self.dtype = clip_model.transformer.get_cast_dtype()
        width = int(clip_model.ln_final.weight.shape[0])

        self.base_context = nn.Parameter(torch.empty(context_length, width))
        nn.init.normal_(self.base_context, std=0.02)
        self.query_offsets = nn.Parameter(torch.zeros(context_length, width))

        placeholder = " ".join(["X"] * context_length)
        prompt_texts: list[str] = []
        anchor_texts: list[str] = []
        for _, clean, degraded in FACTOR_PROMPTS:
            prompt_texts.extend((f"{placeholder} {clean}", f"{placeholder} {degraded}"))
            anchor_texts.extend((clean, degraded))

        tokenized_prompts = torch.cat([tokenize(text) for text in prompt_texts])
        tokenized_anchors = torch.cat([tokenize(text) for text in anchor_texts])
        device = next(clip_model.parameters()).device
        with torch.no_grad():
            prompt_embeddings = clip_model.token_embedding(
                tokenized_prompts.to(device)
            ).to(self.dtype)
            anchor_embeddings = clip_model.token_embedding(
                tokenized_anchors.to(device)
            ).to(self.dtype)
            anchor_mask = tokenized_anchors.to(device).ne(0).float()
            anchor_mask[:, 0] = 0.0
            anchor_features = (
                anchor_embeddings.float() * anchor_mask.unsqueeze(-1)
            ).sum(dim=1) / anchor_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            anchor_features = F.normalize(anchor_features, dim=-1)
            anchor_features = anchor_features.reshape(self.num_factors, 2, -1)
            factor_axes = F.normalize(
                anchor_features[:, 1] - anchor_features[:, 0], dim=-1
            )

        if factor_axes.shape[-1] != width:
            raise ValueError(
                f"Text embedding width {factor_axes.shape[-1]} does not match {width}"
            )
        self.register_buffer("factor_axes", factor_axes.cpu())
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", prompt_embeddings[:, :1].cpu())
        self.register_buffer(
            "token_suffix", prompt_embeddings[:, 1 + context_length :].cpu()
        )
        self.prototype_extractor = PrototypeExtractor(width, 12)

    def forward(
        self, clip_model: ModifiedCLIP, patch_features: torch.Tensor
    ) -> torch.Tensor:
        batch_size = patch_features.shape[0]
        device = patch_features.device
        factor_queries = self.factor_axes.to(device).unsqueeze(1)
        factor_queries = factor_queries + 0.1 * torch.tanh(
            self.query_offsets
        ).unsqueeze(0)
        extracted = self.prototype_extractor(
            factor_queries.reshape(self.num_factors * self.context_length, -1).float(),
            patch_features.float(),
        ).reshape(batch_size, self.num_factors, self.context_length, -1)
        residual = torch.tanh(extracted - factor_queries.unsqueeze(0))
        context = self.base_context.view(1, 1, self.context_length, -1)
        context = context + self.condition_scale * residual

        # A factor's clean and degraded prompts share the same visual residual.
        context = context.unsqueeze(2).expand(-1, -1, 2, -1, -1)
        context = context.reshape(
            batch_size, 2 * self.num_factors, self.context_length, -1
        ).to(self.dtype)
        prefix = self.token_prefix.to(device).unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = self.token_suffix.to(device).unsqueeze(0).expand(batch_size, -1, -1, -1)
        prompts = torch.cat((prefix, context, suffix), dim=2)
        prompts = prompts.reshape(-1, prompts.shape[-2], prompts.shape[-1])
        tokenized = self.tokenized_prompts.to(device).unsqueeze(0)
        tokenized = tokenized.expand(batch_size, -1, -1).reshape(-1, tokenized.shape[-1])
        text_features = clip_model.encode_text(prompts, tokenized).float()
        return F.normalize(text_features, dim=-1).reshape(
            batch_size, self.num_factors, 2, -1
        )


class MIQANet(nn.Module):
    """Pool clean/degraded patch similarities and regress normalized quality."""

    feature_layers = (6, 12, 18, 24)

    def __init__(self, clip_model: ModifiedCLIP):
        super().__init__()
        self.prompt_learner = FactorPromptLearner(clip_model)
        self.similarity_mlp = nn.Sequential(
            nn.Linear(2 * len(FACTOR_PROMPTS), 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, clip_model: ModifiedCLIP, fused_patch_features: torch.Tensor
    ) -> torch.Tensor:
        # Normalize the cross-level mean before computing cosine similarities.
        patches = F.normalize(fused_patch_features.float(), dim=-1)
        text = self.prompt_learner(clip_model, F.normalize(patches, dim=-1))
        similarity = torch.einsum("bnc,bksc->bnks", patches, text)
        pooled = similarity.mean(dim=1)  # B x 9 x [clean, degraded]
        vector = torch.cat((pooled[..., 0], pooled[..., 1]), dim=-1)
        return torch.sigmoid(self.similarity_mlp(vector).squeeze(-1))

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self.prompt_learner.factor_names


@torch.no_grad()
def extract_fused_patches(
    clip_model: ModifiedCLIP,
    images: torch.Tensor,
    feature_layers: Sequence[int] = MIQANet.feature_layers,
) -> torch.Tensor:
    layer_features = clip_model.encode_image(images, feature_layers)
    layer_features = [F.normalize(features.float(), dim=-1) for features in layer_features]
    return torch.stack(layer_features, dim=0).mean(dim=0)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


EXPECTED_TRAINABLE_PARAMETERS = 3_553_409


__all__ = [
    "EXPECTED_TRAINABLE_PARAMETERS",
    "FACTOR_PROMPTS",
    "MIQANet",
    "extract_fused_patches",
    "trainable_parameter_count",
]
