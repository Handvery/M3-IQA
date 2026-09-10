"""The two objectives used by the final method."""

import torch


def pairwise_fidelity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    upper = torch.triu(
        torch.ones(
            prediction.numel(),
            prediction.numel(),
            dtype=torch.bool,
            device=prediction.device,
        ),
        diagonal=1,
    )
    predicted_probability = torch.sigmoid(
        (prediction[:, None] - prediction[None, :])[upper] / temperature
    )
    target_probability = torch.sigmoid(
        (target[:, None] - target[None, :])[upper] / temperature
    )
    predicted_probability = predicted_probability.clamp(eps, 1.0 - eps)
    target_probability = target_probability.clamp(eps, 1.0 - eps)
    fidelity = (
        1.0
        - torch.sqrt(predicted_probability * target_probability)
        - torch.sqrt(
            (1.0 - predicted_probability) * (1.0 - target_probability)
        )
    )
    return fidelity.mean()


def pearson_loss(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    correlation = (prediction * target).mean() / (
        prediction.square().mean().add(eps).sqrt()
        * target.square().mean().add(eps).sqrt()
    )
    return 1.0 - correlation


__all__ = ["pairwise_fidelity_loss", "pearson_loss"]
