import dataclasses

import torch


@dataclasses.dataclass
class Checkpoint:
    """Pickle-friendly training checkpoint."""

    step: int
    """1-based step count."""
    model_type: str  # explicit str to pickle correctly
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, torch.Tensor]
    train_loss: list[float]
    val_loss: list[tuple[int, float]]
