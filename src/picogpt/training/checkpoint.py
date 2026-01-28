import dataclasses

import torch


@dataclasses.dataclass
class Checkpoint:
    """Pickle-friendly training checkpoint."""

    model_type: str  # explicit str to pickle correctly
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, torch.Tensor]
    epoch: int
    loss_history: list[float]
