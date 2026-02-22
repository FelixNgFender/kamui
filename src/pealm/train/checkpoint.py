import dataclasses
import logging
import pathlib

import torch

from pealm import model, train

logger = logging.getLogger(__name__)


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

    @classmethod
    def load(cls, ckpt: pathlib.Path, map_location: torch.device) -> "Checkpoint":
        logger.debug("loading checkpoint from %s", ckpt)
        checkpoint = torch.load(
            ckpt,
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(checkpoint, train.Checkpoint):
            msg = f"expected Checkpoint instance, got {type(checkpoint)}"
            raise TypeError(msg)
        return checkpoint

    @classmethod
    def load_weights(cls, ckpt: pathlib.Path, map_location: torch.device) -> dict[str, torch.Tensor]:
        logger.debug("loading weights from %s", ckpt)
        with torch.serialization.safe_globals([train.Checkpoint, model.Type]):
            checkpoint = torch.load(ckpt, map_location=map_location, weights_only=True)
            if isinstance(checkpoint, train.Checkpoint):
                model_state_dict = checkpoint.model_state_dict
            elif isinstance(checkpoint, dict):
                model_state_dict = checkpoint
            else:
                msg = f"unsupported checkpoint type: {type(checkpoint)}"
                raise TypeError(msg)
        return model_state_dict
