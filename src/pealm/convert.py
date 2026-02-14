import logging
import pathlib

import torch

from pealm import training

logger = logging.getLogger(__name__)


def ckpt_to_weights(
    checkpoint_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Convert a full checkpoint to a weights-only checkpoint.

    This utility extracts just the model weights and tokenizer from a full checkpoint,
    discarding other training information.
    """
    if not checkpoint_path.exists():
        msg = f"checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(checkpoint, training.Checkpoint):
        msg = f"expected Checkpoint instance, got {type(checkpoint)}"
        raise TypeError(msg)

    # save weights only
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.model_state_dict, output_path)

    # report size reduction
    original_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
    new_size = output_path.stat().st_size / (1024 * 1024)  # MB
    reduction = (1 - new_size / original_size) * 100

    logger.info("converted: %s -> %s", checkpoint_path, output_path)
    logger.info("size: %.1fMB -> %.1fMB (reduced by %.1f%%)", original_size, new_size, reduction)
