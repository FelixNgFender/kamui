import dataclasses
import logging
import pathlib
import sys
from collections.abc import Iterator

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import optim
from torch.utils import data as data_utils

from pealm import constants, model, tokenice, training

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Context:
    """Run-specific training context holding all state and configuration."""

    device: torch.device
    model: model.LanguageModel
    """Raw model weights."""
    optimizer: optim.Optimizer
    tokenizer: tokenice.Tokenizer
    train_loader: data_utils.DataLoader
    train_iter: Iterator[tuple[torch.Tensor, torch.Tensor]]
    val_loader: data_utils.DataLoader
    checkpoint_dir: pathlib.Path
    step: int
    """1-based step count."""
    train_loss: list[float]
    val_loss: list[tuple[int, float]]
    use_mixed_precision: bool
    is_ddp: bool
    """Whether distributed data parallel (DDP) is used for training. False if single-process training."""
    ddp_model: torch.nn.parallel.DistributedDataParallel | None
    """DDP wrapper for `model` if DDP is used."""
    rank: int
    """Global process rank for distributed training. 0 for master process."""
    world_size: int
    """Total number of processes for distributed training. 1 for single-process training."""
    is_master_process: bool
    """Whether this context is running on the master process."""

    @property
    def training_model(self) -> model.LanguageModel | torch.nn.parallel.DistributedDataParallel:
        """The model to be used for training. Wrapped with DDP if DDP is enabled."""
        if self.is_ddp:
            assert self.ddp_model is not None, "ddp_model should not be None when is_ddp is True"  # noqa: S101
            return self.ddp_model
        return self.model

    def checkpoint(
        self,
        filename: str | pathlib.Path,
    ) -> None:
        """Save model checkpoint to checkpoint_dir."""
        # only master process saves checkpoints
        if not self.is_master_process:
            return

        checkpoint = training.Checkpoint(
            step=self.step,
            model_type=self.model.TYPE,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            train_loss=self.train_loss,
            val_loss=self.val_loss,
        )
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info("checkpoint saved: %s", checkpoint_path)

    def load(
        self,
        checkpoint_path: pathlib.Path,
    ) -> None:
        """Load part of the training context from the checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, training.Checkpoint):
            msg = f"invalid checkpoint format: {type(checkpoint)}"
            raise TypeError(msg)
        if model.Type(checkpoint.model_type) != self.model.TYPE:
            msg = f"checkpoint model type {checkpoint.model_type} does not match current model type {self.model.TYPE}"
            raise ValueError(msg)

        self.model.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.train_loss = checkpoint.train_loss
        self.val_loss = checkpoint.val_loss
        self.step = checkpoint.step + 1  # start at the next step after the checkpoint
        logger.info("checkpoint loaded from %s (step %d)", checkpoint_path, checkpoint.step)

        # ensure all ranks finish loading
        if self.is_ddp:
            dist.barrier()

    def plot_losses(self) -> None:
        """Save training and validation loss curve plot."""
        # only master process saves the loss plot
        if not self.is_master_process:
            return

        loss_path = self.checkpoint_dir / constants.LOSS_PLOT_FILENAME
        plt.figure()
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label="train_loss")
        if self.val_loss:
            val_steps, val_losses = zip(*self.val_loss, strict=False)
            plt.plot(val_steps, val_losses, label="val_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.title(f"training and validation loss curve for {self.model.TYPE}")
        plt.legend()
        plt.savefig(loss_path)
        plt.close()

        logger.info("loss plot saved to %s", loss_path)

    @torch.inference_mode()
    def sample(self, tokens_to_generate: int, *, save_to_file: bool = False) -> None:
        # only master process samples and saves the output
        if not self.is_master_process:
            return

        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        out = self.tokenizer.decode(self.model.generate(context, max_new_tokens=tokens_to_generate)[0].tolist())
        if save_to_file:
            with (self.checkpoint_dir / constants.SAMPLE_OUTPUT_FILENAME).open("w", encoding="utf-8") as f:
                print(out, file=f)
                logger.info("sample saved to %s", f.name)
        else:
            print(out, file=sys.stdout)  # noqa: T201
