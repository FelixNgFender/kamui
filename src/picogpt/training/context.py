import dataclasses
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import optim
from torch.utils import data as data_utils

from picogpt import constants, model, tokenice, training

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
    val_loader: data_utils.DataLoader
    checkpoint_dir: pathlib.Path
    epoch: int
    loss_history: list[float]
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
            model_type=self.model.TYPE,
            epoch=self.epoch,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            loss_history=self.loss_history,
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
        self.epoch = checkpoint.epoch
        self.loss_history = checkpoint.loss_history
        logger.info("checkpoint loaded from %s (epoch %d)", checkpoint_path, self.epoch)

        # ensure all ranks finish loading
        if self.is_ddp:
            dist.barrier()

    def save_loss_plot(self) -> None:
        """Save training loss curve plot."""
        # only master process saves the loss plot
        if not self.is_master_process:
            return

        loss_path = self.checkpoint_dir / constants.LOSS_PLOT_FILENAME
        plt.figure()
        plt.plot(self.loss_history)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.title(f"training loss curve for {self.model.TYPE}")
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
