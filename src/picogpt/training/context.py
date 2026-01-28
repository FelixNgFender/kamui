import dataclasses
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils import data as data_utils

from picogpt import constants, model, tokenizers, training

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Context:
    """Run-specific training context holding all state and configuration."""

    model_type: model.Type
    model: model.LanguageModel
    optimizer: optim.Optimizer
    device: torch.device
    tokenizer: tokenizers.Tokenizer
    train_loader: data_utils.DataLoader
    val_loader: data_utils.DataLoader
    checkpoint_dir: pathlib.Path
    epoch: int = 1
    loss_history: list[float] = dataclasses.field(default_factory=list)

    def checkpoint(
        self,
        filename: str | pathlib.Path,
    ) -> None:
        """Save model checkpoint to checkpoint_dir."""
        checkpoint = training.Checkpoint(
            model_type=self.model_type,
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

        self.model_type = model.Type(checkpoint.model_type)
        self.model.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.epoch = checkpoint.epoch
        self.loss_history = checkpoint.loss_history
        logger.info("checkpoint loaded from %s (epoch %d)", checkpoint_path, self.epoch)

    def save_loss_plot(self) -> None:
        """Save training loss curve plot."""

        loss_path = self.checkpoint_dir / constants.LOSS_PLOT_FILENAME
        plt.figure()
        plt.plot(self.loss_history)
        plt.xlabel("iteration")
        plt.ylabel("log10 loss")
        plt.title(f"training loss curve for {self.model_type}")
        plt.savefig(loss_path)
        plt.close()
        logger.info("loss plot saved to %s", loss_path)

    @torch.no_grad()
    def sample(self, tokens_to_generate: int, *, save_to_file: bool = False) -> None:
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        out = self.tokenizer.decode(self.model.generate(context, max_new_tokens=tokens_to_generate)[0].tolist())
        if save_to_file:
            with pathlib.Path(self.checkpoint_dir / constants.SAMPLE_OUTPUT_FILENAME).open("w", encoding="utf-8") as f:
                print(out, file=f)
                logger.info("sample saved to %s", f.name)
        else:
            print(out, file=sys.stdout)  # noqa: T201
