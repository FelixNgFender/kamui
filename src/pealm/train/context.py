import dataclasses
import logging
import pathlib
import sys
import time
from collections.abc import Iterator, Sized

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import optim
from torch.utils import data as data_utils

from pealm import constants, tokenizer, train
from pealm import model as model_mod

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Context:
    """Run-specific training context holding all state and configuration."""

    # compute
    device: torch.device
    use_mixed_precision: bool

    # ddp
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

    # model
    model: model_mod.LanguageModel
    """Raw model weights."""
    tokenizer: tokenizer.Tokenizer
    optimizer: optim.Optimizer

    # train
    global_batch_size: int
    train_loader: data_utils.DataLoader
    train_iter: Iterator[tuple[torch.Tensor, torch.Tensor]]
    val_loader: data_utils.DataLoader
    step: int
    """1-based step count."""
    max_steps: int
    """1-based max step count for training. Training stops when step exceeds steps."""
    train_loss: list[float]
    val_loss: list[tuple[int, float]]

    # checkpoint
    save_every: int | None
    """Saves checkpoint every `save_every` steps if not None."""
    checkpoint_dir: pathlib.Path
    resume_from_checkpoint: pathlib.Path | None
    """If not None, path to checkpoint to resume training from."""

    # eval
    val_every: int | None
    """Validates every `val_every` steps if not None."""
    tokens_to_save: int
    """Number of tokens to generate and save after training is done. Only master process will save the output."""

    @property
    def training_model(self) -> model_mod.LanguageModel | torch.nn.parallel.DistributedDataParallel:
        """The model to be used for training. Wrapped with DDP if DDP is enabled."""
        if self.is_ddp:
            assert self.ddp_model is not None, "ddp_model should not be None when is_ddp is True"  # noqa: S101
            return self.ddp_model
        return self.model

    def log_status(
        self,
    ) -> None:
        if not self.is_master_process:
            return

        logger.info("rank: %d, device: %s", self.rank, self.device)
        logger.info("global batch size: %d", self.global_batch_size)
        logger.info("rank batch size: %s", self.train_loader.batch_size)
        logger.info("step: %d/%d", self.step, self.max_steps)
        logger.info("model type: %s", self.model.TYPE)
        logger.info("model info:\n%s", self.model)
        logger.info("tokenizer: %s", self.tokenizer.TYPE)
        logger.info("context size: %d", self.model.context_size)
        logger.info("model parameters: %d", sum(p.numel() for p in self.model.parameters()))

        logger.info("automatic mixed precision: %s", self.use_mixed_precision)
        logger.info("float32 matmul precision: %s", torch.get_float32_matmul_precision())
        logger.info("ddp enabled: %s", self.is_ddp)
        logger.info("world size: %d", self.world_size)
        if self.is_ddp:
            logger.info("ddp backend: %s", dist.get_backend())

    def _checkpoint(
        self,
        filename: str | pathlib.Path,
    ) -> None:
        """Save model checkpoint to checkpoint_dir."""
        # only master process saves checkpoints
        if not self.is_master_process:
            return

        checkpoint = train.Checkpoint(
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

    def _load(
        self,
        ckpt: pathlib.Path,
    ) -> None:
        """Load part of the training context from the checkpoint."""
        checkpoint = train.Checkpoint.load(ckpt, map_location=self.device)
        if model_mod.Type(checkpoint.model_type) != self.model.TYPE:
            msg = f"checkpoint model type {checkpoint.model_type} does not match current model type {self.model.TYPE}"
            raise ValueError(msg)

        self.model.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.train_loss = checkpoint.train_loss
        self.val_loss = checkpoint.val_loss
        self.step = checkpoint.step + 1  # start at the next step after the checkpoint
        logger.info("checkpoint loaded from %s (step %d)", ckpt, checkpoint.step)

        # ensure all ranks finish loading
        if self.is_ddp:
            dist.barrier()

    def _plot_losses(self) -> None:
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

    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Called on every training step to get next batch. Handles dataloader exhaustion and restarting."""
        try:
            X, y = next(self.train_iter)
        except StopIteration:
            logger.info("dataloader exhausted before reaching max steps, restarting dataloader")

            # update DDP sampler epoch to shuffle differently every epoch if applicable
            if self.is_ddp and isinstance(self.train_loader.sampler, data_utils.DistributedSampler):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)

            self.train_iter = iter(self.train_loader)
            X, y = next(self.train_iter)
        return X, y

    def train_step(self) -> None:
        """Default training step for all models."""
        # set train mode
        self.training_model.train()
        t0 = time.perf_counter()
        X, y = self._next_batch()
        X_d, y_d = X.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()

        # forward with automatic mixed precision
        # bfloat16 does not require gradient scaling like float16 because
        # it possesses a much wider dynamic range, matching that of float32
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_mixed_precision):
            _, loss = self.training_model(X_d, y_d)

        # backprop
        loss.backward()
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        # can add barrier here to ensure all ranks finish before timing
        # but it's unnecessary overhead
        # let's assume master is the fastest :)

        t1 = time.perf_counter()

        # only master process saves and logs global loss
        loss = loss.detach()
        if self.is_ddp:
            # avoid unnecessary graph communication
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        if self.is_master_process:
            dt = t1 - t0  # seconds
            tps = (X_d.shape[0] * X_d.shape[1] * self.world_size) / dt
            loss = loss.item()
            self.train_loss.append(loss)
            logger.info(
                "step: %7d | loss: %.7f | dt: %6.2fms | tok/sec: %8.2f",
                self.step,
                loss,
                dt * 1000,
                tps,
            )

    def evaluate(self) -> float:
        """Evaluate on the validation set and return the average loss. Only works if val_loader is Sized for correct
        averaging."""
        if not isinstance(self.val_loader, Sized):
            msg = "val_loader must be bounded to use default eval function for correct loss averaging"
            raise TypeError(msg)

        num_batches = len(self.val_loader)
        self.training_model.eval()
        test_loss = torch.tensor(0.0, device=self.device)
        with torch.inference_mode():
            for X, y in self.val_loader:
                X_d, y_d = X.to(self.device), y.to(self.device)
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_mixed_precision
                ):
                    _, loss = self.training_model(X_d, y_d)
                test_loss += loss.detach()
        test_loss /= num_batches

        # aggregates losses across all replicas for correct metric
        if self.is_ddp:
            dist.all_reduce(test_loss, op=dist.ReduceOp.AVG)
        if self.is_master_process:
            logger.info("avg loss: %.7f", test_loss)
        return test_loss.item()

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

    def train_loop(
        self,
    ) -> None:
        """Entrypoint for training loop. Handles checkpointing, validation, and sampling. Can be customized by
        overriding train_step, evaluate, sample, or even train_loop itself."""
        # load resume checkpoint if exists
        if self.resume_from_checkpoint is not None:
            self._load(self.resume_from_checkpoint)

        # log status before training
        self.log_status()

        best_val_loss = float("inf")
        for step in range(self.step, self.max_steps + 1):
            # perform train step
            self.step = step
            self.train_step()

            # optional periodic validation
            if self.val_every is not None and step % self.val_every == 0:
                logger.info("evaluating at step %d", step)
                val_loss = self.evaluate()
                self.val_loss.append((step, val_loss))

                if self.is_master_process and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._checkpoint(constants.BEST_CKPT_FILENAME)
                    logger.info("new best model at step %d with val loss %.7f", step, val_loss)

            # optional periodic checkpointing
            if self.save_every is not None and step % self.save_every == 0:
                self._plot_losses()
                self._checkpoint(
                    f"step_{step}.pt",
                )
                self._checkpoint(constants.LATEST_CKPT_FILENAME)

        # ckpt and sample
        if self.is_master_process:
            self._plot_losses()
            self._checkpoint(
                constants.FINAL_CKPT_FILENAME,
            )
            logger.info("sampling after training")
            self.sample(self.tokens_to_save, save_to_file=True)

        # wait for master process to finish checkpointing and sampling before cleanup
        if self.is_ddp:
            dist.barrier()
