import dataclasses
import logging
import math
import time
from typing import override

import torch
import torch.distributed as dist
import torch.utils.data as data_utils
from torch import nn

from pealm import constants, dataset, model, settings, tokenizer, train, utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GPT2Context(train.Context):
    """Custom GPT-2 training context"""

    # train
    micro_batch_size: int
    """Each micro-batch is a subset of the full batch that fits in memory for gradient accumulation"""
    grad_accumulation_steps: int
    """The number of micro-batches to accumulate before each optimizer step"""

    # lr schedule
    max_lr: float
    """The peak learning rate to use after linear warmup and during cosine decay"""
    min_lr: float
    """The minimum learning rate to use after cosine decay"""
    max_lr_steps: int
    """The number of steps to decay from max_lr to min_lr. After this, min_lr is used for the rest of training"""
    warmup_lr_steps: int
    """The number of steps to linearly increase the learning rate from 0 to max_lr"""

    @override
    def log_status(self) -> None:
        super().log_status()
        # account for ddp
        if self.is_master_process:
            logger.info(
                "micro batch size: %d, calculated grad accumulation steps: %d",
                self.micro_batch_size,
                self.grad_accumulation_steps,
            )

    def _get_lr(
        self,
        it: int,
    ) -> float:
        """Get the learning rate for this step. Note that iteration here is 0-based, which is step - 1."""
        # linear warm up
        if it < self.warmup_lr_steps:
            return self.max_lr * (it + 1) / self.warmup_lr_steps
        # min lr for step > max_lr_steps
        if it > self.max_lr_steps:
            return self.min_lr
        # cosine decay in between
        decay_ratio = (it - self.warmup_lr_steps) / (self.max_lr_steps - self.warmup_lr_steps)
        assert 0 <= decay_ratio <= 1, "decay_ratio out of bounds"  # noqa: S101
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    @override
    def train_step(self) -> None:
        """GPT2 uses a custom training step to handle grad accumulation and custom learning rate schedule."""
        # set train mode
        self.training_model.train()
        t0 = time.perf_counter()

        X, y = self._next_batch()
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        for micro_step in range(self.grad_accumulation_steps):
            micro_start = micro_step * self.micro_batch_size
            micro_end = micro_start + self.micro_batch_size
            # move only micro batch to device
            X_d, y_d = X[micro_start:micro_end].to(self.device), y[micro_start:micro_end].to(self.device)

            # when ddp, only sync gradients on the last micro step
            if self.is_ddp:
                self.training_model.require_backward_grad_sync = micro_step == (self.grad_accumulation_steps - 1)  # ty:ignore[invalid-assignment]

            # forward
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_mixed_precision):
                _, loss = self.training_model(X_d, y_d)
            # scale loss for grad accumulation because cross entropy averages over batch size
            # by default, but isn't aware of our micro batching
            loss /= self.grad_accumulation_steps
            # backprop
            loss.backward()
            total_loss += loss.detach()

        # grad clipping
        norm = nn.utils.clip_grad_norm_(self.training_model.parameters(), max_norm=1.0)

        # determine lr for this step
        lr = self._get_lr(self.step - 1)
        # manually apply new lr uniformly to all param groups
        # normally, a pytorch scheduler.step() occurs here
        # but only after optimizer.step(), but we want to change
        # lr before optimizer.step() so it's here
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # update only once per full batch, after all micro batches
        self.optimizer.step()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        # can add barrier here to ensure all ranks finish before timing
        # but it's unnecessary overhead
        # let's assume master is the fastest :)

        t1 = time.perf_counter()

        if self.is_ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        # only master process saves and logs loss
        if self.is_master_process:
            dt = t1 - t0  # seconds
            tps = (X.shape[0] * X.shape[1] * self.world_size) / dt
            loss = total_loss.item()
            self.train_loss.append(loss)
            logger.info(
                "step: %7d | loss: %.7f | lr: %.4e | norm: %.4f | dt: %6.2fms | tok/sec: %8.2f",
                self.step,
                loss,
                lr,
                norm,
                dt * 1000,
                tps,
            )

    @override
    def evaluate(self) -> float:
        """
        Custom evaluation loop for GPT-2 that handles micro batching and aggregates loss across batches and replicas.
        """
        self.training_model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        with torch.inference_mode():
            for _, (X, y) in enumerate(self.val_loader):
                for micro_step in range(self.grad_accumulation_steps):
                    micro_start = micro_step * self.micro_batch_size
                    micro_end = micro_start + self.micro_batch_size
                    X_d, y_d = X[micro_start:micro_end].to(self.device), y[micro_start:micro_end].to(self.device)

                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_mixed_precision
                    ):
                        _, loss = self.training_model(X_d, y_d)

                    loss /= self.grad_accumulation_steps
                    total_loss += loss.detach()
                num_batches += 1

        total_loss /= num_batches
        # aggregates losses across all replicas for correct metric
        if self.is_ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        if self.is_master_process:
            logger.info("avg loss: %.7f", total_loss)
        return total_loss.item()


def train_gpt2(train_settings: settings.TrainGPT2, model_settings: settings.GPT2) -> None:
    # aliases
    batch_size = train_settings.batch_size
    context_size = model_settings.context_size
    rank, local_rank, world_size = (
        train_settings.ddp.rank,
        train_settings.ddp.local_rank,
        train_settings.ddp.world_size,
    )
    # each rank processes a subset of the effective batch
    rank_batch_size = batch_size // world_size
    grad_accumulation_steps = rank_batch_size // train_settings.micro_batch_size

    device = utils.compute_init(
        use_accelerator=train_settings.use_accelerator,
        seed=train_settings.seed,
        torch_seed=train_settings.torch_seed,
        fp32_matmul_precision=train_settings.fp32_matmul_precision,
        ddp_enabled=train_settings.ddp.enabled,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
    )

    # prepare dataset and model
    _tokenizer = tokenizer.GPT2Tokenizer()
    train_dataset = dataset.ShardedNpy(
        train_settings.input_dir,
        split="train",
        context_size=context_size,
        batch_size=rank_batch_size,
        rank=rank,
        world_size=world_size,
        seed=train_settings.torch_seed,
        shuffle=True,
    )
    val_dataset = dataset.ShardedNpy(
        train_settings.input_dir,
        split="val",
        context_size=context_size,
        batch_size=rank_batch_size,
        rank=rank,
        world_size=world_size,
    )
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=None, pin_memory=True)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=None, pin_memory=True)

    _model = model.GPT2(
        context_size=context_size,
        # don't use tokenizer.vocab_size for GPT2 cuz we want 50304 for cuda niceness
        vocab_size=model_settings.vocab_size,
        embedding_size=model_settings.embedding_size,
        num_layers=model_settings.num_layers,
        num_heads=model_settings.num_heads,
    )
    # follow gpt-3 hparams
    optimizer = _model.create_optimizer(train_settings.weight_decay, train_settings.lr, device)

    # move and then compile so compiler don't have to reason about device-specific copies
    _model.to(device)
    # https://docs.pytorch.org/docs/main/notes/ddp#example
    # DDP works with TorchDynamo. When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
    # such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes
    if not train_settings.ddp.enabled:
        ddp_model = None
        _model.compile()
    else:
        # ddp_local_rank is the gpu id that the model lives on
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            # gradient_as_bucket_view=True to reduce GPU memory fragmentation
            # and slightly improve performance
            _model,
            device_ids=[train_settings.ddp.local_rank],
            gradient_as_bucket_view=True,
        )
        ddp_model.compile()

    # create training context
    run_checkpoint_dir = train_settings.ckpt_dir / _model.TYPE / utils.current_dt()
    if train_settings.ddp.is_master_process:
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _tokenizer.save(run_checkpoint_dir / constants.TOKENIZER_DIR)

    ctx = GPT2Context(
        device=device,
        use_mixed_precision=train_settings.use_mixed_precision,
        is_ddp=train_settings.ddp.enabled,
        ddp_model=ddp_model,
        rank=rank,
        world_size=world_size,
        is_master_process=train_settings.ddp.is_master_process,
        model=_model,
        tokenizer=_tokenizer,
        optimizer=optimizer,
        global_batch_size=batch_size,
        train_loader=train_dataloader,
        train_iter=iter(train_dataloader),
        val_loader=val_dataloader,
        step=1,
        max_steps=train_settings.steps,
        train_loss=[],
        val_loss=[],
        save_every=train_settings.save_every,
        checkpoint_dir=run_checkpoint_dir,
        resume_from_checkpoint=train_settings.resume_from_ckpt,
        tokens_to_save=train_settings.tokens_to_save,
        val_every=train_settings.val_every,
        # custom
        micro_batch_size=train_settings.micro_batch_size,
        grad_accumulation_steps=grad_accumulation_steps,
        max_lr=train_settings.max_lr,
        min_lr=train_settings.min_lr,
        max_lr_steps=train_settings.max_lr_steps,
        warmup_lr_steps=train_settings.warmup_lr_steps,
    )

    ctx.train_loop()
