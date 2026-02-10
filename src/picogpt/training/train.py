# ruff: noqa: N806, PLR0913, PLR0912, PLR0915, S101, C901
import atexit
import logging
import math
import os
import random
import time
from collections.abc import Callable
from typing import assert_never

import torch
import torch.distributed as dist
import torch.utils.data as data_utils
from torch import nn, optim

from picogpt import constants, settings, tokenice, training, utils
from picogpt import model as model_mod

logger = logging.getLogger(__name__)


class PicoDataset(data_utils.Dataset):
    def __init__(
        self,
        tokenizer: tokenice.Tokenizer,
        text: str,
        context_size: int,
    ) -> None:
        """Contexts and labels both have dims (batch_size, context_size) e.g.,
        contexts[5, 6] will have the label at labels[5, 6]"""
        self.context_size = context_size
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        # create all possible windows instead of skipping by context_size
        idx = torch.arange(len(data) - self.context_size).view(-1, 1)
        self.contexts = data[idx + torch.arange(self.context_size)]
        self.labels = data[idx + torch.arange(self.context_size) + 1]

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int | list[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # ty:ignore[invalid-method-override]
        return self.contexts[idx], self.labels[idx]


def train_lm(ctx: training.Context) -> None:
    size = len(ctx.train_loader.dataset)  # ty:ignore[invalid-argument-type]
    # set train mode
    ctx.training_model.train()
    for batch, (X, y) in enumerate(ctx.train_loader):
        t0 = time.perf_counter()
        X_d, y_d = X.to(ctx.device), y.to(ctx.device)

        ctx.optimizer.zero_grad()

        # forward with automatic mixed precision
        # bfloat16 does not require gradient scaling like float16 because
        # it possesses a much wider dynamic range, matching that of float32
        with torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16, enabled=ctx.use_mixed_precision):
            _, loss = ctx.training_model(X_d, y_d)

        # backprop
        loss.backward()
        ctx.optimizer.step()
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)

        # can add barrier here to ensure all ranks finish before timing
        # but it's unnecessary overhead
        # let's assume master is the fastest :)

        t1 = time.perf_counter()

        # only master process saves and logs global loss
        loss = loss.detach()
        if ctx.is_ddp:
            # avoid unnecessary graph communication
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        if ctx.is_master_process:
            dt = t1 - t0  # seconds
            tps = (X_d.shape[0] * X_d.shape[1] * ctx.world_size) / dt
            loss = loss.item()
            ctx.loss_history.append(loss)
            if batch % 100 == 0:
                logger.info(
                    "[%7d/%7d] | loss: %.7f | dt: %6.2fms | tok/sec: %8.2f",
                    (batch + 1) * len(X),
                    size,
                    loss,
                    dt * 1000,
                    tps,
                )


def _get_lr(
    step: int,
    *,
    min_lr: float,
    max_lr: float,
    warm_up_steps: int,
    max_steps: int,
) -> float:
    # linear warm up for warm_up_steps
    if step < warm_up_steps:
        return max_lr * (step + 1) / warm_up_steps
    # min lr for step > max_steps
    if step > max_steps:
        return min_lr
    # cosine decay in between
    decay_ratio = (step - warm_up_steps) / (max_steps - warm_up_steps)
    assert 0 <= decay_ratio <= 1, "decay_ratio out of bounds"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 to 0
    return min_lr + coeff * (max_lr - min_lr)


def create_train_gpt2_loop(
    min_lr: float,
    max_lr: float,
    warm_up_steps: int,
    max_steps: int,
    micro_batch_size: int,
    grad_accumulation_steps: int,
) -> Callable[[training.Context], None]:
    """Custom train loop for GPT-2 model"""

    def train_gpt2(ctx: training.Context) -> None:
        size = len(ctx.train_loader.dataset)  # ty:ignore[invalid-argument-type]
        ctx.training_model.train()  # set train mode
        for batch, (X, y) in enumerate(ctx.train_loader):
            t0 = time.perf_counter()

            ctx.optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=ctx.device)
            for micro_step in range(grad_accumulation_steps):
                micro_start = micro_step * micro_batch_size
                micro_end = micro_start + micro_batch_size
                # move only micro batch to device
                X_d, y_d = X[micro_start:micro_end].to(ctx.device), y[micro_start:micro_end].to(ctx.device)

                # forward
                with torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16, enabled=ctx.use_mixed_precision):
                    _, loss = ctx.training_model(X_d, y_d)
                # scale loss for grad accumulation because cross entropy averages over batch size
                # by default, but isn't aware of our micro batching
                loss /= grad_accumulation_steps
                # backprop
                loss.backward()
                total_loss += loss.detach()

            # grad clipping
            norm = nn.utils.clip_grad_norm_(ctx.training_model.parameters(), max_norm=1.0)

            # determine lr for this step
            lr = _get_lr(
                batch,
                max_lr=max_lr,
                min_lr=min_lr,
                max_steps=max_steps,
                warm_up_steps=warm_up_steps,
            )
            # manually apply new lr uniformly to all param groups
            # normally, a pytorch scheduler.step() occurs here
            # but only after optimizer.step(), but we want to change
            # lr before optimizer.step() so it's here
            for param_group in ctx.optimizer.param_groups:
                param_group["lr"] = lr

            # update only once per full batch, after all micro batches
            ctx.optimizer.step()

            if ctx.device.type == "cuda":
                torch.cuda.synchronize(ctx.device)

            # can add barrier here to ensure all ranks finish before timing
            # but it's unnecessary overhead
            # let's assume master is the fastest :)

            t1 = time.perf_counter()

            if ctx.is_ddp:
                dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            # only master process saves and logs loss
            if ctx.is_master_process:
                dt = t1 - t0  # seconds
                tps = (X.shape[0] * X.shape[1] * ctx.world_size) / dt
                loss = total_loss.item()
                ctx.loss_history.append(loss)
                logger.info(
                    "[%7d/%7d] | loss: %.7f | lr: %.4e | norm: %.4f | dt: %6.2fms | tok/sec: %8.2f",
                    (batch + 1) * len(X),
                    size,
                    loss,
                    lr,
                    norm,
                    dt * 1000,
                    tps,
                )

    return train_gpt2


def test_lm(ctx: training.Context) -> float:
    num_batches = len(ctx.val_loader)
    ctx.training_model.eval()
    test_loss = torch.tensor(0.0, device=ctx.device)
    with torch.inference_mode():
        for _, (X, y) in enumerate(ctx.val_loader):
            X_d, y_d = X.to(ctx.device), y.to(ctx.device)
            with torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16, enabled=ctx.use_mixed_precision):
                _, loss = ctx.training_model(X_d, y_d)
            test_loss += loss.detach()
    test_loss /= num_batches

    # aggregates losses across all replicas for correct metric
    if ctx.is_ddp:
        dist.all_reduce(test_loss, op=dist.ReduceOp.AVG)
    if ctx.is_master_process:
        logger.info("avg loss: %.7f", test_loss)
    return test_loss.item()


def train(train_settings: settings.Train, model_settings: settings.Model) -> None:
    # aliases
    batch_size = train_settings.batch_size
    context_size = model_settings.context_size
    tokens_to_save = train_settings.tokens_to_save
    rank, local_rank, world_size = (
        train_settings.ddp.rank,
        train_settings.ddp.local_rank,
        train_settings.ddp.world_size,
    )

    # device
    if train_settings.ddp.enabled:
        if not torch.cuda.is_available():
            msg = "DDP training requires CUDA, but no CUDA devices are available."
            raise RuntimeError(msg)

        # process affinity: set device before doing any cuda operations
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        # init process group
        backend = dist.get_default_backend_for_device(device)
        # environment variables which need to be
        # set when using c10d's default "env"
        # initialization mode.
        os.environ["MASTER_ADDR"] = train_settings.ddp.master_addr
        os.environ["MASTER_PORT"] = str(train_settings.ddp.master_port)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # attach cleanup so we don't forget
        atexit.register(dist.destroy_process_group)
    else:
        device = (
            torch.accelerator.current_accelerator()
            if torch.accelerator.is_available() and train_settings.use_accelerator
            else torch.device("cpu")
        )
        assert device is not None, "device cannot be None"

    # setup
    random.seed(train_settings.seed)
    torch.manual_seed(train_settings.torch_seed)
    torch.cuda.manual_seed_all(train_settings.torch_seed)
    # tells pytorch to use different kernels depending on precision
    torch.set_float32_matmul_precision(train_settings.fp32_matmul_precision)

    # prepare model and dataset
    with train_settings.input_file.open("r", encoding="utf-8") as f:
        text = f.read()

    match model_settings:
        case settings.CharBigram():
            tokenizer = tokenice.CharTokenizer.train([text])
            vocab_size = tokenizer.vocab_size
            model = model_mod.CharBigram(context_size=context_size, vocab_size=vocab_size)
            optimizer = optim.AdamW(model.parameters(), lr=train_settings.learning_rate)
        case settings.CharTransformer():
            tokenizer = tokenice.CharTokenizer.train([text])
            vocab_size = tokenizer.vocab_size
            model = model_mod.CharTransformer(
                context_size=context_size,
                vocab_size=vocab_size,
                embedding_size=model_settings.embedding_size,
                num_blocks=model_settings.num_blocks,
                num_heads=model_settings.num_heads,
                ffw_projection_factor=model_settings.feedforward_projection_factor,
                dropout=model_settings.dropout,
            )
            optimizer = optim.AdamW(model.parameters(), lr=train_settings.learning_rate)
        case settings.GPT2():
            tokenizer = tokenice.GPT2Tokenizer()
            # don't use tokenizer.vocab_size for GPT2 cuz we want 50304 for cuda niceness
            vocab_size = model_settings.vocab_size
            model = model_mod.GPT2(
                context_size=context_size,
                vocab_size=vocab_size,
                embedding_size=model_settings.embedding_size,
                num_layers=model_settings.num_layers,
                num_heads=model_settings.num_heads,
                ffw_projection_factor=model_settings.feedforward_projection_factor,
            )
            # follow gpt-3 hparams
            optimizer = model.create_optimizer(model_settings.weight_decay, train_settings.learning_rate, device)
            # account for ddp
            grad_accumulation_steps = batch_size // (model_settings.micro_batch_size * world_size)
            # use custom train loop for gpt2
            train_lm = create_train_gpt2_loop(
                model_settings.min_lr,
                model_settings.max_lr,
                model_settings.warmup_steps,
                model_settings.max_steps,
                model_settings.micro_batch_size,
                grad_accumulation_steps,
            )
            if train_settings.ddp.is_master_process:
                logger.info(
                    "micro batch size: %d, calculated grad accumulation steps: %d",
                    model_settings.micro_batch_size,
                    grad_accumulation_steps,
                )
        case _:
            assert_never(model_settings)

    # move and then compile so compiler don't have to reason about device-specific copies
    model.to(device)
    # https://docs.pytorch.org/docs/main/notes/ddp#example
    # DDP works with TorchDynamo. When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
    # such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes
    if not train_settings.ddp.enabled:
        ddp_model = None
        model.compile()
    else:
        # ddp_local_rank is the gpu id that the model lives on
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            # gradient_as_bucket_view=True to reduce GPU memory fragmentation
            # and slightly improve performance
            model,
            device_ids=[train_settings.ddp.local_rank],
            gradient_as_bucket_view=True,
        )
        ddp_model.compile()

    n1 = int(train_settings.train_split * len(text))
    train_dataset = PicoDataset(
        tokenizer,
        text[:n1],
        context_size,
    )
    val_dataset = PicoDataset(
        tokenizer,
        text[n1:],
        context_size,
    )

    if train_settings.ddp.enabled:
        # sampler is mutual exclusive with shuffle in dataloader, so we set shuffle to False
        # and use sampler for shuffling
        train_sampler = data_utils.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = data_utils.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = data_utils.DataLoader(
        train_dataset, batch_size, shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=True
    )
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size, shuffle=False, sampler=val_sampler, pin_memory=True)

    # create training context
    run_checkpoint_dir = train_settings.checkpoint_dir / model.TYPE / utils.get_current_datetime()
    if train_settings.ddp.is_master_process:
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(run_checkpoint_dir / constants.TOKENIZER_DIR)

    ctx = training.Context(
        model=model,
        optimizer=optimizer,
        device=device,
        tokenizer=tokenizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        checkpoint_dir=run_checkpoint_dir,
        epoch=1,
        loss_history=[],
        use_mixed_precision=train_settings.use_mixed_precision,
        is_ddp=train_settings.ddp.enabled,
        ddp_model=ddp_model,
        rank=rank,
        world_size=world_size,
        is_master_process=train_settings.ddp.is_master_process,
    )

    # load resume checkpoint if exists
    if train_settings.resume_from_checkpoint is not None:
        ctx.load(train_settings.resume_from_checkpoint)

    if ctx.is_master_process:
        logger.info("dataset size: %d", len(train_dataset) + len(val_dataset))
        logger.info("train size: %d", len(train_dataset))
        logger.info("val size: %d", len(val_dataset))
        logger.info("batch size: %d", ctx.train_loader.batch_size)
        logger.info("epoch: %d/%d", ctx.epoch, train_settings.num_epochs)

        logger.info("model type: %s", ctx.model.TYPE)
        logger.info("model info:\n%s", ctx.model)
        logger.info("tokenizer: %s", ctx.tokenizer.TYPE)
        logger.info("vocab size: %d", vocab_size)
        logger.info("context size: %d", context_size)
        logger.info("model parameters: %d", sum(p.numel() for p in ctx.model.parameters()))

        logger.info("automatic mixed precision: %s", ctx.use_mixed_precision)
        logger.info("float32 matmul precision: %s", torch.get_float32_matmul_precision())
        logger.info("ddp enabled: %s", ctx.is_ddp)
        logger.info("world size: %d", ctx.world_size)
        if ctx.is_ddp:
            logger.info("ddp backend: %s", dist.get_backend())

    logger.info("rank: %d, device: %s", ctx.rank, ctx.device)

    start_epoch = ctx.epoch
    max_epochs = start_epoch + train_settings.num_epochs
    best_val_loss = float("inf")
    for epoch in range(start_epoch, max_epochs):
        if ctx.is_master_process:
            logger.info("starting epoch %d/%d", epoch, max_epochs - 1)

        # shuffle data differently every epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_lm(ctx)
        val_loss = test_lm(ctx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ctx.checkpoint(constants.BEST_CKPT_FILENAME)
            if ctx.is_master_process:
                logger.info("new best model at epoch %d with val loss %.7f", epoch, val_loss)

        if epoch % train_settings.save_every_n_epochs == 0:
            ctx.checkpoint(
                f"epoch_{epoch}.pt",
            )
            ctx.checkpoint(constants.LATEST_CKPT_FILENAME)

        ctx.epoch += 1
        ctx.save_loss_plot()

    ctx.checkpoint(
        constants.FINAL_CKPT_FILENAME,
    )

    # sample
    if ctx.is_master_process:
        logger.info("sampling after training")
        ctx.sample(tokens_to_save, save_to_file=True)

    # wait for master process to finish checkpointing and sampling before cleanup
    if ctx.is_ddp:
        dist.barrier()
