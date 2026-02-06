# ruff: noqa: N806, PLR0915, S101
import logging
import math
import random
import time
from collections.abc import Callable
from typing import assert_never

import torch
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
    ctx.model.train()  # set train mode
    for batch, (X, y) in enumerate(ctx.train_loader):
        t0 = time.perf_counter()
        X_d, y_d = X.to(ctx.device), y.to(ctx.device)

        ctx.optimizer.zero_grad()

        # forward with automatic mixed precision
        # bfloat16 does not require gradient scaling like float16 because
        # it possesses a much wider dynamic range, matching that of float32
        with torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16, enabled=ctx.use_mixed_precision):
            loss = ctx.model.estimate_loss(X_d, y_d)

        # backprop
        loss.backward()
        ctx.optimizer.step()
        torch.cuda.synchronize(ctx.device) if ctx.device.type == "cuda" else None
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000  # ms
        tps = (ctx.train_loader.batch_size * X_d.shape[1]) / (t1 - t0)

        ctx.loss_history.append(loss.log10().item())
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(
                "[%7d/%7d] | loss: %.7f | dt: %6.2fms | tok/sec: %8.2f",
                current,
                size,
                loss,
                dt,
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
    min_lr: float, max_lr: float, warm_up_steps: int, max_steps: int
) -> Callable[[training.Context], None]:
    """Custom train loop for GPT-2 model"""

    def train_gpt2(ctx: training.Context) -> None:
        size = len(ctx.train_loader.dataset)  # ty:ignore[invalid-argument-type]
        ctx.model.train()  # set train mode
        for batch, (X, y) in enumerate(ctx.train_loader):
            t0 = time.perf_counter()
            X_d, y_d = X.to(ctx.device), y.to(ctx.device)

            ctx.optimizer.zero_grad()

            # forward
            with torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16, enabled=ctx.use_mixed_precision):
                loss = ctx.model.estimate_loss(X_d, y_d)

            # backprop
            loss.backward()

            # grad clipping
            norm = nn.utils.clip_grad_norm_(ctx.model.parameters(), max_norm=1.0)

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

            ctx.optimizer.step()
            torch.cuda.synchronize(ctx.device) if ctx.device.type == "cuda" else None

            t1 = time.perf_counter()
            dt = (t1 - t0) * 1000  # ms
            tps = (ctx.train_loader.batch_size * X_d.shape[1]) / (t1 - t0)

            ctx.loss_history.append(loss.log10().item())
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                logger.info(
                    "[%7d/%7d] | loss: %.7f | lr: %.4e | norm: %.4f | dt: %6.2fms | tok/sec: %8.2f",
                    current,
                    size,
                    loss,
                    lr,
                    norm,
                    dt,
                    tps,
                )

    return train_gpt2


def test_lm(ctx: training.Context) -> float:
    num_batches = len(ctx.val_loader)
    ctx.model.eval()
    test_loss = 0
    with torch.inference_mode():
        for _, (X, y) in enumerate(ctx.val_loader):
            X_d, y_d = X.to(ctx.device), y.to(ctx.device)
            with torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16, enabled=ctx.use_mixed_precision):
                loss = ctx.model.estimate_loss(X_d, y_d)
            test_loss += loss.item()
    test_loss /= num_batches
    logger.info("avg loss: %.7f", test_loss)
    return test_loss


def train(train_settings: settings.Train, model_settings: settings.Model) -> None:
    # aliases
    batch_size = train_settings.batch_size
    context_size = model_settings.context_size
    tokens_to_save = train_settings.tokens_to_save

    # setup
    random.seed(train_settings.seed)
    torch.manual_seed(train_settings.torch_seed)
    torch.cuda.manual_seed_all(train_settings.torch_seed)
    # tells pytorch to use different kernels depending on precision
    torch.set_float32_matmul_precision(train_settings.fp32_matmul_precision)
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available() and train_settings.use_accelerator
        else torch.device("cpu")
    )
    assert device is not None, "device cannot be None"

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
            # use custom train loop for gpt2
            train_lm = create_train_gpt2_loop(
                model_settings.min_lr,
                model_settings.max_lr,
                model_settings.warmup_steps,
                model_settings.max_steps,
            )
        case _:
            assert_never(model_settings)

    # move and then compile so compiler don't have to reason about device-specific copies
    model.to(device).compile()

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

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size)

    # create training context
    run_checkpoint_dir = train_settings.checkpoint_dir / model.TYPE / utils.get_current_datetime()
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
        use_mixed_precision=train_settings.use_mixed_precision,
    )

    # load resume checkpoint if exists
    if train_settings.resume_from_checkpoint is not None:
        ctx.load(train_settings.resume_from_checkpoint)
        logger.info("resumed from checkpoint: %s", train_settings.resume_from_checkpoint)

    logger.info("dataset size: %d", len(train_dataset) + len(val_dataset))
    logger.info("train size: %d", len(train_dataset))
    logger.info("val size: %d", len(val_dataset))
    logger.info("batch size: %d", ctx.train_loader.batch_size)
    logger.info("epoch: %d/%d", ctx.epoch, train_settings.num_epochs)

    logger.info("model type: %s", ctx.model.TYPE)
    logger.info("model info %s", ctx.model)
    logger.info("device: %s", ctx.device)
    logger.info("tokenizer %s", ctx.tokenizer.TYPE)
    logger.info("vocab size: %d", vocab_size)
    logger.info("context size: %d", context_size)
    logger.info("model parameters: %d", sum(p.numel() for p in ctx.model.parameters()))

    logger.info("automatic mixed precision: %s", ctx.use_mixed_precision)
    logger.info("float32 matmul precision: %s", torch.get_float32_matmul_precision())

    start_epoch = ctx.epoch
    max_epochs = start_epoch + train_settings.num_epochs
    best_val_loss = float("inf")
    for epoch in range(start_epoch, max_epochs):
        logger.info("starting epoch %d/%d", epoch, max_epochs - 1)
        train_lm(ctx)
        val_loss = test_lm(ctx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ctx.checkpoint(constants.BEST_CKPT_FILENAME)
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

    # eval after training
    logger.info("evaluating after training")
    test_lm(ctx)
    # sample
    logger.info("sampling after training")
    ctx.sample(tokens_to_save, save_to_file=True)
