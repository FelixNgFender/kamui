# ruff: noqa: N806, PLR0915
import logging
import random
import time
from typing import assert_never

import torch
import torch.utils.data as data_utils
from torch import optim

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
            logger.info("avg loss: %.7f  [%d/%d], dt: %.2fms, tok/sec: %.2f", loss, current, size, dt, tps)


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
    assert device is not None, "device cannot be None"  # noqa: S101

    # prepare model and dataset
    with train_settings.input_file.open("r", encoding="utf-8") as f:
        text = f.read()

    match model_settings:
        case settings.CharBigram():
            tokenizer = tokenice.CharTokenizer.train([text])
            vocab_size = tokenizer.vocab_size
            model = model_mod.CharBigram(context_size=context_size, vocab_size=vocab_size)
        case settings.CharTransformer():
            tokenizer = tokenice.CharTokenizer.train([text])
            vocab_size = tokenizer.vocab_size
            model = model_mod.CharTransformer(
                num_blocks=model_settings.transformer_num_blocks,
                num_heads=model_settings.transformer_num_heads,
                context_size=context_size,
                vocab_size=vocab_size,
                embedding_size=model_settings.transformer_embedding_size,
                ffw_projection_factor=model_settings.transformer_feedforward_projection_factor,
                dropout=model_settings.transformer_dropout,
            )
        case settings.GPT2():
            tokenizer = tokenice.GPT2Tokenizer()
            # don't use tokenizer.vocab_size for GPT2 cuz we want 50304 for cuda niceness
            vocab_size = model_settings.gpt2_vocab_size
            model = model_mod.GPT2(
                context_size=context_size,
                vocab_size=vocab_size,
                embedding_size=model_settings.gpt2_embedding_size,
                num_layers=model_settings.gpt2_num_layers,
                num_heads=model_settings.gpt2_num_heads,
                ffw_projection_factor=model_settings.gpt2_feedforward_projection_factor,
            )
        case _:
            assert_never(model_settings)

    # move and then compile so compiler don't have to reason about device-specific copies
    model.to(device).compile()
    optimizer = optim.AdamW(model.parameters(), lr=train_settings.learning_rate)

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
