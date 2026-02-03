# ruff: noqa: S101, N806, PLR0915
import logging
import random
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
        X_d, y_d = X.to(ctx.device), y.to(ctx.device)

        # forward
        _, loss = ctx.model.calculate_loss(X_d, y_d)

        # backprop
        ctx.optimizer.zero_grad()
        loss.backward()
        ctx.optimizer.step()

        ctx.loss_history.append(loss.log10().item())
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info("avg loss: %.7f  [%d/%d]", loss, current, size)


def test_lm(ctx: training.Context) -> float:
    num_batches = len(ctx.val_loader)
    ctx.model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(ctx.val_loader):
            X_d, y_d = X.to(ctx.device), y.to(ctx.device)
            _, loss = ctx.model.calculate_loss(X_d, y_d)
            test_loss += loss.item()
    test_loss /= num_batches
    logger.info("avg loss: %.7f", test_loss)
    return test_loss


def train(train_settings: settings.Train, model_settings: settings.Model) -> None:
    # aliases
    batch_size = train_settings.batch_size
    context_size = model_settings.context_size
    tokens_to_generate = train_settings.tokens_to_generate
    tokens_to_save = train_settings.tokens_to_save

    # setup
    random.seed(train_settings.seed)
    torch.manual_seed(train_settings.torch_seed)
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available() and train_settings.use_accelerator
        else torch.device("cpu")
    )
    assert device is not None, "device cannot be None"

    # prepare dataset
    with train_settings.input_file.open("r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = tokenice.CharTokenizer.train([text])

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
    match model_settings:
        case settings.CharBigram():
            model = model_mod.CharBigram(context_size=context_size, vocab_size=tokenizer.vocab_size).to(device)
        case settings.CharTransformer():
            model = model_mod.CharTransformer(
                num_blocks=model_settings.transformer_num_blocks,
                num_heads=model_settings.transformer_num_heads,
                context_size=context_size,
                vocab_size=tokenizer.vocab_size,
                embedding_size=model_settings.transformer_embedding_size,
                ffw_projection_factor=model_settings.transformer_feedforward_projection_factor,
                dropout=model_settings.transformer_dropout,
            ).to(device)
        case settings.GPT2():
            model = model_mod.GPT2(
                context_size=context_size,
                vocab_size=tokenizer.vocab_size,
                embedding_size=model_settings.gpt2_embedding_size,
                num_layers=model_settings.gpt2_num_layers,
                num_heads=model_settings.gpt2_num_heads,
                ffw_projection_factor=model_settings.gpt2_feedforward_projection_factor,
            ).to(device)
        case _:
            assert_never(model_settings)

    optimizer = optim.AdamW(model.parameters(), lr=train_settings.learning_rate)

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
    )

    # load resume checkpoint if exists
    if train_settings.resume_from_checkpoint is not None:
        ctx.load(train_settings.resume_from_checkpoint)
        logger.info("resumed from checkpoint: %s", train_settings.resume_from_checkpoint)

    logger.info("dataset size: %d", len(train_dataset) + len(val_dataset))
    logger.info("train size: %d", len(train_dataset))
    logger.info("val size: %d", len(val_dataset))
    logger.info("batch size: %d", ctx.train_loader.batch_size)
    logger.info("epoch: %d", ctx.epoch)

    logger.info("model type: %s", ctx.model.TYPE)
    logger.info("model info %s", ctx.model)
    logger.info("device: %s", ctx.device)
    logger.info("tokenizer %s", ctx.tokenizer.TYPE)
    logger.info("vocab size: %d", ctx.tokenizer.vocab_size)
    logger.info("context size: %d", context_size)
    logger.info("model parameters: %d", sum(p.numel() for p in ctx.model.parameters()))

    # eval before training
    logger.info("evaluating before training")
    test_lm(ctx)
    logger.info("sampling before training")
    ctx.sample(tokens_to_generate)

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
