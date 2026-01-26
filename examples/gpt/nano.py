# ruff: noqa: INP001, N812, S101, N806, T201, PLR0915
import enum
import pathlib
import random
from typing import Protocol, assert_never

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import nn, optim


class ModelType(enum.StrEnum):
    CHAR_BIGRAM = enum.auto()
    WORD_MLP = enum.auto()
    CHAR_TRANSFORMER = enum.auto()


USE_ACCELERATOR = True
TORCH_SEED = 2147483647
SEED = 42
INPUT_FILE = "tinyshakespeare.txt"
MODEL_TYPE: ModelType = ModelType.CHAR_TRANSFORMER


TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.1
assert TRAIN_SPLIT + VAL_SPLIT == 1.0

# hyperparams
CONTEXT_SIZE = 256  # the maximum length of predictions
EMBEDDING_SIZE = 384
TRANSFORMER_NUM_HEADS = 6
TRANSFORMER_NUM_BLOCKS = 6
FEEDFORWARD_PROJECTION_FACTOR = 4
DROPOUT = 0.2
# training
BATCH_SIZE = 64  # the number of independent sequences to process at once
LEARNING_RATE = 3e-4
NUM_EPOCHS = 2

random.seed(SEED)
torch.manual_seed(TORCH_SEED)

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available() and USE_ACCELERATOR
    else torch.device("cpu")
)

with pathlib.Path(INPUT_FILE).open("r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
itos = chars
stoi = {ch: i for i, ch in enumerate(chars)}
words = sorted(set(text.split()))
wtoi = {word: i for i, word in enumerate(words)}
itow = words


class Encoder(Protocol):
    def __call__(self, s: str) -> list[int]: ...


class Decoder(Protocol):
    def __call__(self, indices: list[int]) -> str: ...


def encode_char(s: str) -> list[int]:
    """Encodes each character to an index based on the vocab."""
    return [stoi[c] for c in s]


def decode_char(indices: list[int]) -> str:
    """Decode each index back to a character based on the vocab."""
    return "".join([itos[idx] for idx in indices])


def encode_word(s: str) -> list[int]:
    """Encodes each word to an index based on the vocab."""
    return [wtoi[word] for word in s.split()]


def decode_word(indices: list[int]) -> str:
    """Decode each index back to a word based on the vocab."""
    return " ".join([itow[idx] for idx in indices])


# build the dataset
class TinyShakespeare(data_utils.Dataset):
    def __init__(self, text: str, context_size: int, encoder: Encoder) -> None:
        """Contexts and labels both have dims (batch_size, context_size) e.g.,
        contexts[5, 6] will have the label at labels[5, 6]"""
        self.context_size = context_size
        data = torch.tensor(encoder(text), dtype=torch.long)
        # create all possible windows instead of skipping by CONTEXT_SIZE
        idx = torch.arange(len(data) - CONTEXT_SIZE).view(-1, 1)
        self.contexts = data[idx + torch.arange(self.context_size)]
        self.labels = data[idx + torch.arange(self.context_size) + 1]

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int | list[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # ty:ignore[invalid-method-override]
        return self.contexts[idx], self.labels[idx]


class LanguageModel(nn.Module):
    def __init__(self, context_size: int) -> None:
        super().__init__()
        self.context_size = context_size

    def calculate_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """idx and targets both have shape (B, T)"""
        logits = self(idx)

        # batch size x time steps x channels/embeddings/features
        # instead of batch x time x channels, we group batch x time so the rows
        # become individual embeddings at each time step (essentially treating
        # them as individual examples)
        logits = logits.flatten(0, 1)
        # corresponding with each embedding row is a single label index, so we
        # stretch out targets
        targets = targets.flatten()
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
        """idx is (B, T) array of indices in the current context"""
        for _ in range(max_new_tokens):
            # crop idx  to the last context_size tokens
            idx_cropped = idx[:, -self.context_size :]
            logits = self(idx_cropped)  # (B, T, C)
            # focus on the last timestep
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index into the running batches of contexts
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx


class CharBigram(LanguageModel):
    def __init__(self, *, context_size: int, vocab_size: int) -> None:
        super().__init__(context_size)
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx has shape (B, T)"""
        return self.embedding(idx)  # (B, T, vocab_size)


class WordMLP(LanguageModel):
    def __init__(self, *, context_size: int, vocab_size: int, embedding_size: int) -> None:
        super().__init__(context_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.proj = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx has shape (B, T)"""
        embeddings = self.embedding(idx)  # (B, T, embedding_size)
        return self.proj(embeddings)  # (B, T, vocab_size)


class AttentionHead(nn.Module):
    def __init__(self, *, context_size: int, embedding_size: int, head_size: int, dropout: float) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        x_q = self.query(x)  # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        x_k = self.key(x)  # same
        x_v = self.value(x)  # same

        # attention scores/affinities
        attn = x_q @ x_k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, T)
        attn = attn.masked_fill(self.tril[:T, :T] == 0.0, float("-inf"))  # ty:ignore[not-subscriptable]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        # weighted aggregation of the values
        return attn @ x_v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        *,
        context_size: int,
        num_heads: int,
        head_size: int,
        embedding_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            AttentionHead(
                context_size=context_size, embedding_size=embedding_size, head_size=head_size, dropout=dropout
            )
            for _ in range(num_heads)
        )
        self.proj = nn.Linear(num_heads * head_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (B, T, embedding_size)"""
        # cat((B, T, head_size) x num_heads) -> (B, T, head_size * num_heads)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    """a simple feed-forward network"""

    def __init__(self, embedding_size: int, projection_factor: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, projection_factor * embedding_size),
            nn.ReLU(),
            nn.Linear(projection_factor * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """a simple transfomer block: communication followed by computation, also a light touch of residual connections"""

    def __init__(
        self, *, context_size: int, num_heads: int, embedding_size: int, ffw_projection_factor: int, dropout: float
    ) -> None:
        super().__init__()
        head_size = embedding_size // num_heads
        self.sa = MultiHeadAttention(
            context_size=context_size,
            num_heads=num_heads,
            head_size=head_size,
            embedding_size=embedding_size,
            dropout=dropout,
        )
        self.ffwd = FeedForward(embedding_size=embedding_size, projection_factor=ffw_projection_factor, dropout=dropout)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))  # pre-norm formulation
        return x + self.ffwd(self.ln2(x))


class CharTransformer(LanguageModel):
    def __init__(  # noqa: PLR0913
        self,
        *,
        vocab_size: int,
        context_size: int,
        embedding_size: int,
        num_blocks: int,
        num_heads: int,
        ffw_projection_factor: int,
        dropout: float,
    ) -> None:
        super().__init__(context_size)
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_embed = nn.Embedding(context_size, embedding_size)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    context_size=context_size,
                    num_heads=num_heads,
                    embedding_size=embedding_size,
                    ffw_projection_factor=ffw_projection_factor,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ],
        )
        self.ln_f = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx has shape (B, T)"""
        _, T = idx.shape
        tok_embed = self.embed(idx)  # (B, T, embedding_size)
        pos_embed = self.pos_embed(torch.arange(T, device=idx.device))  # (T, embedding_size)
        embeddings = tok_embed + pos_embed  # (B, T, embedding_size) broadcast over B
        embeddings = self.blocks(embeddings)  # (B, T, embedding_size)
        embeddings = self.ln_f(embeddings)  # (B, T, embedding_size)
        return self.lm_head(embeddings)  # (B, T, vocab_size)


def train_lm(
    dataloader: data_utils.DataLoader,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    loss_accumulator: list[float] | None = None,
) -> None:
    size = len(dataloader.dataset)  # ty:ignore[invalid-argument-type]
    loss_accumulator = loss_accumulator if loss_accumulator is not None else []
    model.train()  # set train mode
    for batch, (X, y) in enumerate(dataloader):
        X_d, y_d = X.to(device), y.to(device)

        # forward
        _, loss = model.calculate_loss(X_d, y_d)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accumulator.append(loss.log10().item())
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_lm(
    dataloader: data_utils.DataLoader,
    model: LanguageModel,
) -> None:
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            X_d, y_d = X.to(device), y.to(device)
            _, loss = model.calculate_loss(X_d, y_d)
            test_loss += loss.item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


def main() -> None:
    match MODEL_TYPE:
        case ModelType.CHAR_BIGRAM | ModelType.CHAR_TRANSFORMER:
            vocab_size = len(chars)
            encoder: Encoder = encode_char
            decoder: Decoder = decode_char
        case ModelType.WORD_MLP:
            vocab_size = len(words)
            encoder: Encoder = encode_word
            decoder: Decoder = decode_word

    n1 = int(TRAIN_SPLIT * len(text))
    train_dataset = TinyShakespeare(text[:n1], CONTEXT_SIZE, encoder)
    val_dataset = TinyShakespeare(text[n1:], CONTEXT_SIZE, encoder)

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    match MODEL_TYPE:
        case ModelType.CHAR_BIGRAM:
            model = CharBigram(context_size=CONTEXT_SIZE, vocab_size=vocab_size).to(device)
        case ModelType.WORD_MLP:
            model = WordMLP(context_size=CONTEXT_SIZE, vocab_size=vocab_size, embedding_size=EMBEDDING_SIZE).to(device)
        case ModelType.CHAR_TRANSFORMER:
            model = CharTransformer(
                num_blocks=TRANSFORMER_NUM_BLOCKS,
                num_heads=TRANSFORMER_NUM_HEADS,
                context_size=CONTEXT_SIZE,
                vocab_size=vocab_size,
                embedding_size=EMBEDDING_SIZE,
                ffw_projection_factor=FEEDFORWARD_PROJECTION_FACTOR,
                dropout=DROPOUT,
            ).to(device)
        case _:
            assert_never(model)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(model)
    print("dataset size:", len(train_dataset) + len(val_dataset))
    print("train size:", len(train_dataset))
    print("val size:", len(val_dataset))
    print("batch size:", BATCH_SIZE)
    print("device:", device)
    print("model type:", MODEL_TYPE)
    print("vocab size:", vocab_size)
    print("context size:", CONTEXT_SIZE)
    print("model parameters:", sum(p.numel() for p in model.parameters()))

    test_lm(train_dataloader, model)
    test_lm(val_dataloader, model)
    with torch.no_grad():
        print(
            decoder(
                model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()
            )
        )

    lossi = []
    for t in range(NUM_EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_lm(train_dataloader, model, optimizer, lossi)
        test_lm(val_dataloader, model)

    plt.plot(lossi)
    plt.xlabel("iteration")
    plt.ylabel("log10 loss")
    plt.title(f"training loss curve for {MODEL_TYPE}")
    plt.savefig(f"{MODEL_TYPE}_loss.png")

    # sample
    test_lm(train_dataloader, model)
    test_lm(val_dataloader, model)
    with torch.no_grad():
        print(
            decoder(
                model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()
            )
        )


if __name__ == "__main__":
    main()
