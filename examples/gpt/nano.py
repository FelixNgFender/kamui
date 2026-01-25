# ruff: noqa: INP001, N812, S101, N806, T201
import enum
import pathlib
import random
from typing import Protocol

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import nn, optim


class ModelType(enum.StrEnum):
    CHAR_BIGRAM = "char_bigram"
    WORD_MLP = "word_mlp"


USE_ACCELERATOR = True
TORCH_SEED = 2147483647
SEED = 42
INPUT_FILE = "tinyshakespeare.txt"
MODEL_TYPE: ModelType = ModelType.WORD_MLP


TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.1
assert TRAIN_SPLIT + VAL_SPLIT == 1.0

# hyperparams
CONTEXT_SIZE = 8  # the maximum length of predictions
EMBEDDING_SIZE = 24
HIDDEN_SIZE = 128
NUM_LAYERS = 5
BATCH_SIZE = 64  # the number of independent sequences to process at once
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

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
            # extremely expensive for now because we're feeding the whole
            # context into the model despite it only needing the last character
            # but keep for generalizability
            logits = self(idx)  # (B, T, C)
            # focus on the last timestep
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index into the running batches of contexts
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx


class CharBigram(LanguageModel):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx has shape (B, T)"""
        return self.embedding(idx)  # (B, T, vocab_size)


class WordMLP(LanguageModel):
    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.proj = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx has shape (B, T)"""
        embeddings = self.embedding(idx)  # (B, T, embedding_size)
        return self.proj(embeddings)  # (B, T, vocab_size)


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
        case ModelType.CHAR_BIGRAM:
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
            model = CharBigram(vocab_size).to(device)
        case ModelType.WORD_MLP:
            model = WordMLP(vocab_size, EMBEDDING_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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
