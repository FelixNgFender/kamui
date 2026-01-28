# ruff: noqa: N812, N806, PLR0913
import enum

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class Type(enum.StrEnum):
    CHAR_BIGRAM = enum.auto()
    CHAR_TRANSFORMER = enum.auto()


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

    def generate(self, idx: torch.Tensor, *, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """idx is (B, T) array of indices in the current context"""
        for _ in range(max_new_tokens):
            # crop idx  to the last context_size tokens
            idx_cropped = idx[:, -self.context_size :]
            logits = self(idx_cropped)  # (B, T, C)
            # focus on the last timestep
            logits = logits[:, -1, :]  # (B, C)
            # apply temperature
            logits = logits / temperature
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
    def __init__(
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

        self.apply(self._init_weights)

    @torch.no_grad
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx has shape (B, T)"""
        _, T = idx.shape
        tok_embed = self.embed(idx)  # (B, T, embedding_size)
        pos_embed = self.pos_embed(torch.arange(T, device=idx.device))  # (T, embedding_size)
        embeddings = tok_embed + pos_embed  # (B, T, embedding_size) broadcast over B
        embeddings = self.blocks(embeddings)  # (B, T, embedding_size)
        embeddings = self.ln_f(embeddings)  # (B, T, embedding_size)
        return self.lm_head(embeddings)  # (B, T, vocab_size)
