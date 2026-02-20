# ruff: noqa: N812, N806, PLR0913
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from pealm import constants
from pealm.models import language_model as lm

logger = logging.getLogger(__name__)


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
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # ty:ignore[not-subscriptable]
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

    def __init__(self, embedding_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, constants.TRANSFORMER_FFW_PROJECTION_FACTOR * embedding_size),
            nn.ReLU(),
            nn.Linear(constants.TRANSFORMER_FFW_PROJECTION_FACTOR * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """a simple transfomer block: communication followed by computation, also a light touch of residual connections"""

    def __init__(self, *, context_size: int, num_heads: int, embedding_size: int, dropout: float) -> None:
        super().__init__()
        head_size = embedding_size // num_heads
        self.sa = MultiHeadAttention(
            context_size=context_size,
            num_heads=num_heads,
            head_size=head_size,
            embedding_size=embedding_size,
            dropout=dropout,
        )
        self.ffwd = FeedForward(embedding_size=embedding_size, dropout=dropout)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))  # pre-norm formulation
        return x + self.ffwd(self.ln2(x))


class CharTransformer(lm.LanguageModel):
    TYPE = lm.Type.CHAR_TRANSFORMER

    def __init__(
        self,
        *,
        context_size: int,
        vocab_size: int,
        embedding_size: int,
        num_blocks: int,
        num_heads: int,
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
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ],
        )
        self.ln_f = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

        self.apply(self._init_weights)

    @torch.inference_mode()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape
        tok_embed = self.embed(idx)  # (B, T, embedding_size)
        pos_embed = self.pos_embed(torch.arange(T, dtype=torch.long, device=idx.device))  # (T, embedding_size)
        embeddings = tok_embed + pos_embed  # (B, T, embedding_size) broadcast over B
        embeddings = self.blocks(embeddings)  # (B, T, embedding_size)
        embeddings = self.ln_f(embeddings)  # (B, T, embedding_size)
        logits = self.lm_head(embeddings)  # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
