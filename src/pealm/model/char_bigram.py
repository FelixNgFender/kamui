# ruff: noqa: N812
import torch
import torch.nn.functional as F
from torch import nn

from pealm.model import base


class CharBigram(base.LanguageModel):
    TYPE = base.Type.CHAR_BIGRAM

    def __init__(self, *, context_size: int, vocab_size: int) -> None:
        super().__init__(context_size)
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.embedding(idx)  # (B, T, vocab_size)

        # batch size x time steps x channels/embeddings/features
        # instead of batch x time x channels, we group batch x time so the rows
        # become individual embeddings at each time step (essentially treating
        # them as individual examples)
        # corresponding with each embedding row is a single label index, so we
        # stretch out targets
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
