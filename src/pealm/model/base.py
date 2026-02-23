import abc
import enum
import logging
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class Type(enum.StrEnum):
    CHAR_BIGRAM = enum.auto()
    CHAR_TRANSFORMER = enum.auto()
    GPT2 = enum.auto()
    PEASHOOTER = enum.auto()


class LanguageModel(nn.Module, abc.ABC):
    TYPE: Type

    def __init__(self, context_size: int) -> None:
        super().__init__()
        self.context_size = context_size

    @abc.abstractmethod
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx and targets both have shape (B, T)"""

    def sample_next_token(self, idx: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
        # crop idx  to the last context_size tokens
        idx_cropped = idx[:, -self.context_size :]
        logits, _ = self(idx_cropped)  # (B, T, vocab_size)
        # focus on the last timestep
        logits = logits[:, -1, :]  # (B, vocab_size)

        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            # samples into the (B, K) distributions, NOT (B, vocab_size)!
            # meaning ix is sampled indices into topk_indices
            topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)  # (B, K)
            topk_logits = topk_logits / temperature
            probs = F.softmax(topk_logits, dim=-1)  # (B, K)
            ix = torch.multinomial(probs, num_samples=1)  # (B, 1)
            return topk_indices.gather(dim=-1, index=ix)  # (B, 1)
        # non-top k sampling: apply temperature
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        # sample from the full distribution
        return torch.multinomial(probs, num_samples=1)  # (B, 1)

    def generate_stream(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> Iterator[torch.Tensor]:
        """Yield next token tensor (B, 1) at each step."""
        for _ in range(max_new_tokens):
            idx_next = self.sample_next_token(idx, temperature, top_k)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1)

            yield idx_next

    def generate(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        """Non-streaming generate. Returns full sequence."""
        for idx_next in self.generate_stream(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            idx = torch.cat((idx, idx_next), dim=-1)

        return idx
