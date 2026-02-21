import logging
import pathlib
from collections.abc import Sequence

import tiktoken

from pealm.tokenizer import base

logger = logging.getLogger(__name__)


class GPT2Tokenizer:
    TYPE = base.Type.GPT2

    def __init__(self) -> None:
        self.encoder = tiktoken.get_encoding("gpt2")

    def encode(self, s: str) -> list[int]:
        return self.encoder.encode(s)

    def decode(self, ids: Sequence[int]) -> str:
        return self.encoder.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.encoder.max_token_value + 1

    @classmethod
    def train(
        cls,
        texts: Sequence[str],  # noqa: ARG003
    ) -> "GPT2Tokenizer":
        logger.warning("GPT2Tokenizer uses a pre-trained tokenizer; training is not required.")
        return cls()

    def save(self, path: str | pathlib.Path) -> None:  # noqa: ARG002
        logger.warning("GPT2Tokenizer uses a pre-trained tokenizer; saving is not required.")

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "GPT2Tokenizer":  # noqa: ARG003
        logger.warning("GPT2Tokenizer uses a pre-trained tokenizer; loading is not required.")
        return cls()
