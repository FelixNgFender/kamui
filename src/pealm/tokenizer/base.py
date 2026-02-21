import enum
import pathlib
from collections.abc import Sequence
from typing import Protocol


class Type(enum.StrEnum):
    CHAR = enum.auto()
    GPT2 = enum.auto()
    PEASHOOTER = enum.auto()


class Tokenizer(Protocol):
    TYPE: Type

    def encode(self, s: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    @classmethod
    def train(cls, texts: Sequence[str]) -> "Tokenizer": ...
    def save(self, path: str | pathlib.Path) -> None: ...
    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Tokenizer": ...
