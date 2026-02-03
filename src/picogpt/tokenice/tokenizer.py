import enum
import pathlib
from collections.abc import Iterable
from typing import Protocol


class Type(enum.StrEnum):
    CHAR = enum.auto()


class Tokenizer(Protocol):
    TYPE: Type

    def encode(self, s: str) -> list[int]: ...
    def decode(self, ids: Iterable[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    @classmethod
    def train(cls, texts: Iterable[str]) -> "Tokenizer": ...
    def save(self, path: str | pathlib.Path) -> None: ...
    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Tokenizer": ...
