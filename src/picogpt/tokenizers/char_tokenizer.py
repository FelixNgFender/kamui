import json
import logging
import pathlib
from collections.abc import Iterable

from picogpt import constants
from picogpt.tokenizers import tokenizer
from picogpt.tokenizers import vocab as vocab_mod

logger = logging.getLogger(__name__)


class CharTokenizer:
    TYPE = tokenizer.Type.CHAR

    def __init__(self, vocab: vocab_mod.Vocab) -> None:
        self.vocab = vocab

    def encode(self, s: str) -> list[int]:
        return [self.vocab.stoi[ch] for ch in s]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.vocab.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab.itos)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
    ) -> "CharTokenizer":
        chars = sorted(set("".join(texts)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        vocab = vocab_mod.Vocab(stoi=stoi, itos=chars)
        return cls(vocab=vocab)

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        vocab_path = path / constants.VOCAB_FILENAME
        with vocab_path.open("w", encoding="utf-8") as f:
            json.dump(self.vocab.to_dict(), f, ensure_ascii=False, indent=2)  # ty:ignore[possibly-missing-attribute]
        logger.info("saved vocab to %s", vocab_path)

        tokenizer_path = path / constants.TOKENIZER_FILENAME
        with tokenizer_path.open("w") as f:
            json.dump(
                {
                    "type": self.TYPE,
                },
                f,
                indent=2,
            )
        logger.info("saved tokenizer config to %s", tokenizer_path)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "CharTokenizer":
        path = pathlib.Path(path)

        tokenizer_path = path / constants.TOKENIZER_FILENAME
        with tokenizer_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("type") != cls.TYPE:
            msg = f"invalid tokenizer type: expected {cls.TYPE}, got {config.get('type')}"
            raise ValueError(msg)

        vocab_path = path / constants.VOCAB_FILENAME
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = vocab_mod.Vocab.from_dict(json.load(f))
        return cls(vocab=vocab)
