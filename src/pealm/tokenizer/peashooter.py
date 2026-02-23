import functools
import logging
import pathlib
import pickle
from collections.abc import Iterator, Sequence

import rustbpe
import tiktoken

from pealm import constants
from pealm.tokenizer import base

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=32)
def _cached_encode_special(enc: tiktoken.Encoding, text_or_bytes: str | bytes) -> int:
    """Workaround to https://docs.astral.sh/ruff/rules/cached-instance-method/"""
    return enc.encode_single_token(text_or_bytes)


class PeashooterTokenizer:
    TYPE = base.Type.PEASHOOTER

    def __init__(self, enc: tiktoken.Encoding, bos: str) -> None:
        self.enc = enc
        self.bos_token_id = self.encode_special(bos)

    def _encode(
        self,
        text: str,
        prepend: int | str | None = None,
        append: int | str | None = None,
    ) -> list[int]:
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        ids = self.enc.encode_ordinary(text)
        if prepend is not None:
            ids = [prepend_id, *ids]
        if append is not None:
            ids.append(append_id)
        return ids

    def _encode_batch(
        self,
        text: list[str],
        prepend: int | str | None = None,
        append: int | str | None = None,
        num_threads: int = 8,
    ) -> list[list[int]]:
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
        if prepend_id is not None:
            for ids_row in ids:
                ids_row.insert(0, prepend_id)
        if append_id is not None:
            for ids_row in ids:
                ids_row.append(append_id)
        return ids

    def encode(self, s: str) -> list[int]:
        return self._encode(s)

    def encode_special(self, text_or_bytes: str | bytes) -> int:
        """Encode a single special token via exact match"""
        return _cached_encode_special(self.enc, text_or_bytes)

    def decode(self, ids: Sequence[int]) -> str:
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    @property
    def special_tokens_set(self) -> set[str]:
        return self.enc.special_tokens_set

    @classmethod
    def train(cls, texts: Sequence[str]) -> "PeashooterTokenizer":
        msg = (
            f"this code path is using the default vocab_size={constants.PS_VOCAB_SIZE}. if you intend to use a "
            "different vocab_size, call PeashooterTokenizer.train_from_iterator directly."
        )
        logger.warning(msg)
        return cls.train_from_iterator(iter(texts))

    @classmethod
    def train_from_iterator(
        cls, text_iter: Iterator[str], vocab_size: int = constants.PS_VOCAB_SIZE
    ) -> "PeashooterTokenizer":
        """Trains Peashooter tokenizer from an iterator"""
        if vocab_size < constants.PS_MIN_VOCAB_SIZE:
            msg = (
                f"vocab_size must be at least {constants.PS_MIN_VOCAB_SIZE} to accommodate the special tokens and the "
                f"minimum number of tokens needed for bpe merges. got vocab_size={vocab_size}."
            )
            raise ValueError(msg)

        # train using rustbpe first
        tokenizer = rustbpe.Tokenizer()  # ty:ignore[unresolved-attribute]
        # special tokens are inserted later, don't train now
        tokenizer.train_from_iterator(
            iterator=text_iter,
            vocab_size=vocab_size - len(constants.PS_SPECIAL_TOKENS),
            pattern=constants.PS_SPLIT_PATTERN,
        )
        # mergeable ranks (token bytes -> token id / rank)
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(b): rank for b, rank in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(constants.PS_SPECIAL_TOKENS)}

        enc = tiktoken.Encoding(
            name="peashooter",
            pat_str=tokenizer.get_pattern(),
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
            explicit_n_vocab=tokenizer.vocab_size + len(constants.PS_SPECIAL_TOKENS),
        )
        return PeashooterTokenizer(enc=enc, bos=constants.PeashooterSpecialTokens.BOS)

    @classmethod
    def from_pretrained(cls, name: str) -> "PeashooterTokenizer":
        """Construct a PeashooterTokenizer from a pretrained tiktoken tokenizer"""
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in peashooter we always use "<|bos|>" short for "beginning of sequence", but historically it is often
        # called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer_path = path / constants.PS_TOKENIZER_FILENAME
        with tokenizer_path.open("wb") as f:
            pickle.dump(self.enc, f)
        logger.info("saved tokenizer encoding to %s", tokenizer_path)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "PeashooterTokenizer":
        path = pathlib.Path(path)
        tokenizer_path = path / constants.PS_TOKENIZER_FILENAME
        with tokenizer_path.open("rb") as f:
            enc = pickle.load(f)  # noqa: S301
        return cls(enc, constants.PeashooterSpecialTokens.BOS)
