import enum
import logging
import pathlib
from collections.abc import Iterator
from typing import Literal

import numpy as np
import torch
import torch.utils.data as data_utils

from pealm import tokenizer as tokenizer_mod

logger = logging.getLogger(__name__)


class Type(enum.StrEnum):
    TEXT = enum.auto()
    NPY_SHARD = enum.auto()


class Text(data_utils.Dataset):
    TYPE = Type.TEXT

    def __init__(
        self,
        tokenizer: tokenizer_mod.Tokenizer,
        text: str,
        context_size: int,
    ) -> None:
        """Contexts and labels both have (B, C)"""
        self.context_size = context_size
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        # create all possible windows instead of skipping by context_size
        idx = torch.arange(len(data) - self.context_size).view(-1, 1)
        self.contexts = data[idx + torch.arange(self.context_size)]
        self.labels = data[idx + torch.arange(self.context_size) + 1]

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty:ignore[invalid-method-override]
        return self.contexts[idx], self.labels[idx]


class ShardedNpy(data_utils.IterableDataset):
    """Must be used with DataLoader(batch_size=None) to disable automatic batching."""

    TYPE = Type.NPY_SHARD

    def __init__(  # noqa: PLR0913
        self,
        data_dir: str | pathlib.Path,
        split: Literal["train", "val"],
        context_size: int,
        batch_size: int,
        rank: int,
        world_size: int,
        *,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        self.context_size = context_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.data_dir = pathlib.Path(data_dir)
        self.split = split
        self.shard_paths = sorted(self.data_dir.glob(f"{self.split}_*.npy"))
        if not self.shard_paths:
            msg = f"no shards found for split {self.split} in {self.data_dir}"
            raise ValueError(msg)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self.shuffle:
            self.rng.shuffle(self.shard_paths)

        for shard_path in self.shard_paths:
            logger.info("loading shard: %s", shard_path)
            data_np = np.load(shard_path).astype(np.int64)
            tokens = torch.from_numpy(data_np)
            start = self.batch_size * self.context_size * self.rank
            step = self.batch_size * self.context_size * self.world_size
            while start + self.batch_size * self.context_size + 1 < len(tokens):
                buf = tokens[start : start + self.batch_size * self.context_size + 1]
                contexts = buf[:-1].view(self.batch_size, self.context_size)
                labels = buf[1:].view(self.batch_size, self.context_size)
                yield contexts, labels
                start += step
