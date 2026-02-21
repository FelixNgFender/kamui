import dataclasses
import json
import pathlib
from collections.abc import Iterable

import torch

from pealm import tokenizer as tokenizer_mod


@dataclasses.dataclass
class Example:
    """
    Example HellaSwag json item:

    {"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

    ind: dataset ID
    activity_label: The ActivityNet or WikiHow label for this example
    context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.

    endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
    split: train, val, or test.
    split_type: indomain if the activity label is seen during training, else zeroshot
    source_id: Which video or WikiHow article this example came from
    """  # noqa: E501

    ind: int
    activity_label: str
    ctx_a: str
    ctx_b: str
    split: str
    split_type: str
    source_id: str

    ctx: str
    endings: list[str]
    label: int

    def render(self, tokenizer: tokenizer_mod.Tokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Given the example as a dictionary, render it as three torch tensors:
        - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
        - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
        - label (the index of the correct completion, which we hope has the highest likelihood)
        """
        ctx = self.ctx
        label = self.label
        endings = self.endings

        # gather up all the tokens
        ctx_tokens = tokenizer.encode(ctx)
        tok_rows = []
        mask_rows = []
        for end in endings:
            end_tokens = tokenizer.encode(" " + end)  # note: prepending " " because GPT-2 tokenizer
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

        # have to be careful during the collation because the number of tokens in each row can differ
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows, strict=True)):
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            mask[i, : len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, label


def iter_file(jsonl: pathlib.Path) -> Iterable[Example]:
    """Iterates a jsonl file of HellaSwag examples, yielding each example as an Example dataclass instance."""
    with jsonl.open("r") as f:
        for line in f:
            data = json.loads(line)
            yield Example(**data)
