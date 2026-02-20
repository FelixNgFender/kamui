import dataclasses
import json
import random
from collections.abc import Iterable

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import logging

from pealm import constants, settings, tokenice, training
from pealm import models as model_mod

logger = logging.get_logger(__name__)


@dataclasses.dataclass
class HellaSwagExample:
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


def render_example(
    tokenizer: tokenice.GPT2Tokenizer, example: HellaSwagExample
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example.ctx
    label = example.label
    endings = example.endings

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


def iter_examples(split: settings.HellaSwagSplit) -> Iterable[HellaSwagExample]:
    with (constants.HELLASWAG_DIR / f"{split}.jsonl").open("r") as f:
        for line in f:
            yield HellaSwagExample(**json.loads(line))


def evaluate(eval_settings: settings.Evaluate, model_settings: settings.Model) -> None:  # noqa: PLR0915
    # setup
    random.seed(eval_settings.seed)
    torch.manual_seed(eval_settings.torch_seed)
    torch.cuda.manual_seed_all(eval_settings.torch_seed)
    # tells pytorch to use different kernels depending on precision
    torch.set_float32_matmul_precision(eval_settings.fp32_matmul_precision)
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available() and eval_settings.use_accelerator
        else torch.device("cpu")
    )
    assert device is not None, "device cannot be None"  # noqa: S101

    # create model with correct architecture
    match model_settings:
        case settings.GPT2():
            tokenizer = tokenice.GPT2Tokenizer()
            if eval_settings.checkpoint is None:
                model = model_mod.GPT2.from_pretrained("gpt2")
            else:
                model = model_mod.GPT2(
                    context_size=model_settings.context_size,
                    # don't use tokenizer.vocab_size for GPT2 cuz we want 50304 for cuda niceness
                    vocab_size=model_settings.vocab_size,
                    embedding_size=model_settings.embedding_size,
                    num_layers=model_settings.num_layers,
                    num_heads=model_settings.num_heads,
                )
        case _:
            msg = f"unsupport model for evaluation: {type(model_settings).__name__}"
            raise RuntimeError(msg)

    model.eval().to(device).compile()

    # load checkpoint
    if eval_settings.checkpoint is not None:
        logger.debug("loading checkpoint from %s", eval_settings.checkpoint)
        with torch.serialization.safe_globals([training.Checkpoint, model_mod.Type]):
            checkpoint = torch.load(eval_settings.checkpoint, map_location=device, weights_only=True)
            if isinstance(checkpoint, training.Checkpoint):
                model_state_dict = checkpoint.model_state_dict
            elif isinstance(checkpoint, dict):
                model_state_dict = checkpoint
            else:
                msg = f"unsupported checkpoint type: {type(checkpoint)}"
                raise TypeError(msg)
        model.load_state_dict(model_state_dict)

    num_correct_norm = 0
    num_correct = 0
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=eval_settings.use_mixed_precision),
    ):
        for num_total, example in enumerate(iter_examples(eval_settings.split)):
            tokens, mask, label = render_example(tokenizer, example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # get the logits
            # for each position in the sequence, the model ouputs a score for every token in the vocab
            logits, _ = model(tokens)  # (4, 20, 50304)
            # evaluate the autoregressive loss at all positions
            shift_logits = (logits[..., :-1, :]).contiguous()  # crop off last logits (4, 19, 50304)
            shift_tokens = (tokens[..., 1:]).contiguous()  # crop off first tokens (4, 19)
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (76, 50304)
            flat_shift_tokens = shift_tokens.view(-1)  # (76)
            shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")  # (76)
            shift_losses = shift_losses.view(tokens.size(0), -1)  # (4, 19)
            # now get the average loss just for the completion region (where mask == 1), in each row
            shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
            masked_shift_losses = shift_losses * shift_mask  # (4, 19)
            # sum and divide by the number of 1s in the mask
            sum_loss = masked_shift_losses.sum(dim=-1)  # (4,)
            avg_loss = sum_loss / shift_mask.sum(dim=-1)  # (4,)
            # now we have a loss for each of the 4 completions
            # the one with the lowest loss should be the most likely
            pred = sum_loss.argmin().item()
            # also compare normalized by length to remove length bias: shorter text = fewer tokens = lower total loss
            pred_norm = avg_loss.argmin().item()

            # accumulate stats
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)
            logger.info(
                "%d | acc: %d/%d=%.4f | acc_norm: %d/%d=%.4f",
                num_total + 1,
                num_correct,
                num_total + 1,
                num_correct / (num_total + 1),
                num_correct_norm,
                num_total + 1,
                num_correct_norm / (num_total + 1),
            )

            # debug: pretty print a few examples, and the losses in each case
            if num_total % 100 == 0:
                logger.debug("-" * 20)
                logger.debug("example %d:", num_total)
                logger.debug("context:%s\n", example.ctx)
                logger.debug("endings:")
                for i, end in enumerate(example.endings):
                    logger.debug("  %d (loss: %.4f) %s", i, avg_loss[i].item(), end)
                logger.debug("predicted: %d, actual: %d", pred_norm, label)
