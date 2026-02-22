import logging

import torch
import torch.nn.functional as F

from pealm import model as model_mod
from pealm import settings, train
from pealm import tokenizer as tokenizer_mod
from pealm.eval import hellaswag

logger = logging.getLogger(__name__)


def evaluate(
    device: torch.device,
    model: model_mod.LanguageModel,
    tokenizer: tokenizer_mod.Tokenizer,
    eval_settings: settings.Eval,
) -> None:
    """Common to all GPT2-* models to evaluate on HellaSwag"""
    model.eval().to(device).compile()

    # load checkpoint
    if eval_settings.checkpoint is not None:
        model_state_dict = train.Checkpoint.load_weights(eval_settings.checkpoint, map_location=device)
        model.load_state_dict(model_state_dict)

    num_correct_norm = 0
    num_correct = 0
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=eval_settings.use_mixed_precision),
    ):
        for num_total, example in enumerate(
            hellaswag.iter_file(eval_settings.input_dir / f"{eval_settings.split}.jsonl")
        ):
            tokens, mask, label = example.render(tokenizer)
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
