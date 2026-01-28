import logging
import random
import time
from typing import assert_never

import torch

from picogpt import model as model_mod
from picogpt import settings, tokenizers, training

logger = logging.getLogger(__name__)


def sample(sample_settings: settings.Sample, model_settings: settings.Model) -> None:
    if not sample_settings.checkpoint.exists():
        msg = f"checkpoint file not found: {sample_settings.checkpoint}"
        raise FileNotFoundError(msg)

    random.seed(sample_settings.seed)
    torch.manual_seed(sample_settings.torch_seed)
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available() and sample_settings.use_accelerator
        else torch.device("cpu")
    )

    tokenizer = tokenizers.CharTokenizer.load(sample_settings.tokenizer_dir)
    context_size = sample_settings.context_size

    # create model with correct architecture
    match model_settings:
        case settings.CharBigram():
            model = model_mod.CharBigram(
                context_size=context_size,
                vocab_size=tokenizer.vocab_size,
            ).to(device)
        case settings.CharTransformer():
            model = model_mod.CharTransformer(
                num_blocks=model_settings.transformer_num_blocks,
                num_heads=model_settings.transformer_num_heads,
                context_size=context_size,
                vocab_size=tokenizer.vocab_size,
                embedding_size=model_settings.embedding_size,
                ffw_projection_factor=model_settings.transformer_feedforward_projection_factor,
                dropout=model_settings.transformer_dropout,
            ).to(device)
        case _:
            assert_never(model_settings)

    # load checkpoint
    logger.debug("loading checkpoint from %s", sample_settings.checkpoint)
    with torch.serialization.safe_globals([training.Checkpoint, model_mod.Type]):
        checkpoint = torch.load(sample_settings.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()

    # prepare initial context
    if sample_settings.prompt:
        logger.debug("using prompt: %s", sample_settings.prompt)
        idx = tokenizer.encode(sample_settings.prompt)
        context = torch.tensor([idx], dtype=torch.long, device=device)
    else:
        logger.debug("starting with empty context")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    prompt_len = context.shape[1]

    logger.debug(
        "generating %d tokens with temperature %.2f",
        sample_settings.tokens,
        sample_settings.temperature,
    )

    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=sample_settings.tokens,
            temperature=sample_settings.temperature,
        )

    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # decode
    generated_idx = generated[0].tolist()
    generated_text = tokenizer.decode(generated_idx)

    print(generated_text)  # noqa: T201

    total_tokens = len(generated_idx)
    new_tokens = total_tokens - prompt_len
    tps = new_tokens / elapsed if elapsed > 0 else float("inf")

    logger.info(
        "generation complete: %d new tokens in %.3fs (%.2f tokens/sec)",
        new_tokens,
        elapsed,
        tps,
    )
