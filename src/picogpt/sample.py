import logging
import random
import time
from typing import assert_never

import torch

from picogpt import model as model_mod
from picogpt import settings, tokenizers, training

logger = logging.getLogger(__name__)


def sample(sample_settings: settings.Sample, model_settings: settings.Model) -> None:  # noqa: PLR0915
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

    logger.debug(
        "generating %d tokens with temperature %.2f (stream=%s)",
        sample_settings.tokens,
        sample_settings.temperature,
        sample_settings.stream,
    )

    # warmup cuda timing
    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    if sample_settings.stream:
        new_toks = 0

        with torch.no_grad():
            for idx_next in model.generate_stream(
                context,
                max_new_tokens=sample_settings.tokens,
                temperature=sample_settings.temperature,
            ):
                tokens = idx_next[0].tolist()
                text = tokenizer.decode(tokens)

                print(text, end="", flush=True)  # noqa: T201
                new_toks += 1

    else:
        with torch.no_grad():
            out = model.generate(
                context,
                max_new_tokens=sample_settings.tokens,
                temperature=sample_settings.temperature,
            )

        # slice only newly generated tokens
        gen = out[:, context.shape[1] :]
        new_toks = gen.shape[1]

        text = tokenizer.decode(gen[0].tolist())
        print(text, end="", flush=True)  # noqa: T201

    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    elapsed = t1 - t0
    tps = new_toks / elapsed if elapsed > 0 else float("inf")

    logger.info(
        "generated %d tokens in %.3fs (%.2f tokens/sec, stream=%s)",
        new_toks,
        elapsed,
        tps,
        sample_settings.stream,
    )
