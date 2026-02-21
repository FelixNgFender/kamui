import logging
import time

import torch

from pealm import model as model_mod
from pealm import settings, train
from pealm import tokenizer as tokenizer_mod

logger = logging.getLogger(__name__)


def sample(
    device: torch.device,
    model: model_mod.LanguageModel,
    tokenizer: tokenizer_mod.Tokenizer,
    sample_settings: settings.Sample,
) -> None:
    """Common to all models"""
    model.eval().to(device).compile()

    # load checkpoint if told, some pretrained models don't need to
    if sample_settings.checkpoint is not None:
        model_state_dict = train.Checkpoint.load_weights(sample_settings.checkpoint, map_location=device)
        model.load_state_dict(model_state_dict)

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
        sample_settings.n_tokens,
        sample_settings.temperature,
        sample_settings.stream,
    )

    # warmup cuda timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()

    if sample_settings.stream:
        new_toks = 0

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=sample_settings.use_mixed_precision),
        ):
            for idx_next in model.generate_stream(
                context,
                max_new_tokens=sample_settings.n_tokens,
                temperature=sample_settings.temperature,
            ):
                tokens = idx_next[0].tolist()
                text = tokenizer.decode(tokens)

                print(text, end="", flush=True)  # noqa: T201
                new_toks += 1

    else:
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=sample_settings.use_mixed_precision),
        ):
            out = model.generate(
                context,
                max_new_tokens=sample_settings.n_tokens,
                temperature=sample_settings.temperature,
                top_k=sample_settings.top_k,
            )

        # slice only newly generated tokens
        gen = out[:, context.shape[1] :]
        new_toks = gen.shape[1]

        text = tokenizer.decode(gen[0].tolist())
        print(text, end="", flush=True)  # noqa: T201
    # final newline
    print()  # noqa: T201

    if device.type == "cuda":
        torch.cuda.synchronize(device)

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
