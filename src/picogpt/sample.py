import logging
import random
import time
from typing import assert_never

import torch

from picogpt import model as model_mod
from picogpt import settings, tokenice, training

logger = logging.getLogger(__name__)


def sample(sample_settings: settings.Sample, model_settings: settings.Model) -> None:  # noqa: C901, PLR0912, PLR0915
    if sample_settings.checkpoint and not sample_settings.checkpoint.exists():
        msg = f"checkpoint file not found: {sample_settings.checkpoint}"
        raise FileNotFoundError(msg)

    random.seed(sample_settings.seed)
    torch.manual_seed(sample_settings.torch_seed)
    torch.cuda.manual_seed_all(sample_settings.torch_seed)
    # tells pytorch to use different kernels depending on precision
    torch.set_float32_matmul_precision(sample_settings.fp32_matmul_precision)
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available() and sample_settings.use_accelerator
        else torch.device("cpu")
    )
    assert device is not None, "device cannot be None"  # noqa: S101

    context_size = model_settings.context_size

    # create model with correct architecture
    match model_settings:
        case settings.CharBigram():
            tokenizer = tokenice.CharTokenizer.load(sample_settings.tokenizer_dir)
            model = model_mod.CharBigram(
                context_size=context_size,
                vocab_size=tokenizer.vocab_size,
            )
        case settings.CharTransformer():
            tokenizer = tokenice.CharTokenizer.load(sample_settings.tokenizer_dir)
            model = model_mod.CharTransformer(
                num_blocks=model_settings.num_blocks,
                num_heads=model_settings.num_heads,
                context_size=context_size,
                vocab_size=tokenizer.vocab_size,
                embedding_size=model_settings.embedding_size,
                ffw_projection_factor=model_settings.feedforward_projection_factor,
                dropout=model_settings.dropout,
            )
        case settings.GPT2():
            tokenizer = tokenice.GPT2Tokenizer()
            if sample_settings.checkpoint is None:
                model = model_mod.GPT2.from_pretrained("gpt2")
            else:
                model = model_mod.GPT2(
                    context_size=context_size,
                    # don't use tokenizer.vocab_size for GPT2 cuz we want 50304 for cuda niceness
                    vocab_size=model_settings.vocab_size,
                    embedding_size=model_settings.embedding_size,
                    num_layers=model_settings.num_layers,
                    num_heads=model_settings.num_heads,
                    ffw_projection_factor=model_settings.feedforward_projection_factor,
                )
        case _:
            assert_never(model_settings)

    model.eval().to(device).compile()

    # load checkpoint
    if sample_settings.checkpoint is not None:
        logger.debug("loading checkpoint from %s", sample_settings.checkpoint)
        with torch.serialization.safe_globals([training.Checkpoint, model_mod.Type]):
            checkpoint = torch.load(sample_settings.checkpoint, map_location=device, weights_only=True)
            if isinstance(checkpoint, training.Checkpoint):
                model_state_dict = checkpoint.model_state_dict
            elif isinstance(checkpoint, dict):
                model_state_dict = checkpoint
            else:
                msg = f"unsupported checkpoint type: {type(checkpoint)}"
                raise TypeError(msg)
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
