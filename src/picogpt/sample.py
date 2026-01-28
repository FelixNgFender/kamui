import logging
import random
from typing import assert_never

import torch

from picogpt import model as model_mod
from picogpt import settings, tokenizers

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
            model = model_mod.CharBigram(context_size=context_size, vocab_size=tokenizer.vocab_size).to(device)
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
    model.load_state_dict(torch.load(sample_settings.checkpoint, map_location=device, weights_only=True))

    model.eval()

    # prepare initial context
    if sample_settings.prompt:
        logger.debug("using prompt: %s", sample_settings.prompt)
        # encode the prompt
        idx = tokenizer.encode(sample_settings.prompt)
        context = torch.tensor([idx], dtype=torch.long, device=device)  # (1, len(prompt))
    else:
        logger.debug("starting with empty context")
        # start with a single zero token (like in training)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # generate tokens
    logger.debug("generating %d tokens with temperature %.2f", sample_settings.max_tokens, sample_settings.temperature)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=sample_settings.max_tokens,
            temperature=sample_settings.temperature,
        )

    # decode
    generated_idx = generated[0].tolist()
    generated_text = tokenizer.decode(generated_idx)

    print(generated_text)  # noqa: T201

    logger.info("generation complete (%d total tokens)", len(generated_idx))
