from pealm import model as model_mod
from pealm import settings, utils
from pealm import tokenizer as tokenizer_mod
from pealm.sample import base


def sample_gpt2(sample_settings: settings.SampleGPT2, model_settings: settings.GPT2) -> None:
    device = utils.compute_init(
        use_accelerator=sample_settings.use_accelerator,
        seed=sample_settings.seed,
        torch_seed=sample_settings.torch_seed,
        fp32_matmul_precision=sample_settings.fp32_matmul_precision,
    )
    tokenizer = tokenizer_mod.GPT2Tokenizer()
    model = model_mod.GPT2(
        context_size=model_settings.context_size,
        # don't use tokenizer.vocab_size for GPT2 cuz we want 50304 for cuda niceness
        vocab_size=model_settings.vocab_size,
        embedding_size=model_settings.embedding_size,
        num_layers=model_settings.num_layers,
        num_heads=model_settings.num_heads,
    )
    base.sample(
        device=device,
        model=model,
        tokenizer=tokenizer,
        sample_settings=sample_settings,
    )
