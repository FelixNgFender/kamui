from pealm import model as model_mod
from pealm import settings, utils
from pealm import tokenizer as tokenizer_mod
from pealm.sample import base


def sample_gpt2_pretrained(
    sample_settings: settings.SampleGPT2Pretrained, model_settings: settings.GPT2Pretrained
) -> None:
    device = utils.compute_init(
        use_accelerator=sample_settings.use_accelerator,
        seed=sample_settings.seed,
        torch_seed=sample_settings.torch_seed,
        fp32_matmul_precision=sample_settings.fp32_matmul_precision,
    )
    tokenizer = tokenizer_mod.GPT2Tokenizer()
    model = model_mod.GPT2.from_pretrained(model_settings.variant)
    base.sample(
        device=device,
        model=model,
        tokenizer=tokenizer,
        sample_settings=sample_settings,
    )
