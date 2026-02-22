from pealm import model, settings, tokenizer, utils
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
    _tokenizer = tokenizer.GPT2Tokenizer()
    _model = model.GPT2.from_pretrained(model_settings.variant)
    base.sample(
        device=device,
        model=_model,
        tokenizer=_tokenizer,
        sample_settings=sample_settings,
    )
