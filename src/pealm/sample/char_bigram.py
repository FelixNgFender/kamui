from pealm import model, settings, utils
from pealm.sample import base
from pealm.tokenizer import CharTokenizer


def sample_char_bigram(sample_settings: settings.Sample, model_settings: settings.CharBigram) -> None:
    device = utils.compute_init(
        use_accelerator=sample_settings.use_accelerator,
        seed=sample_settings.seed,
        torch_seed=sample_settings.torch_seed,
        fp32_matmul_precision=sample_settings.fp32_matmul_precision,
    )
    tokenizer = CharTokenizer.load(sample_settings.tokenizer_dir)
    _model = model.CharBigram(
        context_size=model_settings.context_size,
        vocab_size=tokenizer.vocab_size,
    )
    base.sample(
        device=device,
        model=_model,
        tokenizer=tokenizer,
        sample_settings=sample_settings,
    )
