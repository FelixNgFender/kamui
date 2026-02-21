from pealm import model as model_mod
from pealm import settings, utils
from pealm.sample import base
from pealm.tokenizer import CharTokenizer


def sample_char_transformer(sample_settings: settings.Sample, model_settings: settings.CharTransformer) -> None:
    device = utils.compute_init(
        use_accelerator=sample_settings.use_accelerator,
        seed=sample_settings.seed,
        torch_seed=sample_settings.torch_seed,
        fp32_matmul_precision=sample_settings.fp32_matmul_precision,
    )
    tokenizer = CharTokenizer.load(sample_settings.tokenizer_dir)
    model = model_mod.CharTransformer(
        num_blocks=model_settings.num_blocks,
        num_heads=model_settings.num_heads,
        context_size=model_settings.context_size,
        vocab_size=tokenizer.vocab_size,
        embedding_size=model_settings.embedding_size,
        dropout=model_settings.dropout,
    )
    base.sample(
        device=device,
        model=model,
        tokenizer=tokenizer,
        sample_settings=sample_settings,
    )
