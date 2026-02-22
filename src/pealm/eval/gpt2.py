from pealm import model, settings, tokenizer, utils
from pealm.eval import base


def eval_gpt2(eval_settings: settings.Eval, model_settings: settings.GPT2) -> None:
    device = utils.compute_init(
        use_accelerator=eval_settings.use_accelerator,
        seed=eval_settings.seed,
        torch_seed=eval_settings.torch_seed,
        fp32_matmul_precision=eval_settings.fp32_matmul_precision,
    )
    _tokenizer = tokenizer.GPT2Tokenizer()
    _model = model.GPT2(
        context_size=model_settings.context_size,
        # don't use tokenizer.vocab_size for GPT2 cuz we want 50304 for cuda niceness
        vocab_size=model_settings.vocab_size,
        embedding_size=model_settings.embedding_size,
        num_layers=model_settings.num_layers,
        num_heads=model_settings.num_heads,
    )
    base.evaluate(
        device=device,
        model=_model,
        tokenizer=_tokenizer,
        eval_settings=eval_settings,
    )
