from pealm import model, settings, tokenizer, utils
from pealm.eval import base


def eval_gpt2_pretrained(eval_settings: settings.EvalGPT2Pretrained, model_settings: settings.GPT2Pretrained) -> None:
    device = utils.compute_init(
        use_accelerator=eval_settings.use_accelerator,
        seed=eval_settings.seed,
        torch_seed=eval_settings.torch_seed,
        fp32_matmul_precision=eval_settings.fp32_matmul_precision,
    )
    _tokenizer = tokenizer.GPT2Tokenizer()
    _model = model.GPT2.from_pretrained(model_settings.variant)
    base.evaluate(
        device=device,
        model=_model,
        tokenizer=_tokenizer,
        eval_settings=eval_settings,
    )
