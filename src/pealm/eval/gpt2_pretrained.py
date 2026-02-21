from pealm import model as model_mod
from pealm import settings, utils
from pealm import tokenizer as tokenizer_mod
from pealm.eval import base


def eval_gpt2_pretrained(eval_settings: settings.EvalGPT2Pretrained, model_settings: settings.GPT2Pretrained) -> None:
    device = utils.compute_init(
        use_accelerator=eval_settings.use_accelerator,
        seed=eval_settings.seed,
        torch_seed=eval_settings.torch_seed,
        fp32_matmul_precision=eval_settings.fp32_matmul_precision,
    )
    tokenizer = tokenizer_mod.GPT2Tokenizer()
    model = model_mod.GPT2.from_pretrained(model_settings.variant)
    base.evaluate(
        device=device,
        model=model,
        tokenizer=tokenizer,
        eval_settings=eval_settings,
    )
