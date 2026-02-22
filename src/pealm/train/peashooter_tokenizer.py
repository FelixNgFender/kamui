"""
Train a tokenizer using our own peashooter tokenizer library.
In the style of GPT-4 tokenizer.
"""

import logging
import time

import torch

from pealm import constants, dataset, report, settings, tokenizer

logger = logging.getLogger(__name__)


def train_peashooter_tokenizer(
    train_settings: settings.TrainPeashooterTokenizer,
    tokenizer_settings: settings.PeashooterTokenizer,
    report: report.DistReport,
) -> None:
    # train the tokenizer from parquet shards on train split
    t0 = time.perf_counter()

    text_iter = dataset.ShardedParquet(
        train_settings.input_dir,
        "train",
        # single device training
        0,
        1,
        train_settings.max_chars_per_doc,
        train_settings.max_chars,
        shuffle=True,
        seed=train_settings.seed,
    ).iter_texts()
    _tokenizer = tokenizer.PeashooterTokenizer.train_from_iterator(text_iter, tokenizer_settings.vocab_size)

    t1 = time.perf_counter()
    train_time = t1 - t0

    logger.info("training time: %.2f seconds", train_time)

    # save the tokenizer to disk
    _tokenizer.save(train_settings.output_dir)

    # quick inline sanity check
    test_text = """Hello world! This is a test.
    Numbers: 123, 4567, 89
    Contractions: I'm, you're, it's
    Special chars: @#$%^&*()
    Unicode: 你好世界 🌍"""
    encoded = _tokenizer.encode(test_text)
    decoded = _tokenizer.decode(encoded)

    if decoded != test_text:
        msg = (
            f"tokenizer sanity check failed! decoded text does not match original.\noriginal: {test_text}\ndecoded: "
            f"{decoded}"
        )
        raise ValueError(msg)

    # -----------------------------------------------------------------------------
    # one more thing: we wish to cache a mapping from token id to number of bytes of that token
    # for efficient evaluation of bits per byte. unlike the typical mean loss, this
    # allows us to report a loss that is invariant to the vocab size of the tokenizer.
    # the bits per byte on the validation set is then one of the primary metrics we care about.
    vocab_size = _tokenizer.vocab_size
    special_set = _tokenizer.special_tokens_set
    token_strings: list[str] = [_tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes: list[int] = []
    for token_id in range(vocab_size):
        token_str = token_strings[token_id]  # the Python string representation of this token
        if token_str in special_set:
            token_bytes.append(0)  # special characters are not counted
            continue
        id_bytes = len(token_str.encode("utf-8"))  # number of bytes that make up this token
        token_bytes.append(id_bytes)
    token_bytes_pt = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    token_bytes_path = train_settings.output_dir / constants.PS_TOKENIZER_BYTES_PER_TOKEN_FILENAME
    with token_bytes_path.open("wb") as f:
        torch.save(token_bytes_pt, f)
    logger.info("saved token_bytes to %s", token_bytes_path)

    # log to report
    token_bytes_nonzero = (token_bytes_pt[token_bytes_pt > 0]).to(dtype=torch.float32)
    report.log(
        section="Tokenizer training",
        data=[
            {
                # translate to nanochat-equivant args
                "max_chars": train_settings.max_chars,
                "doc_cap": train_settings.max_chars_per_doc,
                "vocab_size": tokenizer_settings.vocab_size,
            },
            {"train_time": train_time},
            {"num_special_tokens": len(special_set)},
            {
                "token_bytes_min": int(token_bytes_nonzero.min().item()),
                "token_bytes_max": int(token_bytes_nonzero.max().item()),
                "token_bytes_mean": token_bytes_nonzero.mean().item(),
                "token_bytes_std": token_bytes_nonzero.std().item(),
            },
        ],
    )
