"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ uv run scripts/fineweb.py
Will save shards to the local directory "data/edu_fineweb10B".
"""

import math
import multiprocessing as mp
import os
import pathlib

import datasets
import numpy as np
import tiktoken
from tqdm import tqdm

# ------------------------------------------
local_dir = "data/fineweb_edu10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = pathlib.Path(local_dir)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(doc: dict) -> np.ndarray:
    # init the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (tokens_np >= 0).all(), "token dictionary too small for uint16"
    assert (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def main() -> None:
    # download the dataset
    fw = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() or 0)
    chunksize = math.ceil(len(fw) / nprocs / 16)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=chunksize):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = DATA_CACHE_DIR / f"{split}_{shard_index:06d}"
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)  # ty:ignore[unresolved-attribute]
                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                np.save(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = DATA_CACHE_DIR / f"{split}_{shard_index:06d}"
            np.save(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()
