"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import pathlib

import requests
import tqdm

# -----------------------------------------------------------------------------
local_dir = "data/hellaswag"
DATA_CACHE_DIR = pathlib.Path(local_dir)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, fname: pathlib.Path, chunk_size: int = 1024) -> None:
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True, timeout=10)
    total = int(resp.headers.get("content-length", 0))
    with (
        fname.open("wb") as file,
        tqdm.tqdm(
            desc=str(fname),
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def main() -> None:
    hellaswags = {
        "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
        "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
        "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
    }

    for split, url in hellaswags.items():
        data_filename = DATA_CACHE_DIR / f"{split}.jsonl"
        if not data_filename.exists():
            download_file(url, data_filename)


if __name__ == "__main__":
    main()
