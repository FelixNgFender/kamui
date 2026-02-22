"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import contextlib
import logging
import multiprocessing as mp
import pathlib
import time
from typing import Annotated

import pydantic
import pydantic_settings as ps
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # the last datashard is shard_01822.parquet


DATA_DIR = pathlib.Path("peashooter") / "base_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
def download_single_file(index: int) -> bool:
    """Downloads a single file index, with some backoff"""

    # construct the local filepath for this file and skip if it already exists
    filename = f"shard_{index:05d}.parquet"  # format of the filenames
    filepath = DATA_DIR / filename
    if filepath.exists():
        logger.info("%s already exists, skipping download", filepath)
        return True

    logger.info("downloading %s...", filename)

    # download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(f"{BASE_URL}/{filename}", stream=True, timeout=30)
            response.raise_for_status()
            # write to temporary file first
            temp_path = filepath.with_suffix(".tmp")
            with temp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # move temp file to final location
            temp_path.rename(filepath)
        except (OSError, requests.RequestException):
            logger.exception("attempt %d/%d failed for %s", attempt, max_attempts, filename)
            # clean up any partial files
            for path in [filepath.with_suffix(".tmp"), filepath]:
                if path.exists():
                    with contextlib.suppress(Exception):
                        path.unlink()
            # try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2**attempt
                logger.info("waiting %d seconds before retrying %s", wait_time, filename)
                time.sleep(wait_time)
            else:
                logger.warning("failed to download %s after %d attempts", filename, max_attempts)
                return False
        else:
            logger.info("successfully downloaded %s", filename)
            return True

    return False


class FineWebEduData(ps.BaseSettings, cli_parse_args=True, cli_use_class_docs_for_groups=True, cli_kebab_case=True):
    """Download FineWeb-Edu 100BT dataset shards"""

    num_files: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            le=MAX_SHARD + 1,
            validation_alias=pydantic.AliasChoices("n", "num-files"),
            description="Number of shards to download",
        ),
    ] = MAX_SHARD + 1
    num_workers: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("w", "num-workers"),
            description="Number of parallel download workers",
        ),
    ] = 4

    def cli_cmd(self) -> None:
        ids_to_download = list(range(self.num_files))
        logger.info("download %d shards using %d workers to %s\n", len(ids_to_download), self.num_workers, DATA_DIR)
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(download_single_file, ids_to_download)

        # Report results
        successful = sum(1 for success in results if success)
        logger.info("done! downloaded %d/%d shards to %s", successful, len(ids_to_download), DATA_DIR)

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


if __name__ == "__main__":
    ps.CliApp.run(FineWebEduData)
