import functools
import os
from typing import Annotated, Literal

import pydantic
import pydantic_settings as ps

from pealm import constants
from pealm.settings import base


class Log(base.Base):
    verbose: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("v", "verbose"), description="Logs extra debugging information"
        ),
    ] = False


class Seed(base.Base):
    seed: Annotated[
        int,
        pydantic.Field(description="Random seed for Python"),
    ] = constants.SEED
    torch_seed: Annotated[
        int,
        pydantic.Field(description="Random seed for PyTorch"),
    ] = constants.TORCH_SEED


class Device(base.Base):
    use_accelerator: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(description="Whether to use accelerator for training"),
    ] = constants.USE_ACCELERATOR


class Precision(base.Base):
    fp32_matmul_precision: Annotated[
        Literal["highest", "high", "medium"],
        pydantic.Field(description="FP32 matrix multiplication precision for PyTorch"),
    ] = constants.FP32_MATMUL_PRECISION
    use_mixed_precision: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(
            description=(
                "Whether to use mixed precision for training and inference. If enabled, use bfloat16 where applicable."
            )
        ),
    ] = constants.USE_MIXED_PRECISION


class DDP(base.Base):
    """DDP settings auto-populated from environment variables set by torchrun."""

    @pydantic.computed_field
    @functools.cached_property[bool]
    def enabled(self) -> bool:
        """DDP is enabled when running with torchrun."""
        return all(var in os.environ for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

    rank: Annotated[
        pydantic.NonNegativeInt, pydantic.Field(description="Global process rank. 0 for master process.")
    ] = constants.DDP_RANK
    local_rank: Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(description="Local process rank used for device assignment."),
    ] = constants.DDP_LOCAL_RANK
    world_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Total number of processes in the process group."),
    ] = constants.DDP_WORLD_SIZE

    @pydantic.computed_field
    @functools.cached_property[bool]
    def is_master_process(self) -> bool:
        return self.rank == 0
