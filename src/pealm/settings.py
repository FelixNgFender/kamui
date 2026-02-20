import functools
import os
import pathlib
from typing import Annotated, Literal

import pydantic
import pydantic_settings as ps

from pealm import constants


class Log(ps.BaseSettings):
    verbose: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("v", "verbose"), description="Logs extra debugging information"
        ),
    ] = False

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class Seed(ps.BaseSettings):
    seed: Annotated[
        int,
        pydantic.Field(description="Random seed for Python"),
    ] = constants.SEED
    torch_seed: Annotated[
        int,
        pydantic.Field(description="Random seed for PyTorch"),
    ] = constants.TORCH_SEED


class Device(ps.BaseSettings):
    use_accelerator: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(description="Whether to use accelerator for training"),
    ] = constants.USE_ACCELERATOR


class Precision(ps.BaseSettings):
    fp32_matmul_precision: Annotated[
        Literal["highest", "high", "medium"],
        pydantic.Field(description="FP32 matrix multiplication precision for PyTorch"),
    ] = constants.FP32_MATMUL_PRECISION
    use_mixed_precision: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(description="Whether to use mixed precision for training and inference"),
    ] = constants.USE_MIXED_PRECISION


class DDP(ps.BaseSettings):
    """DDP settings auto-populated from environment variables set by torchrun."""

    @pydantic.computed_field
    @functools.cached_property[bool]
    def enabled(self) -> bool:
        """DDP is enabled when running with torchrun (RANK env var is set)."""
        return "RANK" in os.environ

    rank: Annotated[int, pydantic.Field(description="Global process rank. 0 for master process.")] = constants.DDP_RANK
    local_rank: Annotated[
        int,
        pydantic.Field(description="Local process rank used for device assignment."),
    ] = constants.DDP_LOCAL_RANK
    world_size: Annotated[
        int,
        pydantic.Field(description="Total number of processes in the process group."),
    ] = constants.DDP_WORLD_SIZE

    @pydantic.computed_field
    @functools.cached_property[bool]
    def is_master_process(self) -> bool:
        return self.rank == 0

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class ModelBase(ps.BaseSettings):
    """Settings common to all model type creation."""

    context_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Context size for the model"),
    ] = constants.CONTEXT_SIZE

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class CharBigram(ModelBase):
    """Settings for creating a character-level bigram model."""

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class CharTransformer(ModelBase):
    """Settings for creating a character-level transformer model."""

    embedding_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Embedding size for the model"),
    ] = constants.TRANSFORMER_EMBEDDING_SIZE
    num_blocks: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer blocks"),
    ] = constants.TRANSFORMER_NUM_BLOCKS
    num_heads: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer heads per layer"),
    ] = constants.TRANSFORMER_NUM_HEADS
    dropout: Annotated[
        float,
        pydantic.Field(ge=0.0, le=1.0, description="Transformer dropout rate"),
    ] = constants.TRANSFORMER_DROPOUT

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


GPT2Type = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


class GPT2(ModelBase):
    """Settings for creating the GPT-2 (124M) model from OpenAI."""

    # override ModelBase
    context_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Context size for the model"),
    ] = constants.GPT2_CONTEXT_SIZE
    vocab_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Vocabulary size for the model"),
    ] = constants.GPT2_VOCAB_SIZE
    embedding_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Embedding size for the model"),
    ] = constants.GPT2_EMBEDDING_SIZE
    num_layers: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer layers"),
    ] = constants.GPT2_NUM_LAYERS
    num_heads: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer heads per layer"),
    ] = constants.GPT2_NUM_HEADS

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


Model = CharBigram | CharTransformer | GPT2


class Input(ps.CliMutuallyExclusiveGroup):
    """Settings for input data. If none are specified, the tinyshakespeare text dataset is used."""

    txt: Annotated[
        pathlib.Path | None,
        pydantic.Field(
            description="Input text file for training",
        ),
    ] = None
    npy_shards: Annotated[
        pathlib.Path | None,
        pydantic.Field(
            description="Directory containing .npy shard files for training (shards should be named train_0.npy, "
            "train_1.npy, val_0.npy, etc.)",
        ),
    ] = None


class Train(Log, Seed, Device, Precision):
    """Settings for the `train` CLI subcommand."""

    # train settings
    input: Input = Input(txt=constants.TINYSHAKESPEARE_PATH)
    train_split: Annotated[
        float,
        pydantic.Field(gt=0.0, lt=1.0, description="Proportion of data to use for training"),
    ] = constants.TRAIN_SPLIT
    batch_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Batch size for the model"),
    ] = constants.BATCH_SIZE
    lr: Annotated[
        pydantic.PositiveFloat,
        pydantic.Field(description="Learning rate for the AdamW optimizer"),
    ] = constants.LEARNING_RATE
    steps: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of gradient updates to perform during training"),
    ] = constants.NUM_STEPS
    save_every: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(description="Checkpoint training state every N steps"),
    ] = constants.SAVE_EVERY
    val_every: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(description="Evaluate on validation set every N steps"),
    ] = constants.VAL_EVERY
    checkpoint_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Directory to save model checkpoints"),
    ] = constants.CHECKPOINT_DIR
    resume_from_checkpoint: Annotated[
        pathlib.Path | None,
        pydantic.Field(description="Checkpoint to resume training from"),
    ] = None

    # sample settings
    tokens_to_save: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of tokens to save to output file"),
    ] = constants.TOKENS_TO_SAVE

    # ddp
    ddp: DDP = DDP()

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class TrainGPT2(Train):
    """Overrides and extends base Train class with GPT-2 specific training settings."""

    # override Train
    batch_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Batch size for the model"),
    ] = constants.GPT2_BATCH_SIZE
    micro_batch_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Micro-batch size for gradient accumulation"),
    ] = constants.GPT2_MICRO_BATCH_SIZE

    # lr schedule
    min_lr: Annotated[
        pydantic.PositiveFloat,
        pydantic.Field(description="Minimum learning rate to decay to for learning rate schedule"),
    ] = constants.GPT2_MIN_LR
    max_lr: Annotated[
        pydantic.PositiveFloat,
        pydantic.Field(description="Maximum learning rate for learning rate schedule"),
    ] = constants.GPT2_MAX_LR
    warmup_lr_steps: Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(description="Number of linear warmup steps for learning rate schedule"),
    ] = constants.GPT2_WARMUP_LR_STEPS
    max_lr_steps: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Maximum number of training steps before reaching minimum learning rate"),
    ] = constants.GPT2_MAX_LR_STEPS
    weight_decay: Annotated[
        pydantic.NonNegativeFloat,
        pydantic.Field(description="Weight decay for AdamW optimizer"),
    ] = constants.GPT2_WEIGHT_DECAY

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class Convert(Log):
    """Settings for the `convert` CLI subcommand."""

    checkpoint: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("ckpt", "checkpoint"), description="Model checkpoint to convert"
        ),
    ]
    output: Annotated[
        pathlib.Path,
        pydantic.Field(validation_alias=pydantic.AliasChoices("o", "output"), description="Output model weights path"),
    ]

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class Sample(Log, Device, Precision):
    """Settings for the `sample` CLI subcommand."""

    # we don't want fixed seed for sampling but still have option
    seed: Annotated[
        int | None,
        pydantic.Field(description="Random seed for Python"),
    ] = None
    torch_seed: Annotated[
        int | None,
        pydantic.Field(description="Random seed for PyTorch"),
    ] = None

    checkpoint: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("ckpt", "checkpoint"),
            description="Full or weights-only checkpoint to sample from",
        ),
    ]
    tokenizer_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Directory containing tokenizer config"),
    ]
    n_tokens: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("n", "n_tokens"),
            description="Number of tokens to generate",
        ),
    ] = constants.SAMPLE_MAX_TOKENS
    temperature: Annotated[
        float,
        pydantic.Field(
            ge=0.0,
            le=2.0,
            validation_alias=pydantic.AliasChoices("t", "temperature"),
            description="Sampling temperature i.e., how random the sampling should be",
        ),
    ] = constants.SAMPLE_TEMPERATURE
    top_k: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("k", "top_k"),
            description="Number of top K tokens to consider for sampling. If none, no top-k filtering)",
        ),
    ] = constants.SAMPLE_TOP_K
    prompt: Annotated[
        str | None,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("p", "prompt"),
            description="Prompt text to start sampling from",
        ),
    ] = None
    stream: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(description="Stream output tokens to console as they are generated"),
    ] = True

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


HellaSwagSplit = Literal["train", "val", "test"]


class Evaluate(Log, Seed, Device, Precision):
    """Settings for the `evaluate` CLI subcommand."""

    checkpoint: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("ckpt", "checkpoint"),
            description="Full or weights-only checkpoint to evaluate on",
        ),
    ]
    split: Annotated[
        HellaSwagSplit,
        pydantic.Field(description="Dataset split to evaluate on"),
    ] = "val"

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class Report(Log, DDP):
    """Settings for the `report` CLI subcommand."""

    report_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Directory to save generated report to"),
    ] = constants.CHATGPT2_REPORT_DIR

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class Clean(Log):
    """Settings for the `clean` CLI subcommand."""

    checkpoint_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Model checkpoints directory to clean"),
    ] = constants.CHECKPOINT_DIR
    force: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("f", "force"),
            description="Force clean without user confirmation (DANGEROUS)",
        ),
    ] = False

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")
