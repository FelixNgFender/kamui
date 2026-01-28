import pathlib
from typing import Annotated

import pydantic
import pydantic_settings as ps

from picogpt import constants


class Log(ps.BaseSettings):
    verbose: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(description="Logs extra debugging information"),
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
    ] = constants.EMBEDDING_SIZE
    transformer_num_blocks: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer blocks"),
    ] = constants.TRANSFORMER_NUM_BLOCKS
    transformer_num_heads: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer heads"),
    ] = constants.TRANSFORMER_NUM_HEADS
    transformer_feedforward_projection_factor: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Transfomer feedforward projection factor"),
    ] = constants.TRANSFORMER_FEEDFORWARD_PROJECTION_FACTOR
    transformer_dropout: Annotated[
        float,
        pydantic.Field(ge=0.0, le=1.0, description="Transformer dropout rate"),
    ] = constants.TRANSFORMER_DROPOUT

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


Model = CharBigram | CharTransformer


class Train(Log, Seed, Device):
    """Settings for the `train` CLI subcommand."""

    # train settings
    input_file: Annotated[
        pathlib.Path,
        pydantic.Field(description="Input text file for training"),
    ] = constants.INPUT_FILE
    train_split: Annotated[
        float,
        pydantic.Field(gt=0.0, lt=1.0, description="Proportion of data to use for training"),
    ] = constants.TRAIN_SPLIT
    batch_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Batch size for the model"),
    ] = constants.BATCH_SIZE
    learning_rate: Annotated[
        pydantic.PositiveFloat,
        pydantic.Field(description="Learning rate for the AdamW optimizer"),
    ] = constants.LEARNING_RATE
    num_epochs: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of training epochs"),
    ] = constants.NUM_EPOCHS
    checkpoint_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Directory to save model checkpoints"),
    ] = constants.CHECKPOINT_DIR
    resume_from_checkpoint: Annotated[
        pathlib.Path | None,
        pydantic.Field(description="Checkpoint to resume training from"),
    ] = None
    save_every_n_epochs: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Save checkpoint every N epochs"),
    ] = constants.SAVE_EVERY_N_EPOCHS

    # sample settings
    tokens_to_generate: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of tokens to sample before and after training"),
    ] = constants.TOKENS_TO_GENERATE
    tokens_to_save: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of tokens to save to output file"),
    ] = constants.TOKENS_TO_SAVE

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class Convert(Log):
    """Settings for the `convert` CLI subcommand."""

    checkpoint: Annotated[
        pathlib.Path,
        pydantic.Field(description="Model checkpoint to convert"),
    ]
    output: Annotated[
        pathlib.Path,
        pydantic.Field(description="Output model weights path"),
    ]

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class Sample(Log, Seed, Device, ModelBase):
    """Settings for the `sample` CLI subcommand."""

    checkpoint: Annotated[
        pathlib.Path,
        pydantic.Field(description="Weights-only checkpoint to sample from"),
    ]
    tokenizer_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Directory containing tokenizer config"),
    ]
    max_tokens: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            description="Maximum number of tokens to generate",
        ),
    ] = constants.SAMPLE_MAX_TOKENS
    temperature: Annotated[
        float,
        pydantic.Field(
            gte=0.0,
            lte=2.0,
            description="Sampling temperature i.e., how random the sampling should be",
        ),
    ] = constants.SAMPLE_TEMPERATURE
    prompt: Annotated[
        str | None,
        pydantic.Field(
            description="Prompt text to start sampling from",
        ),
    ] = None

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class Clean(ps.BaseSettings):
    """Settings for the `clean` CLI subcommand."""

    checkpoint_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Model checkpoints directory to clean"),
    ] = constants.CHECKPOINT_DIR
    force: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(description="Force clean without user confirmation (DANGEROUS)"),
    ] = False

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")
