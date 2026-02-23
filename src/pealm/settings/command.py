import pathlib
from typing import Annotated, ClassVar, Literal

import pydantic
import pydantic_settings as ps

from pealm import constants
from pealm.settings import primitive


class Train(primitive.Log, primitive.Seed, primitive.Device, primitive.Precision):
    """Settings for the `train` CLI subcommand."""

    # train settings
    train_split: Annotated[
        pydantic.PositiveFloat,
        pydantic.Field(lt=1.0, description="Proportion of data to use for training"),
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
    ckpt_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Directory to save model checkpoints"),
    ] = constants.CKPT_DIR
    resume_from_ckpt: Annotated[
        pathlib.Path | None,
        pydantic.Field(description="Checkpoint to resume training from"),
    ] = None

    # sample settings
    tokens_to_save: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of tokens to save to output file"),
    ] = constants.TOKENS_TO_SAVE

    # ddp
    ddp: primitive.DDP = primitive.DDP()


class TrainCharBigram(Train):
    """CharBigram specific training settings."""

    input: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("i", "input"),
            description="Input text file for training",
        ),
    ] = constants.TINYSHAKESPEARE_PATH


class TrainCharTransformer(Train):
    """CharTransformer specific training settings."""

    input: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("i", "input"),
            description="Input text file for training",
        ),
    ] = constants.TINYSHAKESPEARE_PATH


class TrainGPT2(Train):
    """GPT-2 specific training settings."""

    input_dir: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("i", "input_dir"),
            description="Input numpy shards directory for training",
        ),
    ] = constants.FINEWEB_EDU10B_DIR

    # override Train
    batch_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Batch size for the model"),
    ] = constants.GPT2_BATCH_SIZE
    micro_batch_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Micro-batch size for gradient accumulation"),
    ] = constants.GPT2_MICRO_BATCH_SIZE

    @pydantic.model_validator(mode="after")
    def validate_batch_sizes(self) -> "TrainGPT2":
        batch_size = self.batch_size
        world_size = self.ddp.world_size
        if batch_size % world_size != 0:
            msg = f"batch size {batch_size} must be divisible by world size {world_size}"
            raise ValueError(msg)

        rank_batch_size = batch_size // world_size
        micro_batch_size = self.micro_batch_size
        if rank_batch_size % micro_batch_size != 0:
            msg = f"rank batch size {rank_batch_size} must be divisible by micro batch size {micro_batch_size}"
            raise ValueError(msg)

        return self

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


class TrainPeashooterTokenizer(primitive.Log, primitive.Seed):
    """Peashooter tokenizer-specific training settings."""

    input_dir: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("i", "input_dir"),
            description="Input parquet shards directory for training",
        ),
    ] = constants.PS_BASE_DATA_DIR
    output_dir: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("o", "output_dir"),
            description="Output directory for saving the trained tokenizer",
        ),
    ] = constants.PS_TOKENIZER_DIR
    max_chars: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(description="Maximum characters to train on. If none, train on the entire train split."),
    ] = constants.PS_TOKENIZER_MAX_CHARS
    max_chars_per_doc: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(description="Maximum characters per document. If none, train on the entire document."),
    ] = constants.PS_TOKENIZER_MAX_CHARS_PER_DOC


HellaSwagSplit = Literal["train", "val", "test"]


class Eval(primitive.Log, primitive.Seed, primitive.Device, primitive.Precision):
    """Settings for the `evaluate` CLI subcommand."""

    ckpt: Annotated[
        pathlib.Path,
        pydantic.Field(
            description="Full or weights-only checkpoint to evaluate on",
        ),
    ]
    input_dir: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("i", "input_dir"),
            description="Input dataset directory to evaluate on (split has to be named {split}.jsonl)",
        ),
    ] = constants.HELLASWAG_DIR
    split: Annotated[
        HellaSwagSplit,
        pydantic.Field(description="Dataset split to evaluate on"),
    ] = "val"


class EvalGPT2Pretrained(Eval):
    """GPT2 pretrained-specific eval settings"""

    # override from Evaluate: downloads from HF
    ckpt: ClassVar[None] = None


class EvalPeashooterTokenizer(primitive.Log, primitive.Seed):
    """Peashooter tokenizer-specific eval settings."""

    input_dir: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("i", "input_dir"),
            description="Input parquet shards directory for training",
        ),
    ] = constants.PS_BASE_DATA_DIR
    tokenizer_dir: Annotated[
        pathlib.Path,
        pydantic.Field(
            description="Directory containing tokenizer config to evaluate with",
        ),
    ] = constants.PS_TOKENIZER_DIR


class Chat(primitive.Log, primitive.Device, primitive.Precision):
    """Settings for the `chat` CLI subcommand."""

    # we don't want fixed seed for sampling but still have option
    seed: Annotated[
        int | None,
        pydantic.Field(description="Random seed for Python"),
    ] = None
    torch_seed: Annotated[
        int | None,
        pydantic.Field(description="Random seed for PyTorch"),
    ] = None

    ckpt: Annotated[
        pathlib.Path,
        pydantic.Field(
            description="Full or weights-only checkpoint",
        ),
    ]
    tokenizer_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Directory containing tokenizer config"),
    ] = constants.PS_TOKENIZER_DIR

    max_tokens: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("n", "max_tokens"),
            description="Maximum number of tokens to generate",
        ),
    ] = constants.CHAT_MAX_TOKENS
    temperature: Annotated[
        pydantic.NonNegativeFloat,
        pydantic.Field(
            le=constants.MAX_TEMPERATURE,
            validation_alias=pydantic.AliasChoices("t", "temperature"),
            description="Generation temperature i.e., how random the token sampling should be",
        ),
    ] = constants.CHAT_TEMPERATURE
    top_k: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("k", "top_k"),
            description="Number of top K tokens to consider for sampling. If none, no top-k filtering",
        ),
    ] = constants.CHAT_TOP_K


class ChatCli(Chat):
    """CLI-specific chat settings"""

    prompt: Annotated[
        str | None,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("p", "prompt"),
            description="Prompt the model to get a single response back",
        ),
    ] = None


class ChatWeb(Chat):
    """Web-specific chat settings"""

    num_gpus: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            description=(
                "Number of GPUs to use for workers. If more than 1, each GPU will keep a model replica."
                "If none, use all available GPUs"
            )
        ),
    ] = None
    port: Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("p", "port"),
            le=constants.MAX_PORT,
            description="Port to run the server on",
        ),
    ] = constants.CHAT_WEB_PORT
    host: Annotated[str | pydantic.IPvAnyAddress, pydantic.Field(description="Host to bind the server to")] = (
        constants.CHAT_WEB_HOST
    )


class Sample(primitive.Log, primitive.Device, primitive.Precision):
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

    ckpt: Annotated[
        pathlib.Path,
        pydantic.Field(
            description="Full or weights-only checkpoint to sample from",
        ),
    ]
    tokenizer_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Directory containing tokenizer config"),
    ]
    max_tokens: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("n", "max_tokens"),
            description="Maximum number of tokens to generate",
        ),
    ] = constants.SAMPLE_MAX_TOKENS
    temperature: Annotated[
        pydantic.NonNegativeFloat,
        pydantic.Field(
            le=constants.MAX_TEMPERATURE,
            validation_alias=pydantic.AliasChoices("t", "temperature"),
            description="Sampling temperature i.e., how random the sampling should be",
        ),
    ] = constants.SAMPLE_TEMPERATURE
    top_k: Annotated[
        pydantic.PositiveInt | None,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("k", "top_k"),
            description="Number of top K tokens to consider for sampling. If none, no top-k filtering",
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


class SampleGPT2Pretrained(Sample):
    """GPT2 pretrained-specific sample settings"""

    # override from Sample: downloads from HF
    ckpt: ClassVar[None] = None
    # override from Sample: GPT2 uses a built-in tokenizer
    tokenizer_dir: ClassVar[None] = None


class SampleGPT2(Sample):
    """GPT2-specific sample settings"""

    # override from Sample: GPT2 uses a built-in tokenizer
    tokenizer_dir: ClassVar[None] = None


class Report(primitive.Log):
    """Settings for the `report` CLI subcommand."""

    report_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Directory to save generated report to"),
    ] = constants.PS_REPORT_DIR

    # ddp
    ddp: primitive.DDP = primitive.DDP()


class Convert(primitive.Log):
    """Settings for the `convert` CLI subcommand."""

    ckpt: Annotated[
        pathlib.Path,
        pydantic.Field(description="Model checkpoint to convert"),
    ]
    output: Annotated[
        pathlib.Path,
        pydantic.Field(validation_alias=pydantic.AliasChoices("o", "output"), description="Output model weights path"),
    ]


class Clean(primitive.Log):
    """Settings for the `clean` CLI subcommand."""

    ckpt_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Model checkpoints directory to clean"),
    ] = constants.CKPT_DIR
    force: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("f", "force"),
            description="Force clean without user confirmation (DANGEROUS)",
        ),
    ] = False
