import logging
import pathlib
import shutil
from typing import Annotated, ClassVar

import pydantic
import pydantic_settings as ps
import rich.logging
import rich.prompt

from picogpt import convert, model, sample, settings, training

logger = logging.getLogger(__name__)


def configure_logging(log_settings: settings.Log) -> None:
    logging.basicConfig(
        level=logging.DEBUG if log_settings.verbose else logging.INFO,
        format="%(message)s",
        handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
    )
    logger.debug("running with settings %s", log_settings)


class TrainCharBigram(settings.Train, settings.CharBigram):
    """Trains a character-level bigram model."""

    def cli_cmd(self) -> None:
        training.train(self, self)


class TrainCharTransformer(settings.Train, settings.CharTransformer):
    """Trains a character-level transformer model."""

    def cli_cmd(self) -> None:
        training.train(self, self)


# python's mro uses c3 linearization "first come first serve", so GPT2 must come first
# to override any conflicting settings
class TrainGPT2(settings.GPT2, settings.Train):
    """Trains the GPT2 (124M) model from OpenAI."""

    def cli_cmd(self) -> None:
        training.train(self, self)


class Train(settings.Log):
    """Trains a model on the input file, resuming from the latest checkpoint if needed."""

    char_bigram: ps.CliSubCommand[TrainCharBigram]
    char_transformer: ps.CliSubCommand[TrainCharTransformer]
    gpt2: ps.CliSubCommand[TrainGPT2]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class Convert(settings.Convert):
    """Converts a full checkpoint to weights-only checkpoints."""

    def cli_cmd(self) -> None:
        configure_logging(self)
        convert.ckpt_to_weights(
            checkpoint_path=self.checkpoint,
            output_path=self.output,
        )


class SampleCharBigram(settings.Sample, settings.CharBigram):
    """Samples a character-level bigram model."""

    def cli_cmd(self) -> None:
        sample.sample(self, self)


class SampleCharTransformer(settings.Sample, settings.CharTransformer):
    """Samples a character-level transformer model."""

    def cli_cmd(self) -> None:
        sample.sample(self, self)


class SampleGPT2(settings.Sample, settings.GPT2):
    """Samples the GPT2 (124M) model from OpenAI."""

    # override
    checkpoint: Annotated[
        pathlib.Path | None,
        pydantic.Field(
            description="Weights-only checkpoint to sample from. If none, downloads and uses the pre-trained "
            "GPT2 weights."
        ),
    ] = None
    # override: GPT2 uses a built-in tokenizer
    tokenizer_dir: ClassVar[None] = None

    def cli_cmd(self) -> None:
        sample.sample(self, self)


class Sample(settings.Log):
    """Samples text from a trained model from its weights and tokenizer."""

    char_bigram: ps.CliSubCommand[SampleCharBigram]
    char_transformer: ps.CliSubCommand[SampleCharTransformer]
    gpt2: ps.CliSubCommand[SampleGPT2]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class Clean(settings.Clean):
    """Cleans training artifacts."""

    def cli_cmd(self) -> None:
        configure_logging(self)

        model_artifact_dirs = [self.checkpoint_dir / model_type for model_type in model.Type]
        if self.force or rich.prompt.Confirm.ask(
            f"delete all model artifact directories: {', '.join(str(d) for d in model_artifact_dirs)}? THIS ACTION "
            "CANNOT BE UNDONE.",
            default=False,
        ):
            for dir_path in model_artifact_dirs:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    logger.info("deleted directory: %s", dir_path)
                else:
                    logger.info("directory does not exist, skipping: %s", dir_path)


class Command(
    ps.BaseSettings,
    cli_parse_args=True,
    cli_use_class_docs_for_groups=True,
    cli_kebab_case=True,
):
    """PicoGPT CLI for pre-training and sampling from a tiny GPT model."""

    train: ps.CliSubCommand[Train]
    sample: ps.CliSubCommand[Sample]
    convert: ps.CliSubCommand[Convert]
    clean: ps.CliSubCommand[Clean]

    def cli_cmd(self) -> None:
        ps.CliApp.run_subcommand(self)


def main() -> None:
    ps.CliApp.run(Command)


if __name__ == "__main__":
    main()
