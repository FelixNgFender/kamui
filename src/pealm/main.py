import logging
import os
import shutil
from typing import ClassVar

import pydantic
import pydantic_settings as ps
import rich.logging
import rich.prompt

from pealm import convert, models, report, sample, settings, training
from pealm import eval as eval_mod

logger = logging.getLogger(__name__)


def configure_logging(log_settings: settings.Log) -> None:
    class DDPRankFilter(logging.Filter):
        """Only allows logs from rank 0 (master process) in DDP training."""

        def filter(self, record: logging.LogRecord) -> bool:  # noqa: ARG002
            rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
            # not in ddp, allow all
            if rank is None:
                return True
            # only allow logs from rank 0
            return int(rank) == 0

    logging.basicConfig(
        level=logging.DEBUG if log_settings.verbose else logging.INFO,
        format="%(message)s",
        handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
    )
    # add rank filter to root logger handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(DDPRankFilter())
    logger.debug("running with settings %s", log_settings)


class TrainCharBigram(settings.Train, settings.CharBigram):
    """Trains a character-level bigram model."""

    def cli_cmd(self) -> None:
        training.train(self, self)


class TrainCharTransformer(settings.Train, settings.CharTransformer):
    """Trains a character-level transformer model."""

    def cli_cmd(self) -> None:
        training.train(self, self)


class TrainGPT2(settings.GPT2, settings.TrainGPT2):
    """Trains the GPT2 (124M) model from Felix Nguyen."""

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


class SampleGPT2Pretrained(settings.Sample):
    """Samples the GPT2 (124M) model from OpenAI."""

    # override from Sample: downloads from HF
    checkpoint: ClassVar[None] = None
    # override from Sample: GPT2 uses a built-in tokenizer
    tokenizer_dir: ClassVar[None] = None

    def cli_cmd(self) -> None:
        # use default GPT2 to typecheck, not gonna be used anyway
        # when sampling pretrained
        sample.sample(self, settings.GPT2())


class SampleGPT2(settings.Sample, settings.GPT2):
    """Samples the GPT2 (124M) model from Felix Nguyen."""

    # override from Sample: GPT2 uses a built-in tokenizer
    tokenizer_dir: ClassVar[None] = None

    def cli_cmd(self) -> None:
        sample.sample(self, self)


class Sample(settings.Log):
    """Samples text from a trained model from its weights and tokenizer."""

    char_bigram: ps.CliSubCommand[SampleCharBigram]
    char_transformer: ps.CliSubCommand[SampleCharTransformer]
    gpt2_pretrained: ps.CliSubCommand[SampleGPT2Pretrained]
    gpt2: ps.CliSubCommand[SampleGPT2]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class EvalGPT2Pretrained(settings.Evaluate):
    """Evaluates the GPT2 (124M) model from OpenAI on HellaSwag."""

    # override from Evaluate: downloads from HF
    checkpoint: ClassVar[None] = None

    def cli_cmd(self) -> None:
        # use default GPT2 to typecheck, not gonna be used anyway
        # when sampling pretrained
        eval_mod.evaluate(self, settings.GPT2())


class EvalGPT2(settings.Evaluate, settings.GPT2):
    """Evaluates the GPT2 (124M) model from Felix Nguyen on HellaSwag."""

    def cli_cmd(self) -> None:
        eval_mod.evaluate(self, self)


class Eval(settings.Log):
    """Evaluates model on HellaSwag."""

    gpt2_pretrained: ps.CliSubCommand[EvalGPT2Pretrained]
    gpt2: ps.CliSubCommand[EvalGPT2]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class ReportGenerate(settings.Report):
    """Generates the final report."""

    def cli_cmd(self) -> None:
        report.DDPReport(self).generate()


class ReportReset(settings.Report):
    """Resets the final report."""

    def cli_cmd(self) -> None:
        report.DDPReport(self).reset()


class Report(settings.Log):
    """Generate or reset Peashooter training reports."""

    generate: ps.CliSubCommand[ReportGenerate]
    reset: ps.CliSubCommand[ReportReset]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class Clean(settings.Clean):
    """Cleans training artifacts."""

    def cli_cmd(self) -> None:
        configure_logging(self)

        model_artifact_dirs = [self.checkpoint_dir / model_type for model_type in models.Type]
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
    """CLI for playing with kindergartener language models."""

    train: ps.CliSubCommand[Train]
    sample: ps.CliSubCommand[Sample]
    convert: ps.CliSubCommand[Convert]
    eval: ps.CliSubCommand[Eval]
    report: ps.CliSubCommand[Report]
    clean: ps.CliSubCommand[Clean]

    def cli_cmd(self) -> None:
        ps.CliApp.run_subcommand(self)


def main() -> None:
    ps.CliApp.run(Command)


if __name__ == "__main__":
    main()
