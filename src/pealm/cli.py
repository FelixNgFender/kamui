import logging
import os
import shutil

import pydantic_settings as ps
import rich.logging
import rich.prompt

from pealm import chat, convert, model, report, sample, settings, train
from pealm import eval as eval_mod

logger = logging.getLogger(__name__)


def configure_logging(log_settings: settings.Log) -> None:
    class DistRankFilter(logging.Filter):
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
        handler.addFilter(DistRankFilter())
    logger.debug("running with settings %s", log_settings)


class TrainCharBigram(settings.TrainCharBigram, settings.CharBigram):
    """Trains a character-level bigram model."""

    def cli_cmd(self) -> None:
        train.train_char_bigram(self, self)


class TrainCharTransformer(settings.TrainCharTransformer, settings.CharTransformer):
    """Trains a character-level transformer model."""

    def cli_cmd(self) -> None:
        train.train_char_transformer(self, self)


class TrainGPT2(settings.TrainGPT2, settings.GPT2):
    """Trains the GPT2 (124M) model from Felix Nguyen."""

    def cli_cmd(self) -> None:
        train.train_gpt2(self, self)


class TrainPeashooterTokenizer(settings.TrainPeashooterTokenizer, settings.PeashooterTokenizer, settings.Report):
    """Trains the Peashooter tokenizer."""

    def cli_cmd(self) -> None:
        _report = report.DistReport(self)
        train.train_peashooter_tokenizer(self, self, _report)


class Train(settings.Log):
    """Trains a model on the input file, resuming from the latest checkpoint if needed."""

    char_bigram: ps.CliSubCommand[TrainCharBigram]
    char_transformer: ps.CliSubCommand[TrainCharTransformer]
    gpt2: ps.CliSubCommand[TrainGPT2]
    peashooter_tokenizer: ps.CliSubCommand[TrainPeashooterTokenizer]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class EvalGPT2Pretrained(settings.EvalGPT2Pretrained, settings.GPT2Pretrained):
    """Evaluates the GPT2 (124M) model from OpenAI on HellaSwag."""

    def cli_cmd(self) -> None:
        eval_mod.eval_gpt2_pretrained(self, self)


class EvalGPT2(settings.Eval, settings.GPT2):
    """Evaluates the GPT2 (124M) model from Felix Nguyen on HellaSwag."""

    def cli_cmd(self) -> None:
        eval_mod.eval_gpt2(self, self)


class EvalPeashooterTokenizer(settings.EvalPeashooterTokenizer, settings.Report):
    """Evaluates the Peashooter tokenizer."""

    def cli_cmd(self) -> None:
        _report = report.DistReport(self)
        eval_mod.eval_peashooter_tokenizer(self, _report)


class Eval(settings.Log):
    """Evaluates model on HellaSwag."""

    gpt2_pretrained: ps.CliSubCommand[EvalGPT2Pretrained]
    gpt2: ps.CliSubCommand[EvalGPT2]
    peashooter_tokenizer: ps.CliSubCommand[EvalPeashooterTokenizer]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class ChatCli(settings.ChatCli, settings.GPT2):
    """Chat with the Peashooter model through the CLI."""

    def cli_cmd(self) -> None:
        chat.chat_cli(self, self)


class ChatWeb(settings.ChatWeb, settings.GPT2):
    """Chat with the Peashooter model through the web UI."""

    def cli_cmd(self) -> None:
        chat.chat_web(self, self)


class Chat(settings.Log):
    """Chat with the Peashooter model through the CLI or web UI."""

    cli: ps.CliSubCommand[ChatCli]
    web: ps.CliSubCommand[ChatWeb]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class SampleCharBigram(settings.Sample, settings.CharBigram):
    """Samples a character-level bigram model."""

    def cli_cmd(self) -> None:
        sample.sample_char_bigram(self, self)


class SampleCharTransformer(settings.Sample, settings.CharTransformer):
    """Samples a character-level transformer model."""

    def cli_cmd(self) -> None:
        sample.sample_char_transformer(self, self)


class SampleGPT2Pretrained(settings.SampleGPT2Pretrained, settings.GPT2Pretrained):
    """Samples the GPT2 (124M) model from OpenAI."""

    def cli_cmd(self) -> None:
        sample.sample_gpt2_pretrained(self, self)


class SampleGPT2(settings.SampleGPT2, settings.GPT2):
    """Samples the GPT2 (124M) model from Felix Nguyen."""

    def cli_cmd(self) -> None:
        sample.sample_gpt2(self, self)


class Sample(settings.Log):
    """Samples text from a trained model from its weights and tokenizer."""

    char_bigram: ps.CliSubCommand[SampleCharBigram]
    char_transformer: ps.CliSubCommand[SampleCharTransformer]
    gpt2_pretrained: ps.CliSubCommand[SampleGPT2Pretrained]
    gpt2: ps.CliSubCommand[SampleGPT2]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class ReportGenerate(settings.Report):
    """Generates the final report."""

    def cli_cmd(self) -> None:
        report.DistReport(self).generate()


class ReportReset(settings.Report):
    """Resets the final report."""

    def cli_cmd(self) -> None:
        report.DistReport(self).reset()


class Report(settings.Log):
    """Generate or reset Peashooter training reports."""

    generate: ps.CliSubCommand[ReportGenerate]
    reset: ps.CliSubCommand[ReportReset]

    def cli_cmd(self) -> None:
        configure_logging(self)
        ps.CliApp.run_subcommand(self)


class Convert(settings.Convert):
    """Converts a full checkpoint to weights-only checkpoints."""

    def cli_cmd(self) -> None:
        configure_logging(self)
        convert.ckpt_to_weights(
            checkpoint_path=self.ckpt,
            output_path=self.output,
        )


class Clean(settings.Clean):
    """Cleans training artifacts."""

    def cli_cmd(self) -> None:
        configure_logging(self)

        model_artifact_dirs = [self.ckpt_dir / model_type for model_type in model.Type]
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
    eval: ps.CliSubCommand[Eval]
    chat: ps.CliSubCommand[Chat]
    sample: ps.CliSubCommand[Sample]
    report: ps.CliSubCommand[Report]
    convert: ps.CliSubCommand[Convert]
    clean: ps.CliSubCommand[Clean]

    def cli_cmd(self) -> None:
        ps.CliApp.run_subcommand(self)


def main() -> None:
    ps.CliApp.run(Command)


if __name__ == "__main__":
    main()
