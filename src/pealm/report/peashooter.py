# ruff: noqa: PLR2004, C901, PLR0912, PLR0915
"""
Utilities for generating Peashooter training report cards.
"""

import contextlib
import dataclasses
import datetime
import logging
import os
import pathlib
import platform
import re
import shutil
import socket
import subprocess

import psutil
import torch

from pealm import constants, settings, utils

logger = logging.getLogger(__name__)


def run_command(cmd: str) -> str | None:
    """Run a shell command and return output, or None if it fails."""
    with contextlib.suppress(Exception):
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)  # noqa: S603
        # return stdout if we got output (even if some files in xargs failed)
        if result.stdout.strip():
            return result.stdout.strip()
        if result.returncode == 0:
            return ""
    return None


@dataclasses.dataclass
class GitInfo:
    commit: str
    branch: str
    dirty: bool
    message: str


def get_git_info() -> GitInfo:
    """Get current git commit, branch, and dirty status."""
    return GitInfo(
        commit=run_command("git rev-parse --short HEAD") or "unknown",
        branch=run_command("git rev-parse --abbrev-ref HEAD") or "unknown",
        dirty=bool(run_command("git status --porcelain")),
        message=(run_command("git log -1 --pretty=%B") or "").split("\n")[0][:80],  # first line, truncated
    )


@dataclasses.dataclass
class GpuInfo:
    count: int
    names: list[str]
    memory_gb: list[float]
    cuda_version: str


def get_gpu_info() -> GpuInfo | None:
    if not torch.cuda.is_available():
        return None

    num_devices = torch.cuda.device_count()
    info = GpuInfo(count=num_devices, names=[], memory_gb=[], cuda_version=torch.version.cuda or "unknown")
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info.names.append(props.name)
        info.memory_gb.append(props.total_memory / (1024**3))
    return info


@dataclasses.dataclass
class SystemInfo:
    hostname: str
    platform: str
    python_version: str
    torch_version: str
    cpu_count: int
    cpu_count_logical: int
    memory_gb: float
    user: str


def get_system_info() -> SystemInfo:
    return SystemInfo(
        hostname=socket.gethostname(),
        platform=platform.system(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cpu_count=psutil.cpu_count(logical=False),
        cpu_count_logical=psutil.cpu_count(logical=True),
        memory_gb=psutil.virtual_memory().total / (1024**3),
        user=os.environ.get("USER", "unknown"),
    )


@dataclasses.dataclass
class CostInfo:
    hourly_rate: float
    gpu_type: str
    estimated_total: float | None


def estimate_cost(gpu_info: GpuInfo, runtime_hours: int | None = None) -> CostInfo:
    """Estimate training cost based on GPU type and runtime."""

    # rough pricing, from Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    # try to identify GPU type from name
    hourly_rate = default_rate * gpu_info.count  # default estimate
    gpu_name = gpu_info.names[0] if gpu_info.names else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info.count
            break

    return CostInfo(
        hourly_rate=hourly_rate,
        gpu_type=gpu_name,
        estimated_total=hourly_rate * runtime_hours if runtime_hours else None,
    )


def generate_header() -> str:
    """Generate the header for a training report."""
    timestamp = utils.current_dt_human()
    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info) if gpu_info else None

    header = f"""# peashooter training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info.branch}
- Commit: {git_info.commit} {"(dirty)" if git_info.dirty else "(clean)"}
- Message: {git_info.message}

### Hardware
- Platform: {sys_info.platform}
- CPUs: {sys_info.cpu_count} cores ({sys_info.cpu_count_logical} logical)
- Memory: {sys_info.memory_gb:.1f} GB
"""

    if gpu_info is not None:
        gpu_names = ", ".join(set(gpu_info.names))
        total_vram = sum(gpu_info.memory_gb)
        header += f"""- GPUs: {gpu_info.count}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info.cuda_version}
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info.hourly_rate > 0:
        header += f"""- Hourly Rate: ${cost_info.hourly_rate:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info.python_version}
- PyTorch: {sys_info.torch_version}

"""

    # bloat metrics: count lines/chars in git-tracked source files only
    extensions = ["py", "md", "rs", "html", "toml", "sh"]
    git_patterns = " ".join(f"'*.{ext}'" for ext in extensions)
    files_output = run_command(f"git ls-files -- {git_patterns}")
    file_list = [f for f in (files_output or "").split("\n") if f]
    num_files = len(file_list)
    num_lines = 0
    num_chars = 0
    if num_files > 0:
        wc_output = run_command(f"git ls-files -- {git_patterns} | xargs wc -lc 2>/dev/null")
        if wc_output:
            total_line = wc_output.strip().split("\n")[-1]
            parts = total_line.split()
            if len(parts) >= 2:
                num_lines = int(parts[0])
                num_chars = int(parts[1])
    num_tokens = num_chars // 4  # assume approximately 4 chars per token

    # count dependencies via uv.lock
    uv_lock_lines = 0
    uv_lock = pathlib.Path("uv.lock")
    if uv_lock.exists():
        with uv_lock.open("r", encoding="utf-8") as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    return header


def slugify(text: str) -> str:
    """Slugify a text string."""
    return text.lower().replace(" ", "-")


def extract(section: str, keys: list[str] | str) -> dict:
    """simple def to extract a single key from a section"""
    if not isinstance(keys, list):
        keys = [keys]  # convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out


def extract_timestamp(content: str, prefix: str) -> datetime.datetime | None:
    """Extract timestamp from content with given prefix."""
    for line in content.split("\n"):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            return utils.parse_dt_human(time_str)
    return None


class Report:
    """Maintains a bunch of logs, generates a final markdown report."""

    def __init__(self, report_dir: pathlib.Path) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section: str, data: list[str | dict]) -> None:
        """Log a section of data to the report."""
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = self.report_dir / file_name
        with file_path.open("w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {utils.current_dt_human()}\n\n")
            for item in data:
                if not item:
                    # skip falsy values like None or empty dict etc.
                    continue
                if isinstance(item, str):
                    # directly write the string
                    f.write(item)
                else:
                    # render a dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")

    def generate(self) -> None:
        """Generate the final report."""
        report_dir = self.report_dir
        report_file = report_dir / constants.PS_REPORT_FILENAME
        logger.info("generating report to %s", report_file)
        final_metrics = {}  # the most important final metrics we'll add as table at the end
        start_time = None
        end_time = None
        with report_file.open("w", encoding="utf-8") as out_file:
            # write the header first
            header_file = report_dir / constants.PS_REPORT_HEADER_FILENAME
            if header_file.exists():
                with header_file.open("r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    # capture bloat data for summary later (the stuff after Bloat header and until \n\n)
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None  # will cause us to not write the total wall clock time
                bloat_data = "[bloat data missing]"
                logger.warning("%s does not exist, did you forget to run `pealm report reset`?", header_file)
            # process all the individual sections
            for file_name in constants.PS_REPORTS:
                section_file = report_dir / file_name
                if not section_file.exists():
                    logger.warning("%s does not exist, skipping", section_file)
                    continue
                with section_file.open("r", encoding="utf-8") as in_file:
                    section = in_file.read()
                # extract timestamp from this section (the last section's timestamp will "stick" as end_time)
                if "rl" not in file_name:
                    # Skip RL sections for end_time calculation because RL is experimental
                    end_time = extract_timestamp(section, "timestamp:")
                # extract the most important metrics from the sections
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, constants.PS_CHAT_METRICS)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K")  # RL only evals GSM8K
                # append this section of the report
                out_file.write(section)
                out_file.write("\n")
            # add the final metrics table
            out_file.write("## Summary\n\n")
            # copy over the bloat metrics from the header
            out_file.write(bloat_data)
            out_file.write("\n\n")
            # collect all unique metric names
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # custom ordering: CORE first, ChatCORE last, rest in middle
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            # fixed column widths
            stages = ["base", "sft", "rl"]
            metric_width = 15
            value_width = 8
            # write table header
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            # Write separator
            separator = f"|{'-' * (metric_width + 2)}|"
            for _ in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # Write table rows
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # Calculate and write total wall clock time
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        # also cp the report file to current directory
        logger.info("copying report to current directory for convenience")
        shutil.copy(report_file, constants.PS_REPORT_FILENAME)

    def reset(self) -> None:
        """Reset the report."""
        # remove section files
        for file_name in constants.PS_REPORTS:
            file_path = self.report_dir / file_name
            if file_path.exists():
                file_path.unlink()
        # remove report if it exists
        report_file = self.report_dir / constants.PS_REPORT_FILENAME
        if report_file.exists():
            report_file.unlink()
        # generate and write the header section with start timestamp
        header_file = self.report_dir / constants.PS_REPORT_HEADER_FILENAME
        header = generate_header()
        start_time = utils.current_dt_human()
        with header_file.open("w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        logger.info("report reset, header written to %s", header_file)


class DistReport(Report):
    """A Report that only logs on rank 0, does nothing on other ranks."""

    def __init__(self, report_settings: settings.Report) -> None:
        super().__init__(report_settings.report_dir)
        self.is_master_process = report_settings.ddp.is_master_process

    def log(self, section: str, data: list[str | dict]) -> None:
        if not self.is_master_process:
            return

        super().log(section, data)

    def generate(self) -> None:
        if not self.is_master_process:
            return

        super().generate()

    def reset(self) -> None:
        if not self.is_master_process:
            return

        super().reset()
