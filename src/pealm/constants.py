import enum
import pathlib

# general
DATEFMT_STR_HUMAN = "%Y-%m-%d %H:%M:%S"
DATEFMT_STR = "%Y-%m-%d_%H:%M:%S"

# training settings
USE_ACCELERATOR = True
TRAIN_SPLIT = 0.9
VAL_SPLIT = 1.0 - TRAIN_SPLIT
DATA_DIR = pathlib.Path("data")
TINYSHAKESPEARE_PATH = DATA_DIR / "tinyshakespeare.txt"
HELLASWAG_DIR = DATA_DIR / "hellaswag"
FINEWEB_EDU10B_DIR = DATA_DIR / "fineweb_edu10B"
TORCH_SEED = 2_147_483_647
SEED = 42
# https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
FP32_MATMUL_PRECISION = "high"  # "highest", "high", "medium"
USE_MIXED_PRECISION = True
# ddp
DDP_RANK = 0
DDP_LOCAL_RANK = 0
DDP_WORLD_SIZE = 1
VAL_EVERY = 250  # validate every 250 training steps

# hyperparams
BATCH_SIZE = 64  # the number of independent sequences to process at once
LEARNING_RATE = 3e-4
NUM_STEPS = 5000
CONTEXT_SIZE = 256  # the maximum length of predictions

TRANSFORMER_EMBEDDING_SIZE = 384
TRANSFORMER_NUM_HEADS = 6
TRANSFORMER_NUM_BLOCKS = 6
TRANSFORMER_FFW_PROJECTION_FACTOR = 4
TRANSFORMER_DROPOUT = 0.2
MAX_DROPOUT = 1.0
GPT2_CONTEXT_SIZE = 1024
# 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>, round up to nearest multiple of 64 for cuda niceness
GPT2_VOCAB_SIZE = 50_304
GPT2_EMBEDDING_SIZE = 768
GPT2_NUM_LAYERS = 12
GPT2_NUM_HEADS = 12

# gpt-2 specific training hparams
GPT2_BATCH_SIZE_TOKENS = 2**19
GPT2_BATCH_SIZE = GPT2_BATCH_SIZE_TOKENS // GPT2_CONTEXT_SIZE
GPT2_MICRO_BATCH_SIZE = 16

GPT2_NUM_STEPS = 19_073  # per fineweb edu epoch
GPT2_MAX_LR = 6e-4
GPT2_MIN_LR = GPT2_MAX_LR * 0.1
GPT2_WARMUP_LR_STEPS = 715  # before decaying lr
GPT2_MAX_LR_STEPS = 19_073  # before returning min_lr
GPT2_WEIGHT_DECAY = 0.1

# sample during training
TOKENS_TO_SAVE = 10_000

# checkpointing
SAVE_EVERY = 1000  # save checkpoint every 1000 training steps
CKPT_DIR = pathlib.Path("checkpoints")
TOKENIZER_DIR = pathlib.Path("tokenizer")
TOKENIZER_FILENAME = "tokenizer.json"
VOCAB_FILENAME = "vocab.json"
LATEST_CKPT_FILENAME = "latest.pt"
FINAL_CKPT_FILENAME = "final.pt"
BEST_CKPT_FILENAME = "best.pt"
STEP_CKPT_TEMPLATE = "step_{step}.pt"
LOSS_PLOT_FILENAME = "loss.png"
SAMPLE_OUTPUT_FILENAME = "sample.txt"

MAX_TEMPERATURE = 2.0

# sample
SAMPLE_MAX_TOKENS = 512
SAMPLE_TEMPERATURE = 1.0
SAMPLE_TOP_K = 50

# chat
CHAT_TEMPERATURE = 0.7
CHAT_TOP_K = 50
CHAT_MAX_TOKENS = 512
CHAT_WEB_HOST = "localhost"
CHAT_WEB_PORT = 8000

# gpt2 pretrained
GPT2_PRETRAINED_CONFIG: dict[str, dict[str, int]] = {
    "gpt2": {"num_layers": 12, "num_heads": 12, "embedding_size": 768},  # 124M params
    "gpt2-medium": {"num_layers": 24, "num_heads": 16, "embedding_size": 1024},  # 350M params
    "gpt2-large": {"num_layers": 36, "num_heads": 20, "embedding_size": 1280},  # 774M params
    "gpt2-xl": {"num_layers": 48, "num_heads": 25, "embedding_size": 1600},  # 1558M params
}
GPT2_PRETRAINED_VOCAB_SIZE = 50_257
GPT2_PRETRAINED_CONTEXT_SIZE = 1024

# peashooter
PS_BASE_DIR = pathlib.Path("peashooter")
PS_BASE_DATA_DIR = PS_BASE_DIR / "base_data"
PS_VOCAB_SIZE = 32_768  # 2**15, including special tokens

## tokenizer
PS_TOKENIZER_DIR = PS_BASE_DIR / "tokenizer"
# ruff: disable[S105]
PS_TOKENIZER_BYTES_PER_TOKEN_FILENAME = "token_bytes.pt"
PS_TOKENIZER_FILENAME = "tokenizer.pkl"


class PeashooterSpecialTokens(enum.StrEnum):
    """Unique strings that do not appear in normal text but tell the model about the structure of the conversation.
    Useful when doing SFT/RL"""

    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    BOS = "<|bos|>"
    # tokens below are only used during finetuning to render Conversations into token ids
    USER_START = "<|user_start|>"  # user messages
    USER_END = "<|user_end|>"
    ASSISTANT_START = "<|assistant_start|>"  # assistant messages
    ASSISTANT_END = "<|assistant_end|>"
    PYTHON_START = "<|python_start|>"  # assistant invokes python REPL tool
    PYTHON_END = "<|python_end|>"
    OUTPUT_START = "<|output_start|>"  # python repl outputs back to assistant
    OUTPUT_END = "<|output_end|>"


PS_SPECIAL_TOKENS: list[str] = [str(tok.value) for tok in PeashooterSpecialTokens]
# ruff: enable[S105]
# 1. contractions: '(?i:[sdmt]|ll|ve|re) matches common english contractions like "don't", "I'm", "they're", etc. the
# (?i:...) makes it case-insensitive.
#
# 2. letters and words: \p{L}+ matches sequences of letters. the [^\r\n\p{L}\p{N}]?+ prefix allows for an optional
# non-letter, non-number, non-newline character before the letters, which helps treat " hello" and "(hello" and
# ":emoji:hello" the same).
#
# 3. numbers: \p{N}{1,2} matches sequences of 1 or 2 digits. the reason for this is that we want to split off numbers
# into separate tokens, but we don't want to split off every single digit into its own token, as that would be too many
# tokens for smaller vocab sizes. 1 or 2 digits is a good sweet spot for vocab size of 32K, as it allows us to split off
# most numbers into separate tokens without wasting too many tokens on single digits. for larger vocab sizes, we could
# consider increasing this to 3 or more digits, but for our purposes 1 or 2 is sufficient.
#
# 4. punctuation and symbols: " ?[^\s\p{L}\p{N}]++[\r\n]*" matches one or more non-whitespace, non-letter, non-number
# characters (i.e., punctuation and symbols). Optional preceding space and optional trailing newlines.
#
# 5. whitespace and newlines: "\s*[\r\n]" matches optional whitespace followed by a newline character. "\s+(?!\S)"
# matches one or more whitespace characters that are not followed by a non-whitespace character (i.e., trailing
# whitespace at the end of a line). "\s+" matches any other sequence of one or more whitespace characters.
PS_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
r"""This split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}"""
PS_MIN_VOCAB_SIZE = len(PS_SPECIAL_TOKENS) + 256
PS_TOKENIZER_MAX_CHARS = 2_000_000_000  # train on 2B characters
PS_TOKENIZER_MAX_CHARS_PER_DOC = 10_000

## reports
PS_REPORT_DIR = PS_BASE_DIR / "report"
PS_REPORT_FILENAME = "report.md"
PS_REPORT_HEADER_FILENAME = "header.md"
PS_REPORTS: list[str] = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
"""expected reports and their order"""
# ARC: grade school level, multiple-choice science questions
# MMLU: Massive Multitask Language Understanding, 57 subjects from STEM to humanities
# GSM8k: 8.5K high school level math word problems, requires multi-step reasoning
# HumanEval: 164 hand-written programming problems, requires writing code and passing test cases
# ChatCORE: custom chat eval metric in nanochat that aggregates perf across main chat eval tasks
PS_CHAT_METRICS: list[str] = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]
"""the metrics measured when training peashooter"""

# chat
MAX_PORT = 65_535
PS_CHAT_WEB_MAX_MESSAGES = 500
PS_CHAT_WEB_MAX_MESSAGE_LEN = 8_000
PS_CHAT_WEB_MAX_CONVERSATION_LEN = 32_000
PS_CHAT_WEB_MAX_TOP_K = 200
PS_CHAT_WEB_MIN_TOKENS = 1
PS_CHAT_WEB_MAX_TOKENS = 4096
