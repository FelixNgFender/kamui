import pathlib

# general
DATEFMT_STR = "%Y%m%d_%H%M%S"

# training settings
USE_ACCELERATOR = True
TRAIN_SPLIT = 0.9
VAL_SPLIT = 1.0 - TRAIN_SPLIT
DATA_DIR = pathlib.Path("data")
TINYSHAKESPEARE_PATH = DATA_DIR / "tinyshakespeare.txt"
HELLASWAG_DIR = DATA_DIR / "hellaswag"
TORCH_SEED = 2147483647
SEED = 42
# https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
FP32_MATMUL_PRECISION = "high"  # "highest", "high", "medium"
USE_MIXED_PRECISION = True
# ddp
DDP_RANK = 0
DDP_LOCAL_RANK = 0
DDP_WORLD_SIZE = 1

# hyperparams
BATCH_SIZE = 64  # the number of independent sequences to process at once
LEARNING_RATE = 3e-4
NUM_EPOCHS = 2
CONTEXT_SIZE = 256  # the maximum length of predictions

TRANSFORMER_EMBEDDING_SIZE = 384
TRANSFORMER_NUM_HEADS = 6
TRANSFORMER_NUM_BLOCKS = 6
TRANSFORMER_FEEDFORWARD_PROJECTION_FACTOR = 4
TRANSFORMER_DROPOUT = 0.2
GPT2_CONTEXT_SIZE = 1024
# 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>, round up to nearest multiple of 64 for cuda niceness
GPT2_VOCAB_SIZE = 50304
GPT2_EMBEDDING_SIZE = 768
GPT2_NUM_LAYERS = 12
GPT2_NUM_HEADS = 12
GPT2_FEEDFORWARD_PROJECTION_FACTOR = 4

# gpt-2 specific training hparams
GPT2_BATCH_SIZE_TOKENS = 2**19
GPT2_BATCH_SIZE = GPT2_BATCH_SIZE_TOKENS // GPT2_CONTEXT_SIZE
GPT2_MICRO_BATCH_SIZE = 16

GPT2_MAX_LR = 6e-4
GPT2_MIN_LR = GPT2_MAX_LR * 0.1
GPT2_WARMUP_STEPS = 10  # before decaying lr
GPT2_MAX_STEPS = 50  # before returning min_lr
GPT2_WEIGHT_DECAY = 0.1

# sample during training
TOKENS_TO_SAVE = 10000

# checkpointing
SAVE_EVERY_N_EPOCHS = 1  # save checkpoint every N epochs
CHECKPOINT_DIR = pathlib.Path("checkpoints")
TOKENIZER_DIR = pathlib.Path("tokenizer")
TOKENIZER_FILENAME = "tokenizer.json"
VOCAB_FILENAME = "vocab.json"
LATEST_CKPT_FILENAME = "latest.pt"
FINAL_CKPT_FILENAME = "final.pt"
BEST_CKPT_FILENAME = "best.pt"
LOSS_PLOT_FILENAME = "loss.png"
SAMPLE_OUTPUT_FILENAME = "sample.txt"

# sample
SAMPLE_MAX_TOKENS = 500
SAMPLE_TEMPERATURE = 1.0
SAMPLE_TOP_K = 50

# gpt2 pretrained
GPT2_PRETRAINED_CONFIG: dict[str, dict[str, int]] = {
    "gpt2": {"num_layers": 12, "num_heads": 12, "embedding_size": 768},  # 124M params
    "gpt2-medium": {"num_layers": 24, "num_heads": 16, "embedding_size": 1024},  # 350M params
    "gpt2-large": {"num_layers": 36, "num_heads": 20, "embedding_size": 1280},  # 774M params
    "gpt2-xl": {"num_layers": 48, "num_heads": 25, "embedding_size": 1600},  # 1558M params
}
GPT2_PRETRAINED_VOCAB_SIZE = 50257
GPT2_PRETRAINED_CONTEXT_SIZE = 1024
GPT2_PRETRAINED_FFW_PROJECTION_FACTOR = 4
