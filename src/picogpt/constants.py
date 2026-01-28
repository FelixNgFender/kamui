import pathlib

# general
DATEFMT_STR = "%Y%m%d_%H%M%S"

# training settings
USE_ACCELERATOR = True
TRAIN_SPLIT = 0.9
VAL_SPLIT = 1.0 - TRAIN_SPLIT
INPUT_FILE = pathlib.Path("data") / "tinyshakespeare.txt"

# hyperparams
TORCH_SEED = 2147483647
SEED = 42
BATCH_SIZE = 64  # the number of independent sequences to process at once
LEARNING_RATE = 3e-4
NUM_EPOCHS = 2
CONTEXT_SIZE = 256  # the maximum length of predictions
EMBEDDING_SIZE = 384
TRANSFORMER_NUM_HEADS = 6
TRANSFORMER_NUM_BLOCKS = 6
TRANSFORMER_FEEDFORWARD_PROJECTION_FACTOR = 4
TRANSFORMER_DROPOUT = 0.2

# sample during training
TOKENS_TO_GENERATE = 500
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
