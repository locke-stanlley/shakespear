"""
config.py - Central configuration for Project Bard.
All hyperparameters, paths, and constants live here.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

for d in [RAW_DIR, CLEAN_DIR, TOKENIZER_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_TEXT_PATH = RAW_DIR / "shakespeare.txt"
CLEAN_TEXT_PATH = CLEAN_DIR / "shakespeare_clean.txt"
TOKEN_IDS_PATH = CLEAN_DIR / "token_ids.bin"
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

SHAKESPEARE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"

# -----------------------------
# Tokenizer
# -----------------------------
VOCAB_SIZE = 1024            # Small corpus -> small vocab; enterprise uses 32k-128k
SPECIAL_TOKENS: List[str] = field(default_factory=lambda: ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]) if False else ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

# -----------------------------
# Data
# -----------------------------
BLOCK_SIZE = 256             # Context window (enterprise: 2048-4096)
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025
TEST_RATIO = 0.025

# -----------------------------
# Model Architecture
# -----------------------------
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
HEAD_DIM = N_EMBD // N_HEAD
MLP_HIDDEN = 4 * N_EMBD
DROPOUT = 0.1
USE_ROPE = True
USE_RMSNORM = True
ROPE_THETA = 10000.0

# -----------------------------
# Training
# -----------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0
WARMUP_STEPS = 100
MAX_STEPS = 3000
MIN_LR_RATIO = 0.1          # Final LR = LR * MIN_LR_RATIO
LOG_INTERVAL = 20
EVAL_INTERVAL = 200
SAVE_INTERVAL = 500
SEED = 42
DTYPE = "bfloat16"          # Use "float16" if bfloat16 unsupported
DEVICE = "cuda"

# -----------------------------
# SFT / DPO
# -----------------------------
SFT_DATA_PATH = DATA_DIR / "sft_shakespeare.jsonl"
DPO_DATA_PATH = DATA_DIR / "dpo_shakespeare.jsonl"
SFT_EPOCHS = 2
SFT_LR = 1e-5
DPO_LR = 5e-6
DPO_BETA = 0.1