"""
config.py - T4-optimized configuration for ~100M parameter model
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
LOG_DIR = PROJECT_ROOT / "logs"

for d in [RAW_DIR, CLEAN_DIR, TOKENIZER_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_TEXT_PATH = RAW_DIR / "shakespeare.txt"
CLEAN_TEXT_PATH = CLEAN_DIR / "shakespeare_clean.txt"
TOKEN_IDS_PATH = CLEAN_DIR / "token_ids.bin"
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Multiple data sources for richer training
DATA_SOURCES = [
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",  # Shakespeare
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",    # Frankenstein
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt", # Pride & Prejudice
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",    # Alice in Wonderland
    "https://www.gutenberg.org/cache/epub/996/pg996.txt",  # Don Quixote
    "https://www.gutenberg.org/cache/epub/1661/pg1661.txt", # Sherlock Holmes
]

# -----------------------------
# Tokenizer (Larger vocab)
# -----------------------------
VOCAB_SIZE = 8192          # 8x larger than before
SPECIAL_TOKENS: List[str] = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

# -----------------------------
# Data
# -----------------------------
BLOCK_SIZE = 512           # Larger context window
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025
TEST_RATIO = 0.025
NUM_WORKERS = 2            # For DataLoader

# -----------------------------
# Model Architecture (~100M params)
# -----------------------------
N_LAYER = 12               # Doubled
N_HEAD = 12                # Doubled
N_EMBD = 768               # Doubled
HEAD_DIM = N_EMBD // N_HEAD
MLP_HIDDEN = int(2.6667 * N_EMBD)  # SwiGLU ratio (8/3 * embd, rounded to multiple of 64)
MLP_HIDDEN = ((MLP_HIDDEN + 63) // 64) * 64  # Round up to multiple of 64 for efficiency
DROPOUT = 0.1
USE_ROPE = True
USE_RMSNORM = True
USE_SWIGLU = True          # NEW: SwiGLU activation
USE_GRAD_CHECKPOINT = True # NEW: Fits larger models
USE_FLASH_ATTN = True      # NEW: PyTorch SDPA (works on T4)
ROPE_THETA = 10000.0

# -----------------------------
# Training (T4-optimized)
# -----------------------------
BATCH_SIZE = 8             # Smaller per-step
GRAD_ACCUM_STEPS = 8       # Effective batch = 64
BATCH_SIZE_EFFECTIVE = BATCH_SIZE * GRAD_ACCUM_STEPS
NUM_EPOCHS = 3
LEARNING_RATE = 1.5e-4     # FIXED: More conservative for 91M params
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
MAX_STEPS = 5000           # More training
MIN_LR_RATIO = 0.1
LOG_INTERVAL = 20
EVAL_INTERVAL = 250
SAVE_INTERVAL = 500
SEED = 42
DTYPE = "float16"          # CRITICAL: T4 uses fp16, not bf16
DEVICE = "cuda"

# -----------------------------
# Generation
# -----------------------------
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 50
GEN_TOP_P = 0.95           # NEW: nucleus sampling
GEN_REP_PENALTY = 1.1      # NEW: repetition penalty
GEN_MAX_NEW_TOKENS = 500

# -----------------------------
# SFT / DPO
# -----------------------------
SFT_DATA_PATH = DATA_DIR / "sft_shakespeare.jsonl"
DPO_DATA_PATH = DATA_DIR / "dpo_shakespeare.jsonl"
SFT_EPOCHS = 2
SFT_LR = 1e-5
DPO_LR = 5e-6
DPO_BETA = 0.1

# -----------------------------
# WandB (optional)
# -----------------------------
USE_WANDB = True
WANDB_PROJECT = "project-bard"
WANDB_ENTITY = "lockestanley-blueribbonsinvest"  # YOUR ENTITY NAME