"""
tokenizer.py - Phase 2: Tokenization
Trains a BPE tokenizer on the cleaned corpus, adds special tokens,
and digitizes the text into a memory-mapped binary file.
"""
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from config import (
    CLEAN_TEXT_PATH, TOKENIZER_DIR, TOKEN_IDS_PATH,
    VOCAB_SIZE, SPECIAL_TOKENS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
)


def train_bpe_tokenizer() -> Tokenizer:
    """Train a Byte-Pair Encoding tokenizer on the cleaned corpus."""
    print("=" * 60)
    print("[PHASE 2] Tokenization")
    print("=" * 60)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print(f"[*] Training BPE (vocab_size={VOCAB_SIZE}) on {CLEAN_TEXT_PATH}")
    tokenizer.train(files=[str(CLEAN_TEXT_PATH)], trainer=trainer)

    # Wrap with ByteLevel processor to handle BOS/EOS
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Save
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"[+] Tokenizer saved: {tokenizer_path}")
    print(f"[+] Vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer


def digitize_and_save(tokenizer: Tokenizer) -> np.memmap:
    """Convert the cleaned corpus to token IDs and save as a binary memmap."""
    text = CLEAN_TEXT_PATH.read_text(encoding="utf-8")

    # Add BOS/EOS around the whole corpus for clean boundaries
    text = f"{BOS_TOKEN} {text} {EOS_TOKEN}"

    print("[*] Encoding text to token IDs...")
    encoding = tokenizer.encode(text)
    ids = np.array(encoding.ids, dtype=np.uint16)  # uint16 supports up to 65535 tokens

    # Save as raw binary for fast memmap reads
    ids.tofile(str(TOKEN_IDS_PATH))
    print(f"[+] Token IDs saved: {TOKEN_IDS_PATH} ({ids.shape[0]} tokens)")

    # Return a memory-mapped view (zero-copy, disk-backed)
    mmap = np.memmap(str(TOKEN_IDS_PATH), dtype=np.uint16, mode="r", shape=ids.shape)
    return mmap


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))


if __name__ == "__main__":
    tok = train_bpe_tokenizer()
    digitize_and_save(tok)