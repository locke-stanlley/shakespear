"""
evaluate.py - Phase 6: Evaluation & Alignment
  - Final test-set perplexity
  - Supervised Fine-Tuning (SFT) on instruction data
  - Direct Preference Optimization (DPO) for alignment
"""
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from config import (
    DEVICE, DTYPE, CHECKPOINT_DIR, VOCAB_SIZE, BLOCK_SIZE,
    SFT_DATA_PATH, DPO_DATA_PATH, SFT_EPOCHS, SFT_LR, DPO_LR, DPO_BETA,
    TOKENIZER_DIR
)
from model import ShakespeareGPT, ModelConfig
from dataset import get_dataloader
from tokenizer import load_tokenizer


def load_model(checkpoint: str = "best.pt") -> ShakespeareGPT:
    ckpt_path = CHECKPOINT_DIR / checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# -----------------------------
# Final test-set evaluation
# -----------------------------
@torch.no_grad()
def final_test_evaluation():
    print("=" * 60)
    print("[PHASE 6] Final Test Evaluation")
    print("=" * 60)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16

    model = load_model().to(device)
    dl = get_dataloader("test", batch_size=16, shuffle=False)
    losses = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    print(f"[+] Test loss: {mean_loss:.4f}")
    print(f"[+] Test perplexity: {math.exp(mean_loss):.2f}")


# -----------------------------
# SFT Dataset
# -----------------------------
class SFTDataset(Dataset):
    """Each item is a JSONL line: {"prompt": "...", "completion": "..."}"""

    def __init__(self, path: Path, tokenizer, block_size: int = BLOCK_SIZE):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        bos = tokenizer.token_to_id("[BOS]")
        eos = tokenizer.token_to_id("[EOS]")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = f"[BOS]User: {obj['prompt'].strip()}\nAssistant: {obj['completion'].strip()}[EOS]"
                ids = tokenizer.encode(text).ids
                if len(ids) > block_size + 1:
                    ids = ids[: block_size + 1]
                self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        pad_len = self.block_size + 1 - len(ids)
        ids = ids + [0] * pad_len  # 0 = [PAD]
        t = torch.tensor(ids, dtype=torch.long)
        x = t[:-1]
        y = t[1:].clone()
        # Mask loss on padding
        y[x == 0] = -100
        return x, y


def run_sft():
    print("=" * 60)
    print("[PHASE 6] Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    if not SFT_DATA_PATH.exists():
        print(f"[!] SFT data not found at {SFT_DATA_PATH}; generating synthetic sample...")
        create_synthetic_sft()

    tokenizer = load_tokenizer()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = load_model().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.01)

    ds = SFTDataset(SFT_DATA_PATH, tokenizer)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    for epoch in range(SFT_EPOCHS):
        total, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        print(f"[SFT epoch {epoch+1}/{SFT_EPOCHS}] loss={total/max(1,n):.4f}")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "sft_model.pt")
    print("[+] SFT model saved.")


def create_synthetic_sft():
    """Create a tiny synthetic SFT dataset if none exists."""
    samples = [
        {"prompt": "Write a sonnet about the sea.",
         "completion": "Upon the waves where silver moonbeams play,\nMy heart doth sail upon the endless sea."},
        {"prompt": "Describe a king in Shakespearean style.",
         "completion": "A sovereign clad in majesty and grace,\nHis crown doth gleam like Phoebus' golden face."},
        {"prompt": "What is love?",
         "completion": "Love is a spirit of immortal flame,\nThat burns the soul yet leaves no mark of shame."},
    ]
    with SFT_DATA_PATH.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# -----------------------------
# DPO (using HuggingFace trl)
# -----------------------------
def create_synthetic_dpo():
    """Create a tiny synthetic DPO dataset if none exists."""
    samples = [
        {
            "prompt": "Write a haiku about winter.",
            "chosen": "Snow falls soft and still / Silent world in silver cloak / Peace upon the bough",
            "rejected": "Winter cold snow fall me very cold yes freeze ice.",
        },
        {
            "prompt": "Describe a storm.",
            "chosen": "The heavens roar with thunder's mighty voice, as lightning splits the darkened sky above.",
            "rejected": "Storm big loud rain fall down boom boom yes.",
        },
    ]
    with DPO_DATA_PATH.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def run_dpo():
    print("=" * 60)
    print("[PHASE 6] Direct Preference Optimization (DPO)")
    print("=" * 60)

    try:
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset as HFDataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[!] trl/transformers not installed. Skipping DPO. Install with: pip install trl transformers datasets")
        return

    if not DPO_DATA_PATH.exists():
        print(f"[!] DPO data not found; generating synthetic sample...")
        create_synthetic_dpo()

    # For DPO we wrap our model in a HF-compatible CausalLM.
    # For a side project we use a tiny random HF model as a stand-in reference.
    # In production, you'd convert ShakespeareGPT to HF format first.
    print("[*] Loading reference model for DPO (stand-in HF model)...")
    model_name = "sshleifer/tiny-gpt2"  # tiny stand-in; swap for your converted model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load DPO data
    data = []
    with DPO_DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                "prompt": obj["prompt"],
                "chosen": obj["chosen"],
                "rejected": obj["rejected"],
            })
    hf_ds = HFDataset.from_list(data)

    training_args = DPOConfig(
        output_dir=str(CHECKPOINT_DIR / "dpo"),
        learning_rate=DPO_LR,
        beta=DPO_BETA,
        per_device_train_batch_size=2,
        max_steps=20,
        logging_steps=5,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=hf_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(CHECKPOINT_DIR / "dpo_final"))
    print("[+] DPO training complete.")


if __name__ == "__main__":
    final_test_evaluation()
    run_sft()
    run_dpo()