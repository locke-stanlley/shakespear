"""
train.py - Phase 5: Pre-Training
AdamW + Cosine LR with warmup + gradient clipping + mixed precision (bfloat16).
Tracks validation perplexity and checkpoints the model.
"""
import math
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from config import (
    DEVICE, DTYPE, LEARNING_RATE, WEIGHT_DECAY, BETA1, BETA2,
    GRAD_CLIP, WARMUP_STEPS, MAX_STEPS, MIN_LR_RATIO,
    LOG_INTERVAL, EVAL_INTERVAL, SAVE_INTERVAL, CHECKPOINT_DIR, SEED,
    BATCH_SIZE, VOCAB_SIZE
)
from model import ShakespeareGPT, ModelConfig, count_parameters
from dataset import get_dataloader, split_data


# -----------------------------
# Learning Rate Schedule
# -----------------------------
def get_lr(step: int) -> float:
    """Cosine decay with linear warmup."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return LEARNING_RATE * MIN_LR_RATIO
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine)


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model: ShakespeareGPT, device: torch.device, dtype: torch.dtype) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        dl = get_dataloader(split, batch_size=BATCH_SIZE, shuffle=False)
        losses = []
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            with autocast(device_type=device.type, dtype=dtype):
                _, loss = model(x, y)
            losses.append(loss.item())
        mean_loss = sum(losses) / max(1, len(losses))
        out[f"{split}_loss"] = mean_loss
        out[f"{split}_perplexity"] = math.exp(mean_loss)
    model.train()
    return out


# -----------------------------
# Main training loop
# -----------------------------
def train():
    print("=" * 60)
    print("[PHASE 5] Pre-Training")
    print("=" * 60)

    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
    print(f"[*] Device: {device}, dtype: {dtype}")

    # Ensure splits exist
    split_data()

    # Model
    cfg = ModelConfig(vocab_size=VOCAB_SIZE)
    model = ShakespeareGPT(cfg).to(device)
    print(f"[+] Model parameters: {count_parameters(model):,}")

    # Optimizer (AdamW with decoupled weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    # Optional GradScaler for float16 (not needed for bfloat16)
    scaler = GradScaler(enabled=(dtype == torch.float16))

    train_loader = get_dataloader("train", batch_size=BATCH_SIZE, shuffle=True)
    data_iter = iter(train_loader)

    model.train()
    step = 0
    t0 = time.time()
    best_val_loss = float("inf")

    while step < MAX_STEPS:
        # Refresh iterator if exhausted
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Update LR (cosine schedule)
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward (mixed precision)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=dtype):
            _, loss = model(x, y)

        scaler.scale(loss).backward()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % LOG_INTERVAL == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"[step {step:04d}] loss={loss.item():.4f} | lr={lr:.2e} | "
                f"iter_time={dt * 1000 / LOG_INTERVAL:.1f}ms"
            )

        # Evaluation
        if step > 0 and step % EVAL_INTERVAL == 0:
            metrics = evaluate(model, device, dtype)
            print(
                f"[eval step {step}] "
                f"train_loss={metrics['train_loss']:.4f} train_ppl={metrics['train_perplexity']:.2f} | "
                f"val_loss={metrics['val_loss']:.4f} val_ppl={metrics['val_perplexity']:.2f}"
            )
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                ckpt_path = CHECKPOINT_DIR / "best.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": cfg,
                }, ckpt_path)
                print(f"[+] New best model saved: {ckpt_path}")

        # Periodic checkpoint
        if step > 0 and step % SAVE_INTERVAL == 0:
            ckpt_path = CHECKPOINT_DIR / f"step_{step}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

        step += 1

    print("[+] Training complete.")
    return model


if __name__ == "__main__":
    train()