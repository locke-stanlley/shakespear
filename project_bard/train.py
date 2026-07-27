"""
train.py - T4-optimized training with:
  - float16 mixed precision (T4-native)
  - Gradient accumulation (effective batch = 64)
  - WandB logging
  - Cosine LR with warmup
"""
import math
import time
import torch
from torch.amp import autocast, GradScaler

from config import (
    DEVICE, DTYPE, LEARNING_RATE, WEIGHT_DECAY, BETA1, BETA2,
    GRAD_CLIP, WARMUP_STEPS, MAX_STEPS, MIN_LR_RATIO,
    LOG_INTERVAL, EVAL_INTERVAL, SAVE_INTERVAL, CHECKPOINT_DIR, SEED,
    BATCH_SIZE, GRAD_ACCUM_STEPS, VOCAB_SIZE, USE_WANDB,
    WANDB_PROJECT, WANDB_ENTITY
)
from model import ShakespeareGPT, ModelConfig, count_parameters
from dataset import get_dataloader, split_data


def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return LEARNING_RATE * MIN_LR_RATIO
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine)


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
                _, loss, _ = model(x, y)
            losses.append(loss.item())
        mean_loss = sum(losses) / max(1, len(losses))
        out[f"{split}_loss"] = mean_loss
        out[f"{split}_perplexity"] = math.exp(min(mean_loss, 20))  # cap for safety
    model.train()
    return out


def train():
    print("=" * 60)
    print("[PHASE 5] Pre-Training (T4-optimized)")
    print("=" * 60)

    # Initialize WandB
    wandb_run = None
    if USE_WANDB:
        try:
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                config={k: v for k, v in locals().items() if isinstance(v, (int, float, str, bool))},
            )
        except Exception as e:
            print(f"[!] WandB init failed: {e}. Continuing without WandB.")

    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if DTYPE == "float16" else torch.bfloat16
    print(f"[*] Device: {device}, dtype: {dtype}")
    print(f"[*] Effective batch size: {BATCH_SIZE} * {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    if device.type == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    split_data()

    cfg = ModelConfig(vocab_size=VOCAB_SIZE)
    model = ShakespeareGPT(cfg).to(device)
    print(f"[+] Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
        fused=True,  # Faster on modern PyTorch
    )

    scaler = GradScaler(enabled=(dtype == torch.float16))

    train_loader = get_dataloader("train", batch_size=BATCH_SIZE, shuffle=True)
    data_iter = iter(train_loader)

    model.train()
    step = 0
    micro_step = 0
    t0 = time.time()
    best_val_loss = float("inf")
    accumulated_loss = 0.0

    while step < MAX_STEPS:
        # Update LR per optimizer step (not micro-step)
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        for accum_step in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Scale loss by accumulation steps
            is_last_accum = (accum_step == GRAD_ACCUM_STEPS - 1)

            with autocast(device_type=device.type, dtype=dtype):
                _, loss, _ = model(x, y)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            if is_last_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % LOG_INTERVAL == 0:
            dt = time.time() - t0
            t0 = time.time()
            avg_loss = accumulated_loss / GRAD_ACCUM_STEPS
            accumulated_loss = 0.0
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            log_data = {
                "step": step,
                "loss": avg_loss,
                "lr": lr,
                "iter_time_ms": dt * 1000 / LOG_INTERVAL,
                "gpu_mem_gb": mem_gb,
            }
            print(
                f"[step {step:04d}] loss={avg_loss:.4f} | lr={lr:.2e} | "
                f"iter_time={dt * 1000 / LOG_INTERVAL:.1f}ms | "
                f"gpu_mem={mem_gb:.2f}GB"
            )
            if wandb_run:
                wandb_run.log(log_data, step=step)

        # Evaluation
        if step > 0 and step % EVAL_INTERVAL == 0:
            metrics = evaluate(model, device, dtype)
            print(
                f"[eval step {step}] "
                f"train_loss={metrics['train_loss']:.4f} train_ppl={metrics['train_perplexity']:.2f} | "
                f"val_loss={metrics['val_loss']:.4f} val_ppl={metrics['val_perplexity']:.2f}"
            )
            if wandb_run:
                wandb_run.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

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

        if step > 0 and step % SAVE_INTERVAL == 0:
            ckpt_path = CHECKPOINT_DIR / f"step_{step}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

        step += 1

    if wandb_run:
        wandb_run.finish()
    print("[+] Training complete.")
    return model


if __name__ == "__main__":
    train()