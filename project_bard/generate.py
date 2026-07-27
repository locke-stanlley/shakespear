"""
generate.py - Fast generation with KV cache, top-p, repetition penalty
"""
import torch
from config import (
    DEVICE, CHECKPOINT_DIR, GEN_TEMPERATURE, GEN_TOP_K,
    GEN_TOP_P, GEN_REP_PENALTY, GEN_MAX_NEW_TOKENS
)
from model import ShakespeareGPT, ModelConfig
from tokenizer import load_tokenizer


def load_model_for_inference():
    ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def generate(
    prompt: str,
    max_new_tokens: int = GEN_MAX_NEW_TOKENS,
    temperature: float = GEN_TEMPERATURE,
    top_k: int = GEN_TOP_K,
    top_p: float = GEN_TOP_P,
    repetition_penalty: float = GEN_REP_PENALTY,
):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference().to(device)
    tokenizer = load_tokenizer()

    bos = tokenizer.token_to_id("[BOS]")
    ids = [bos] + tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if t0:
        t0.record()

    out_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    if t0:
        t1.record()
        torch.cuda.synchronize()
        elapsed = t0.elapsed_time(t1)
        tokens_per_sec = max_new_tokens / (elapsed / 1000)
        print(f"[Generation] {max_new_tokens} tokens in {elapsed:.1f}ms ({tokens_per_sec:.1f} tok/s)")

    text = tokenizer.decode(out_ids[0].tolist())
    return text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="ROMEO:\n")
    parser.add_argument("--max-tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=GEN_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=GEN_TOP_K)
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P)
    parser.add_argument("--rep-penalty", type=float, default=GEN_REP_PENALTY)
    args = parser.parse_args()

    print(generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
    ))