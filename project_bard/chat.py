"""
chat.py - Production-Grade Contextual Inference Interface
Features:
  - Pure text continuation (no artificial Q&A formatting)
  - Real-time token streaming (typewriter effect)
  - Sliding window context management
  - Interactive CLI commands (/clear, /stats, /quit)
"""
import torch
import argparse
import time
from typing import List, Dict

from config import (
    DEVICE, CHECKPOINT_DIR, GEN_TEMPERATURE, GEN_TOP_K,
    GEN_TOP_P, GEN_REP_PENALTY, BLOCK_SIZE
)
from model import ShakespeareGPT, ModelConfig, count_parameters
from tokenizer import load_tokenizer


def load_model_for_inference():
    """Load the healthy base model for inference."""
    ckpt_path = CHECKPOINT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run train.py first.")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def build_prompt(history: List[str], new_input: str) -> str:
    """
    Production-grade prompt engineering for pure contextual continuation.
    Appends new input to the history to maintain narrative context, 
    without forcing artificial Q&A formats.
    """
    if not history:
        return new_input
    
    # Join history with new input to maintain a continuous text block
    return "\n".join(history) + "\n" + new_input


def chat_loop(model: ShakespeareGPT, tokenizer, device: torch.device, args):
    """Main interactive inference loop with real-time streaming."""
    print("=" * 70)
    print(" PROJECT BARD: Production-Grade Contextual Inference Interface")
    print("=" * 70)
    print("Commands:")
    print("  /clear      - Clear conversation history")
    print("  /stats      - Show model statistics")
    print("  /quit       - Exit the interface")
    print("=" * 70)
    print("Tip: Provide a starting phrase, and the model will continue the text.")
    print("Example: 'It was a dark and stormy night, and the wind howled through'")
    print("=" * 70)
    
    history: List[str] = []

    while True:
        try:
            user_input = input("\nPrompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting interface.")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["/quit", "/exit", "q"]:
            print("Exiting interface.")
            break
        elif user_input.lower() == "/clear":
            history = []
            print("History cleared.")
            continue
        elif user_input.lower() == "/stats":
            print(f"Model Parameters: {count_parameters(model):,}")
            print(f"Vocab Size: {model.cfg.vocab_size}")
            print(f"Context Window: {model.cfg.block_size} tokens")
            print(f"Generation Settings: Temp={args.temp}, Top-K={args.top_k}, Top-P={args.top_p}, Rep-Penalty={args.rep_penalty}")
            continue

        # Format full context for pure continuation
        prompt_text = build_prompt(history, user_input)
        
        # Tokenize
        ids = tokenizer.encode(prompt_text).ids
        
        # Hard truncate to leave room for generation
        max_prompt_len = BLOCK_SIZE - args.max_tokens
        if len(ids) > max_prompt_len:
            print(f"Warning: Context too long ({len(ids)} tokens). Truncating oldest messages...")
            ids = ids[-max_prompt_len:]

        idx = torch.tensor([ids], dtype=torch.long, device=device)

        print("\nContinuation: ", end="", flush=True)

        # Stream generation
        start_time = time.time()
        token_count = 0
        response_text = ""
        
        for token in model.generate_stream(
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
        ):
            # Decode single token
            token_str = tokenizer.decode([token])
            print(token_str, end="", flush=True)
            response_text += token_str
            token_count += 1
            
            # Stop if EOS
            if token == 2:
                break

        elapsed = time.time() - start_time
        speed = token_count / elapsed if elapsed > 0 else 0
        print(f"\n\n[Generated {token_count} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)]")
        
        # Clean up any trailing artifacts
        response_text = response_text.split("[_")[0].split("SCENE")[0].strip()
        
        # Add to history for continuous context
        history.append(user_input + response_text)
        
        # Keep history manageable (last 5 turns)
        if len(history) > 5:
            history = history[-5:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contextual Inference with Project Bard")
    parser.add_argument("--temp", type=float, default=GEN_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=GEN_TOP_K, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P, help="Top-P (nucleus) sampling")
    parser.add_argument("--rep-penalty", type=float, default=GEN_REP_PENALTY, help="Repetition penalty")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max new tokens per turn")
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading healthy base model on {device}...")
    
    model, cfg = load_model_for_inference()
    model = model.to(device)
    
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("[+] Ready! Start typing your prompts.\n")
    
    chat_loop(model, tokenizer, device, args)