"""
chat.py - Production-Grade Interactive Chat Interface
Features:
  - Real-time token streaming (typewriter effect)
  - Robust few-shot prompt steering to prevent mode collapse
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


def build_prompt(history: List[Dict[str, str]], new_input: str) -> str:
    """
    Production-grade prompt engineering for a BASE completion model.
    Uses few-shot steering to force Q&A format and permanently prevent play-script hallucination.
    """
    system_prompt = """You are an expert literary assistant with deep knowledge of classic literature, including Shakespeare, Sherlock Holmes, Jane Austen, Mary Shelley, Lewis Carroll, and Miguel de Cervantes. 

Provide a concise, accurate, and direct answer to the following query. Do not write a play script or use character names unless explicitly asked.

Examples:
Query: Who is Sherlock Holmes's arch-nemesis?
Response: Sherlock Holmes's arch-nemesis is the criminal mastermind Professor Moriarty.

Query: How many Bennet sisters are there in Pride and Prejudice?
Response: There are five Bennet sisters: Jane, Elizabeth, Mary, Kitty, and Lydia.
"""
    
    lines = [system_prompt]
    # Append recent history (keep last 3 turns to save context window)
    for turn in history[-3:]:
        lines.append(f"Query: {turn['user']}\nResponse: {turn['assistant']}")
    
    lines.append(f"Query: {new_input}\nResponse:")
    return "\n".join(lines)


def chat_loop(model: ShakespeareGPT, tokenizer, device: torch.device, args):
    """Main interactive chat loop with real-time streaming."""
    print("=" * 70)
    print(" 🎭 PROJECT BARD: Production-Grade Chat Interface 🎭")
    print("=" * 70)
    print("Commands:")
    print("  /clear      - Clear conversation history")
    print("  /stats      - Show model statistics")
    print("  /quit       - Exit the chat")
    print("=" * 70)
    
    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Exiting chat. Farewell!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["/quit", "/exit", "q"]:
            print("👋 Exiting chat. Farewell!")
            break
        elif user_input.lower() == "/clear":
            history = []
            print("🧹 Conversation history cleared.")
            continue
        elif user_input.lower() == "/stats":
            print(f"⚙️ Model Parameters: {count_parameters(model):,}")
            print(f"⚙️ Vocab Size: {model.cfg.vocab_size}")
            print(f"⚙️ Context Window: {model.cfg.block_size} tokens")
            print(f"⚙️ Generation Settings: Temp={args.temp}, Top-K={args.top_k}, Top-P={args.top_p}, Rep-Penalty={args.rep_penalty}")
            continue

        # Format full context
        prompt_text = build_prompt(history, user_input)
        
        # Tokenize
        ids = tokenizer.encode(prompt_text).ids
        
        # Hard truncate to leave room for generation
        max_prompt_len = BLOCK_SIZE - args.max_tokens
        if len(ids) > max_prompt_len:
            print(f"⚠️ Context too long ({len(ids)} tokens). Truncating oldest messages...")
            ids = ids[-max_prompt_len:]

        idx = torch.tensor([ids], dtype=torch.long, device=device)

        print("\n🎭 Bard: ", end="", flush=True)

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
        print(f"\n\n[⚡ Generated {token_count} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)]")
        
        # Clean up any trailing artifacts or prompt leakage
        response_text = response_text.split("Query:")[0].split("[_")[0].split("SCENE")[0].strip()
        
        # Add to history
        history.append({"user": user_input, "assistant": response_text})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Chat with Project Bard")
    parser.add_argument("--temp", type=float, default=GEN_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=GEN_TOP_K, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P, help="Top-P (nucleus) sampling")
    parser.add_argument("--rep-penalty", type=float, default=GEN_REP_PENALTY, help="Repetition penalty")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max new tokens per turn")
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading healthy base model on {device}...")
    
    model, cfg = load_model_for_inference()
    model = model.to(device)
    
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("[+] Ready! Start typing to converse.\n")
    
    chat_loop(model, tokenizer, device, args)