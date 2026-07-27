"""
chat.py - Interactive continuous prompting interface.
Maintains conversation history and uses KV cache for fast responses.
"""
import torch
import argparse
from typing import List

from config import (
    DEVICE, CHECKPOINT_DIR, GEN_TEMPERATURE, GEN_TOP_K,
    GEN_TOP_P, GEN_REP_PENALTY, BLOCK_SIZE
)
from model import ShakespeareGPT, ModelConfig
from tokenizer import load_tokenizer


def load_model_for_inference():
    """Load the best checkpoint for inference."""
    ckpt_path = CHECKPOINT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run train.py first.")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def format_prompt(history: List[str], new_input: str, system_prompt: str = "") -> str:
    """Format the conversation history for the model."""
    # Since this is a base model trained on raw text, we format it like a script
    # to encourage dialogue-like continuations.
    lines = []
    if system_prompt:
        lines.append(f"[{system_prompt}]\n")
    
    for i, turn in enumerate(history):
        speaker = "USER" if i % 2 == 0 else "ASSISTANT"
        lines.append(f"{speaker}:\n{turn.strip()}")
    
    lines.append("ASSISTANT:\n")
    return "\n".join(lines)


def chat_loop(
    model: ShakespeareGPT,
    tokenizer,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    max_new_tokens: int
):
    """Main interactive chat loop."""
    print("=" * 70)
    print(" 🎭 PROJECT BARD: Interactive Shakespearean Chat 🎭")
    print("=" * 70)
    print("Commands:")
    print("  /clear      - Clear conversation history")
    print("  /settings   - Show current generation settings")
    print("  /quit       - Exit the chat")
    print("=" * 70)
    
    history: List[str] = []
    system_prompt = "A wise and eloquent assistant speaking in Early Modern English."

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
        elif user_input.lower() == "/settings":
            print(f"⚙️ Temp: {temperature} | Top-K: {top_k} | Top-P: {top_p} | Rep Penalty: {rep_penalty}")
            continue

        # Add to history
        history.append(user_input)

        # Format full context
        prompt_text = format_prompt(history, user_input, system_prompt)
        
        # Tokenize
        ids = tokenizer.encode(prompt_text).ids
        
        # Truncate if exceeding context window (leave room for generation)
        max_prompt_len = BLOCK_SIZE - max_new_tokens
        if len(ids) > max_prompt_len:
            print(f"⚠️ Context too long ({len(ids)} tokens). Truncating oldest messages...")
            ids = ids[-max_prompt_len:]

        idx = torch.tensor([ids], dtype=torch.long, device=device)

        print("\n🎭 Bard: ", end="", flush=True)

        # Generate
        with torch.no_grad():
            out_ids = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
            )

        # Decode only the newly generated part
        new_ids = out_ids[0][len(ids):].tolist()
        response_text = tokenizer.decode(new_ids).strip()
        
        print(response_text)
        
        # Add response to history
        history.append(response_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Chat with Project Bard")
    parser.add_argument("--temp", type=float, default=GEN_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=GEN_TOP_K, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P, help="Top-P (nucleus) sampling")
    parser.add_argument("--rep-penalty", type=float, default=GEN_REP_PENALTY, help="Repetition penalty")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per turn")
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model on {device}...")
    
    model, cfg = load_model_for_inference()
    model = model.to(device)
    
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    print("[+] Ready! Start typing to converse.\n")
    
    chat_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        max_new_tokens=args.max_tokens
    )