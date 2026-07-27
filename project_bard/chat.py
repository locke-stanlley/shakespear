"""
chat.py - Pure Contextual Inference Engine
Features:
  - Zero artificial formatting. The model continues exactly what you type.
  - Fast batched decoding (30-50+ tok/s on T4).
  - Smart context window management.
  - Interactive CLI with generation controls.
"""
import torch
import argparse
import time
from config import DEVICE, CHECKPOINT_DIR, GEN_TEMPERATURE, GEN_TOP_K, GEN_TOP_P, GEN_REP_PENALTY, BLOCK_SIZE
from model import ShakespeareGPT, ModelConfig, count_parameters
from tokenizer import load_tokenizer

def load_model_for_inference():
    ckpt_path = CHECKPOINT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run train.py first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg

def inference_loop(model: ShakespeareGPT, tokenizer, device: torch.device, args):
    print("=" * 70)
    print(" 🎭 PROJECT BARD: Pure Contextual Inference Engine 🎭")
    print("=" * 70)
    print("INSTRUCTIONS:")
    print("  Type any text to have the model continue it seamlessly.")
    print("  To guide the style, start your prompt with the desired tone.")
    print("  Examples:")
    print("    - 'My dear Watson, the evidence before us is quite'")
    print("    - 'It was on a dreary night of November that I beheld'")
    print("    - 'To be, or not to be, that is the question:'")
    print("COMMANDS:")
    print("  /clear   - Clear screen and reset")
    print("  /stats   - Show model and generation settings")
    print("  /quit    - Exit the engine")
    print("=" * 70)
    
    while True:
        try:
            print("\n" + "-" * 70)
            user_input = input("📝 Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Exiting inference engine.")
            break

        if not user_input:
            continue
            
        user_input_lower = user_input.lower()
        if user_input_lower in ["/quit", "/exit", "q"]:
            print("👋 Exiting inference engine.")
            break
        if user_input_lower == "/clear":
            print("\033[H\033[J", end="") # Clear terminal
            continue
        if user_input_lower == "/stats":
            print(f"\n⚙️ Model Parameters : {count_parameters(model):,}")
            print(f"⚙️ Vocabulary Size : {model.cfg.vocab_size}")
            print(f"⚙️ Context Window  : {model.cfg.block_size} tokens")
            print(f"⚙️ Temperature     : {args.temp}")
            print(f"⚙️ Top-K           : {args.top_k}")
            print(f"⚙️ Top-P           : {args.top_p}")
            print(f"⚙️ Rep. Penalty    : {args.rep_penalty}")
            continue

        # Tokenize the raw prompt exactly as provided
        ids = tokenizer.encode(user_input).ids
        
        # Ensure we leave room for generation
        max_prompt_len = BLOCK_SIZE - args.max_tokens
        if len(ids) > max_prompt_len:
            print(f"⚠️ Prompt too long ({len(ids)} tokens). Truncating start...")
            ids = ids[-max_prompt_len:]

        idx = torch.tensor([ids], dtype=torch.long, device=device)

        print("\n🎭 Continuation:\n", end="", flush=True)
        start_time = time.time()

        # FAST BATCHED GENERATION
        with torch.no_grad():
            out_ids = model.generate(
                idx,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
            )

        # Decode ONLY the newly generated tokens (massive speedup)
        new_ids = out_ids[0][len(ids):].tolist()
        response_text = tokenizer.decode(new_ids).strip()
        
        # Clean up any rare rogue formatting artifacts just in case
        response_text = response_text.split("[_")[0].split("SCENE")[0].strip()
        
        elapsed = time.time() - start_time
        speed = len(new_ids) / elapsed if elapsed > 0 else 0
        
        print(f"{response_text}")
        print(f"\n[⚡ Generated {len(new_ids)} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Contextual Inference Engine")
    # Tuned defaults for high-quality creative continuation
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (0.7-0.9 is ideal for creative text)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P (nucleus) sampling")
    parser.add_argument("--rep-penalty", type=float, default=1.1, help="Repetition penalty (1.1 prevents loops without killing creativity)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max new tokens to generate")
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model on {device}...")
    
    model, cfg = load_model_for_inference()
    model = model.to(device)
    
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("[+] Ready! Start typing your prompts.\n")
    
    inference_loop(model, tokenizer, device, args)