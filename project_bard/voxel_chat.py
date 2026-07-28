"""
voxel_chat.py - Standalone VoxelPath Geometric Chat Interface
Uses the trained Transformer's weights, mapped into a 32x32x32 3D voxel grid,
to perform geometric pathfinding generation.
"""
import argparse
import time
import torch
from pathlib import Path
from typing import List

try:
    from config import DEVICE, CHECKPOINT_DIR
except ImportError:
    DEVICE = "cuda"
    CHECKPOINT_DIR = Path("checkpoints")

from model import ShakespeareGPT, ModelConfig
from tokenizer import load_tokenizer
from voxel_engine import VoxelEngine


def load_voxel_system():
    """Load the trained model and initialize the Voxel Engine."""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    
    ckpt_path = CHECKPOINT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run train.py first.")
    
    print("[*] Loading trained model...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(dtype=torch.float16).to(device).eval()
    
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    print("[*] Initializing VoxelPath Geometric Engine...")
    print(f"    Vocab size: {cfg.vocab_size} -> Mapped to 32x32x32 3D grid")
    print(f"    Grid memory footprint: ~{(32*32*32*4) / 1024:.1f} KB (L1/L2 Cache friendly)")
    
    engine = VoxelEngine(model, tokenizer, vocab_size=cfg.vocab_size)
    return engine, device


def voxel_chat_loop(engine: VoxelEngine, device: torch.device, args):
    """Main interactive chat loop for the Voxel Engine."""
    print("=" * 70)
    print(" VOXEL LM: Geometric Logit Navigator")
    print("=" * 70)
    print("Commands:")
    print("  /clear       - Clear conversation history")
    print("  /temp <val>  - Set temperature (e.g., /temp 0.8)")
    print("  /quit        - Exit the interface")
    print("=" * 70)
    print("Note: This navigates the trained model's 3D weight landscape.")
    print("Tip: Provide a starting phrase (e.g., 'ROMEO: \\n')")
    print("=" * 70)
    
    history: List[str] = []

    while True:
        try:
            user_input = input("\nPrompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting Voxel Chat.")
            break

        if not user_input:
            continue

        lower_input = user_input.lower()
        if lower_input in ["/quit", "/exit", "q"]:
            print("Exiting Voxel Chat.")
            break
        elif lower_input == "/clear":
            history = []
            print("History cleared.")
            continue
        elif lower_input.startswith("/temp "):
            try:
                args.temp = float(user_input.split(" ", 1)[1].strip())
                print(f"Temperature set to {args.temp}")
            except ValueError:
                print("Invalid temperature value.")
            continue

        # Build context
        if history:
            context = "\n".join(history[-2:]) + "\n" + user_input
        else:
            context = user_input

        print("\nVoxel: ", end="", flush=True)
        
        start_time = time.time()
        
        # Generate using Voxel Geometric Pathfinding
        response = engine.generate_geometric(
            prompt=context,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            device=device
        )
        
        print(response, end="", flush=True)
        
        elapsed = time.time() - start_time
        tokens_generated = len(response.split()) # Approximate token count
        speed = tokens_generated / elapsed if elapsed > 0 else 0
        
        print(f"\n\n[Generated ~{tokens_generated} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)]")
        
        history.append(user_input + " " + response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone VoxelPath Chat")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens to generate")
    args = parser.parse_args()

    print("[*] Initializing VoxelPath Engine...")
    try:
        engine, device = load_voxel_system()
        print("[+] Ready! Start typing your prompts.\n")
        voxel_chat_loop(engine, device, args)
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")