"""
voxel_chat.py - Standalone VoxelPath Chat Interface
Demonstrates the ultra-lightweight, O(n) VoxelEngine running entirely on CPU.
No GPU required. Memory footprint: ~8.4 MB.
"""
import argparse
import time
from pathlib import Path
from typing import List

# Fallback paths if config is not imported
try:
    from config import CHECKPOINT_DIR, CLEAN_TEXT_PATH
except ImportError:
    CHECKPOINT_DIR = Path("checkpoints")
    CLEAN_TEXT_PATH = Path("data/clean/shakespeare_clean.txt")

from voxel_engine import VoxelEngine


def load_voxel_system():
    """Load or train the Voxel Engine."""
    voxel_path = CHECKPOINT_DIR / "voxel_model"
    engine = VoxelEngine(grid_size=128, tunnel_threshold=50)
    
    if voxel_path.with_suffix('.npy').exists():
        print("[*] Loading pre-trained Voxel Grid from disk...")
        engine.load(voxel_path)
        print(f"[+] Loaded. Total trigrams mapped: {engine.total_trigrams:,}")
    else:
        print("[*] Training Voxel Grid from corpus (O(n) complexity)...")
        if not CLEAN_TEXT_PATH.exists():
            raise FileNotFoundError(f"Corpus not found at {CLEAN_TEXT_PATH}. Run data_pipeline.py first.")
        
        text = CLEAN_TEXT_PATH.read_text(encoding="utf-8")
        engine.train(text)
        
        # Save for future use
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        engine.save(voxel_path)
        print(f"[+] Trained and saved. Total trigrams: {engine.total_trigrams:,}")
        print(f"[+] Memory footprint: ~{engine.voxel_grid.nbytes / 1024 / 1024:.2f} MB")
        
    return engine


def voxel_chat_loop(engine: VoxelEngine, args):
    """Main interactive chat loop for the Voxel Engine."""
    print("=" * 70)
    print(" VOXEL LM: Standalone Geometric Chat Interface")
    print("=" * 70)
    print("Commands:")
    print("  /clear       - Clear conversation history")
    print("  /temp <val>  - Set temperature (e.g., /temp 0.8)")
    print("  /quit        - Exit the interface")
    print("=" * 70)
    print("Note: This runs entirely on CPU using ~8.4 MB of RAM.")
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

        # Build context (Voxel engine benefits from recent context for momentum)
        if history:
            # Keep last 2 turns to maintain momentum without blowing up memory
            context = "\n".join(history[-2:]) + "\n" + user_input
        else:
            context = user_input

        print("\nVoxel: ", end="", flush=True)
        
        start_time = time.time()
        
        # Generate using Voxel Engine
        response = engine.generate(
            prompt=context,
            max_length=args.max_tokens,
            temperature=args.temp,
            repetition_penalty=args.rep_penalty
        )
        
        # Extract only the newly generated part
        new_text = response[len(context):]
        print(new_text, end="", flush=True)
        
        elapsed = time.time() - start_time
        chars_per_sec = len(new_text) / elapsed if elapsed > 0 else 0
        
        print(f"\n\n[Generated {len(new_text)} chars in {elapsed:.3f}s ({chars_per_sec:.1f} chars/s)]")
        
        history.append(user_input + new_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone VoxelPath Chat")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max chars to generate")
    parser.add_argument("--rep-penalty", type=float, default=1.2, help="Repetition penalty")
    args = parser.parse_args()

    print("[*] Initializing VoxelPath Engine...")
    engine = load_voxel_system()
    print("[+] Ready! Start typing your prompts.\n")
    
    voxel_chat_loop(engine, args)