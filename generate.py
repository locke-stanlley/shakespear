"""
generate.py - Sample text from the trained model.
"""
import torch
from config import DEVICE, CHECKPOINT_DIR, TOKENIZER_DIR
from model import ShakespeareGPT, ModelConfig
from tokenizer import load_tokenizer


def load_model_for_inference():
    ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def generate(prompt: str, max_new_tokens: int = 300, temperature: float = 0.8, top_k: int = 40):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference().to(device)
    tokenizer = load_tokenizer()

    bos = tokenizer.token_to_id("[BOS]")
    ids = [bos] + tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out_ids = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = tokenizer.decode(out_ids[0].tolist())
    return text


if __name__ == "__main__":
    prompt = "ROMEO:\n"
    print(generate(prompt, max_new_tokens=400, temperature=0.85, top_k=50))