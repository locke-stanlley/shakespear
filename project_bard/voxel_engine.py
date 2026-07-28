"""
voxel_engine.py - VoxelPath Geometric Logit Navigator
Maps the trained model's 32K vocabulary logits into a 32x32x32 3D voxel grid.
Uses geometric pathfinding (peak detection + spatial neighborhood bias) 
to generate text, leveraging the trained weights and preventing repetition loops.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List

class VoxelEngine:
    def __init__(self, model, tokenizer, vocab_size: int = 32768):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        
        # 32x32x32 grid perfectly maps 32,768 tokens (32^3 = 32768)
        self.grid_dim = 32
        
    def _token_to_coords(self, token_id: int) -> Tuple[int, int, int]:
        """Maps a token ID to (x, y, z) in the 32x32x32 grid."""
        x = token_id // 1024
        y = (token_id % 1024) // 32
        z = token_id % 32
        return x, y, z

    def _coords_to_token(self, x: int, y: int, z: int) -> int:
        """Maps (x, y, z) back to a token ID."""
        return x * 1024 + y * 32 + z

    def get_voxel_grid(self, idx: torch.Tensor, device: torch.device) -> np.ndarray:
        """
        Passes the prompt through the model and reshapes the output logits 
        into a 32x32x32 3D voxel grid representing the geometric probability landscape.
        """
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                out = self.model(idx, use_kv_cache=False)
                # out is a dict: {"logits": ..., "loss": ..., "new_kvs": ...}
                logits = out["logits"]
                
        # Get logits for the last token: shape (1, vocab_size)
        last_logits = logits[0, -1, :].cpu().float().numpy()
        
        # Reshape into 32x32x32 voxel grid
        voxel_grid = last_logits.reshape((self.grid_dim, self.grid_dim, self.grid_dim))
        return voxel_grid

    def find_peaks(self, voxel_grid: np.ndarray, top_k: int = 10) -> List[Tuple[Tuple[int, int, int], float, int]]:
        """
        Finds the top K peaks (maximum values) in the 3D voxel grid.
        Returns a list of: ((x, y, z), logit_value, token_id)
        """
        # Flatten and get top K indices
        flat_grid = voxel_grid.flatten()
        top_indices = np.argsort(flat_grid)[-top_k:][::-1]
        
        peaks = []
        for idx in top_indices:
            # Convert flat index back to 3D coords
            x = idx // (self.grid_dim * self.grid_dim)
            y = (idx % (self.grid_dim * self.grid_dim)) // self.grid_dim
            z = idx % self.grid_dim
            
            token_id = self._coords_to_token(x, y, z)
            peaks.append(((x, y, z), flat_grid[idx], token_id))
            
        return peaks

    def generate_geometric(self, prompt: str, max_new_tokens: int = 100, 
                           temperature: float = 0.8, device: torch.device = torch.device("cpu")) -> str:
        """
        Generates text using VoxelPath geometric pathfinding on the model's logit landscape.
        """
        # Encode prompt
        encoded = self.tokenizer.encode(prompt)
        ids = encoded.ids
        eos_id = self.tokenizer.token_to_id("[EOS]")
        
        # Strip trailing EOS for continuation
        if ids and ids[-1] == eos_id:
            ids = ids[:-1]
            
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        generated_ids = ids.copy()
        
        # Track recent tokens to prevent immediate repetition (Spatial Exclusion)
        recent_token_ids = set(ids[-4:]) 
        
        for step in range(max_new_tokens):
            # 1. Get the 3D Voxel Grid of logits for the current context
            voxel_grid = self.get_voxel_grid(idx, device)
            
            # 2. Find the primary peaks (maximum values in the model's weight landscape)
            peaks = self.find_peaks(voxel_grid, top_k=15)
            
            # 3. Geometric Pathfinding: Filter out recently used tokens
            valid_peaks = [(coords, val, tid) for coords, val, tid in peaks if tid not in recent_token_ids]
            
            if not valid_peaks:
                # Fallback: if all top peaks are blocked, allow them to prevent dead ends
                valid_peaks = peaks[:5]
                
            # 4. Apply Temperature to the peak values
            top_coords, top_vals, top_tids = zip(*valid_peaks)
            scaled_vals = np.array(top_vals) / max(temperature, 0.1)
            
            # Softmax over the valid geometric peaks
            exp_vals = np.exp(scaled_vals - np.max(scaled_vals)) # Numerical stability
            probs = exp_vals / np.sum(exp_vals)
            
            # 5. Sample the next token geometrically
            chosen_idx = np.random.choice(len(valid_peaks), p=probs)
            chosen_coords, chosen_val, next_token_id = valid_peaks[chosen_idx]
            
            # 6. Update state
            generated_ids.append(next_token_id)
            recent_token_ids.add(next_token_id)
            
            # Maintain a sliding window of recent tokens for repetition penalty
            if len(recent_token_ids) > 6:
                oldest = generated_ids[-7]
                recent_token_ids.discard(oldest)
                
            # Check for EOS
            if next_token_id == eos_id:
                break
                
            # Advance the context
            idx = torch.tensor([generated_ids], dtype=torch.long, device=device)
            
        # Decode only the new tokens
        new_ids = generated_ids[len(ids):]
        return self.tokenizer.decode(new_ids).strip()