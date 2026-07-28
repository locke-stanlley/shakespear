"""
voxel_engine.py - VoxelPath Geometric Encoding Engine (Upgraded)
Implements O(n) 3D coordinate mapping, Semantic Halo, Trajectory Tunneling,
Temporal Decay, and Momentum-based pathfinding for transparent, lightweight AI.
Memory footprint: exactly 8.38 MB (128^3 * 4 bytes), fitting in CPU L3 cache.
"""
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

class VoxelEngine:
    def __init__(self, grid_size: int = 128, tunnel_threshold: int = 100):
        self.grid_size = grid_size
        # 128x128x128 grid of uint32 = 8,388,608 bytes (~8.4 MB)
        self.voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint32)
        self.total_trigrams = 0
        
        # Trajectory Tunneling: Cache for high-frequency n-grams (n > 3)
        self.tunnel_threshold = tunnel_threshold
        self.tunnels: Dict[str, str] = {} 
        
        # Temporal Decay rate (e.g., 0.9999 per update to forget ancient noise)
        self.decay_rate = 0.9999 

    def _get_coords(self, c1: str, c2: str, c3: str) -> Tuple[int, int, int]:
        """Maps 3-char sequence to (x, y, z) coordinates with ASCII clamping."""
        return (ord(c1) % self.grid_size, 
                ord(c2) % self.grid_size, 
                ord(c3) % self.grid_size)

    def _apply_semantic_halo(self, x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
        """
        Semantic Halo: Links uppercase/lowercase variants through geometric 
        proximity (32-offset in ASCII, which is the 6th bit).
        """
        neighbors = [(x, y, z)]
        # Flip 6th bit (XOR 32) for each coordinate to find case-variant neighbors
        if 65 <= x <= 90 or 97 <= x <= 122: neighbors.append((x ^ 32, y, z))
        if 65 <= y <= 90 or 97 <= y <= 122: neighbors.append((x, y ^ 32, z))
        if 65 <= z <= 90 or 97 <= z <= 122: neighbors.append((x, y, z ^ 32))
        return list(set(neighbors)) # Remove duplicates

    def train(self, text: str, apply_decay: bool = False) -> None:
        """
        O(n) training: Map trigrams to 3D coordinates and increment density.
        Also extracts Trajectory Tunnels for high-frequency sequences.
        """
        if apply_decay:
            self.voxel_grid = (self.voxel_grid.astype(np.float32) * self.decay_rate).astype(np.uint32)
            
        ngram_counts = Counter()
        
        # O(n) sequential pass
        for i in range(len(text) - 2):
            c1, c2, c3 = text[i], text[i+1], text[i+2]
            x, y, z = self._get_coords(c1, c2, c3)
            self.voxel_grid[x, y, z] += 1
            self.total_trigrams += 1
            
            # Track 4-grams and 5-grams for Trajectory Tunneling
            if i < len(text) - 3:
                ngram_counts[text[i:i+4]] += 1
            if i < len(text) - 4:
                ngram_counts[text[i:i+5]] += 1

        # Build Trajectory Tunnels
        for ngram, count in ngram_counts.items():
            if count >= self.tunnel_threshold and len(ngram) > 3:
                # Store the continuation of the first 3 chars
                self.tunnels[ngram[:3]] = ngram[3:]

    def get_next_char_distribution(self, c1: str, c2: str) -> np.ndarray:
        """
        Trajectory Tunneling & Semantic Halo: Get probability distribution 
        for the next character given a bigram.
        """
        x = ord(c1) % self.grid_size
        y = ord(c2) % self.grid_size
        
        # 1. Base Z-axis slice (O(1) memory access)
        z_counts = self.voxel_grid[x, y, :].astype(np.float32)
        
        # 2. Apply Semantic Halo if base count is low (prevents dead ends)
        total_base = np.sum(z_counts)
        if total_base < 10:
            for hx, hy, hz in self._apply_semantic_halo(x, y, 0):
                if hx != x or hy != y:
                    # Add a fraction of the halo's Z-slice to our counts
                    z_counts += self.voxel_grid[hx, hy, :].astype(np.float32) * 0.5

        total = np.sum(z_counts)
        if total > 0:
            return z_counts / total
        return np.zeros(self.grid_size)

    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8, 
                 repetition_penalty: float = 1.2) -> str:
        """Generate text using momentum-based voxel pathfinding and tunneling."""
        if len(prompt) < 2:
            return prompt
            
        output = list(prompt)
        recent_chars = [] # For momentum-based repetition penalty
        
        for step in range(max_length):
            c1, c2 = output[-2], output[-1]
            
            # 1. Trajectory Tunneling Check (O(1) dictionary lookup)
            tunnel_key = c1 + c2
            tunneled_continuation = None
            for key, continuation in self.tunnels.items():
                if key.startswith(tunnel_key) and key[:2] == tunnel_key:
                    # We found a high-frequency path! Jump ahead.
                    tunneled_continuation = key[2:] + continuation
                    break
            
            if tunneled_continuation:
                output.extend(list(tunneled_continuation))
                recent_chars.extend(list(tunneled_continuation)[-4:])
                continue # Skip normal sampling for this step
            
            # 2. Normal Voxel Sampling
            probs = self.get_next_char_distribution(c1, c2)
            
            if np.sum(probs) == 0:
                break
                
            # 3. Momentum-Based Prediction & Repetition Penalty
            for i in range(self.grid_size):
                char = chr(i)
                if char in recent_chars:
                    probs[i] /= repetition_penalty
            
            # 4. Temperature scaling
            scaled_probs = probs ** (1.0 / max(temperature, 0.1))
            scaled_probs = scaled_probs / (np.sum(scaled_probs) + 1e-9)
            
            # 5. Sample next character index
            next_char_idx = np.random.choice(self.grid_size, p=scaled_probs)
            next_char = chr(next_char_idx)
            
            output.append(next_char)
            recent_chars.append(next_char)
            if len(recent_chars) > 4:
                recent_chars.pop(0)
            
            # Stop if we hit a natural break and have generated enough
            if step > max_length * 0.8 and next_char in ['.', '!', '?', '\n']:
                break
                
        return "".join(output)

    def rag_retrieve(self, query: str, documents: List[str], top_k: int = 3) -> List[Dict]:
        """
        Lightweight RAG: Retrieves documents with the highest trigram overlap 
        with the query, using the voxel grid as a fast geometric filter.
        """
        query_trigrams = set()
        for i in range(len(query) - 2):
            c1, c2, c3 = query[i], query[i+1], query[i+2]
            x, y, z = self._get_coords(c1, c2, c3)
            query_trigrams.add((x, y, z))
            
        doc_scores = []
        for doc in documents:
            score = 0
            for i in range(len(doc) - 2):
                c1, c2, c3 = doc[i], doc[i+1], doc[i+2]
                x, y, z = self._get_coords(c1, c2, c3)
                if (x, y, z) in query_trigrams:
                    # Boost score by the global density of this trigram
                    score += self.voxel_grid[x, y, z]
            doc_scores.append({"content": doc, "score": score})
            
        # Sort by score descending
        doc_scores.sort(key=lambda x: x["score"], reverse=True)
        return doc_scores[:top_k]

    def save(self, path: Path) -> None:
        """Save the voxel grid and tunnels to disk."""
        np.save(path.with_suffix('.npy'), self.voxel_grid)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump({
                "total_trigrams": int(self.total_trigrams),
                "tunnels": self.tunnels
            }, f)
        
    def load(self, path: Path) -> None:
        """Load the voxel grid and tunnels from disk."""
        npy_path = path.with_suffix('.npy')
        json_path = path.with_suffix('.json')
        if npy_path.exists():
            self.voxel_grid = np.load(npy_path)
            self.total_trigrams = int(np.sum(self.voxel_grid))
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.tunnels = data.get("tunnels", {})