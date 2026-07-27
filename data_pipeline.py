"""
data_pipeline.py - Phase 1: Data Collection & Curation
Implements: download, format standardization, heuristic filtering,
            exact dedup (SHA-1), fuzzy dedup (MinHash LSH), PII scrubbing.
"""
import re
import hashlib
import urllib.request
from pathlib import Path
from typing import Iterator, List

from datasketch import MinHash, MinHashLSH
from config import (
    SHAKESPEARE_URL, RAW_TEXT_PATH, CLEAN_TEXT_PATH, RAW_DIR
)

# -----------------------------
# 1. Download & Format Standardization
# -----------------------------
def download_shakespeare() -> Path:
    """Download the Complete Works of Shakespeare (Project Gutenberg)."""
    if RAW_TEXT_PATH.exists():
        print(f"[+] Raw text already present: {RAW_TEXT_PATH}")
        return RAW_TEXT_PATH
    print(f"[*] Downloading Shakespeare from {SHAKESPEARE_URL}")
    urllib.request.urlretrieve(SHAKESPEARE_URL, RAW_TEXT_PATH)
    # Ensure UTF-8 (format standardization)
    text = RAW_TEXT_PATH.read_text(encoding="utf-8", errors="ignore")
    RAW_TEXT_PATH.write_text(text, encoding="utf-8")
    print(f"[+] Saved raw text: {RAW_TEXT_PATH}")
    return RAW_TEXT_PATH

# -----------------------------
# 2. Heuristic Filtering
# -----------------------------
GUTENBERG_HEADER_RE = re.compile(
    r"\*\*\*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*(.+?)\*\*\*END OF",
    re.DOTALL | re.IGNORECASE,
)
SYMBOL_HEAVY_RE = re.compile(r"^[\W\d_]{3,}$")  # lines that are mostly symbols

def heuristic_filter(text: str) -> str:
    """Remove Gutenberg boilerplate, low-quality lines, and excessive whitespace."""
    # Strip header/footer boilerplate
    text = GUTENBERG_HEADER_RE.sub("", text)
    # Fallback: strip common Gutenberg markers
    text = re.sub(r"^\*{3,}.*\*{3,}$", "", text, flags=re.MULTILINE)

    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Drop lines that are >50% non-letter characters
        alpha = sum(c.isalpha() for c in line)
        if len(line) > 0 and alpha / len(line) < 0.5:
            continue
        # Drop symbol-heavy short lines
        if SYMBOL_HEAVY_RE.match(line):
            continue
        # Drop very short lines (likely noise)
        if len(line) < 4:
            continue
        lines.append(line)
    return "\n".join(lines)

# -----------------------------
# 3. Exact Deduplication (SHA-1)
# -----------------------------
def exact_dedup(paragraphs: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in paragraphs:
        h = hashlib.sha1(p.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(p)
    return out

# -----------------------------
# 4. Fuzzy Deduplication (MinHash LSH)
# -----------------------------
def _minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(text, 3):
        m.update(shingle.encode("utf-8"))
    return m

def _shingles(text: str, k: int = 3) -> Iterator[str]:
    tokens = text.lower().split()
    for i in range(len(tokens) - k + 1):
        yield " ".join(tokens[i : i + k])

def fuzzy_dedup(paragraphs: List[str], threshold: float = 0.7) -> List[str]:
    """Remove near-duplicate paragraphs using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept: List[str] = []
    for idx, p in enumerate(paragraphs):
        mh = _minhash(p)
        key = f"p{idx}"
        if not lsh.query(mh):
            lsh.insert(key, mh)
            kept.append(p)
    return kept

# -----------------------------
# 5. Toxicity / PII Mitigation (lightweight)
# -----------------------------
def scrub_pii(text: str) -> str:
    """Lightweight regex-based PII scrubbing (emails, phones)."""
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
    return text

# -----------------------------
# Main pipeline
# -----------------------------
def run_data_pipeline() -> Path:
    print("=" * 60)
    print("[PHASE 1] Data Pipeline")
    print("=" * 60)

    # 1. Download
    download_shakespeare()
    raw = RAW_TEXT_PATH.read_text(encoding="utf-8")

    # 2. Heuristic filter
    print("[*] Applying heuristic filters...")
    filtered = heuristic_filter(raw)

    # 3. Split into paragraphs for dedup
    paragraphs = [p.strip() for p in filtered.split("\n\n") if p.strip()]
    print(f"[*] Paragraphs before dedup: {len(paragraphs)}")

    # 4. Exact dedup
    paragraphs = exact_dedup(paragraphs)
    print(f"[*] After exact dedup (SHA-1): {len(paragraphs)}")

    # 5. Fuzzy dedup
    paragraphs = fuzzy_dedup(paragraphs, threshold=0.7)
    print(f"[*] After fuzzy dedup (MinHash LSH): {len(paragraphs)}")

    # 6. PII scrub
    paragraphs = [scrub_pii(p) for p in paragraphs]

    clean_text = "\n\n".join(paragraphs)
    CLEAN_TEXT_PATH.write_text(clean_text, encoding="utf-8")
    print(f"[+] Clean text saved: {CLEAN_TEXT_PATH} ({len(clean_text)} chars)")
    return CLEAN_TEXT_PATH


if __name__ == "__main__":
    run_data_pipeline()