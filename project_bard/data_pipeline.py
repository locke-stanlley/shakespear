"""
data_pipeline.py - Multi-source data collection and curation
"""
import re
import hashlib
import urllib.request
from pathlib import Path
from typing import List, Iterator

from datasketch import MinHash, MinHashLSH
from config import (
    DATA_SOURCES, RAW_DIR, CLEAN_TEXT_PATH, RAW_TEXT_PATH
)


def download_all_sources() -> Path:
    """Download all configured data sources and concatenate."""
    all_text = []
    for i, url in enumerate(DATA_SOURCES):
        filename = RAW_DIR / f"source_{i}.txt"
        if not filename.exists():
            print(f"[*] Downloading {url} -> {filename}")
            try:
                urllib.request.urlretrieve(url, filename)
                text = filename.read_text(encoding="utf-8", errors="ignore")
                filename.write_text(text, encoding="utf-8")
            except Exception as e:
                print(f"[!] Failed to download {url}: {e}")
                continue
        else:
            print(f"[+] Already have {filename}")
        all_text.append(filename.read_text(encoding="utf-8"))
    
    combined = "\n\n".join(all_text)
    RAW_TEXT_PATH.write_text(combined, encoding="utf-8")
    print(f"[+] Combined raw text: {len(combined)} chars from {len(all_text)} sources")
    return RAW_TEXT_PATH


# Keep the same heuristic_filter, exact_dedup, fuzzy_dedup, scrub_pii from before
# ... (use the original functions)

GUTENBERG_HEADER_RE = re.compile(
    r"\*\*\*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*(.+?)\*\*\*END OF",
    re.DOTALL | re.IGNORECASE,
)

def heuristic_filter(text: str) -> str:
    text = GUTENBERG_HEADER_RE.sub("", text)
    text = re.sub(r"^\*{3,}.*\*{3,}$", "", text, flags=re.MULTILINE)
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        alpha = sum(c.isalpha() for c in line)
        if len(line) > 0 and alpha / len(line) < 0.5:
            continue
        if len(line) < 4:
            continue
        lines.append(line)
    return "\n".join(lines)


def exact_dedup(paragraphs: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in paragraphs:
        h = hashlib.sha1(p.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(p)
    return out


def _shingles(text: str, k: int = 3) -> Iterator[str]:
    tokens = text.lower().split()
    for i in range(len(tokens) - k + 1):
        yield " ".join(tokens[i : i + k])


def _minhash(text: str, num_perm: int = 128):
    m = MinHash(num_perm=num_perm)
    for s in _shingles(text, 3):
        m.update(s.encode("utf-8"))
    return m


def fuzzy_dedup(paragraphs: List[str], threshold: float = 0.7) -> List[str]:
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept = []
    for idx, p in enumerate(paragraphs):
        mh = _minhash(p)
        key = f"p{idx}"
        if not lsh.query(mh):
            lsh.insert(key, mh)
            kept.append(p)
    return kept


def scrub_pii(text: str) -> str:
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
    return text


def run_data_pipeline() -> Path:
    print("=" * 60)
    print("[PHASE 1] Multi-Source Data Pipeline")
    print("=" * 60)

    download_all_sources()
    raw = RAW_TEXT_PATH.read_text(encoding="utf-8")

    print("[*] Applying heuristic filters...")
    filtered = heuristic_filter(raw)

    paragraphs = [p.strip() for p in filtered.split("\n\n") if p.strip()]
    print(f"[*] Paragraphs before dedup: {len(paragraphs)}")

    paragraphs = exact_dedup(paragraphs)
    print(f"[*] After exact dedup: {len(paragraphs)}")

    paragraphs = fuzzy_dedup(paragraphs, threshold=0.7)
    print(f"[*] After fuzzy dedup: {len(paragraphs)}")

    paragraphs = [scrub_pii(p) for p in paragraphs]

    clean_text = "\n\n".join(paragraphs)
    CLEAN_TEXT_PATH.write_text(clean_text, encoding="utf-8")
    print(f"[+] Clean text saved: {len(clean_text)} chars")
    return CLEAN_TEXT_PATH


if __name__ == "__main__":
    run_data_pipeline()