"""Step 3: Clean individual ABC files and merge into a single corpus for training."""
from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import hashlib
import re
import string

ROOT = Path(__file__).resolve().parent.parent
ABC_DIR = ROOT / "data" / "raw"          # Input: Verified ABC files
OUT_DIR = ROOT / "data" / "processed"    # Output: Merged corpus
OUT_PATH = OUT_DIR / "all_abc.txt"       # Single file containing all tunes

MIN_LEN = 64    # Minimum characters per tune (skip very short pieces)
MAX_LEN = 8192  # Maximum characters per tune (truncate very long pieces)

ALLOWED = set(string.printable)  # Only keep printable ASCII characters
NOTE_RE = re.compile(r"[A-Ga-g]")  # Pattern to detect music notes
KEY_RE = re.compile(r"(?m)^\s*K:")  # Pattern to detect key signature header


def clean_text(s: str) -> str:
    """Remove comments, filter non-printable chars, and enforce length limits."""
    lines = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("%"):  # Skip ABC comments
            continue
        lines.append(t)

    s = "\n".join(lines)
    s = "".join(ch for ch in s if ch in ALLOWED)  # Keep only printable ASCII

    if len(s) > MAX_LEN:
        s = s[:MAX_LEN]  # Truncate overly long pieces

    return s.strip()


def is_valid(s: str) -> bool:
    """Check if cleaned text is valid ABC with key signature and notes."""
    if len(s) < MIN_LEN:
        return False
    if KEY_RE.search(s) is None:  # Must have key signature
        return False
    if NOTE_RE.search(s) is None:  # Must have actual notes
        return False
    return True


def main() -> None:
    if not ABC_DIR.exists():
        raise FileNotFoundError(f"Missing ABC directory: {ABC_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(ABC_DIR.glob("*.abc"))
    print(f"[clean] input files: {len(files)}")

    kept = 0
    dropped = 0
    dup = 0
    seen: set[str] = set()
    tunes: list[str] = []

    for p in tqdm(files, desc="Cleaning"):
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            dropped += 1
            continue

        s = clean_text(raw)
        if not is_valid(s):
            dropped += 1
            continue

        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        if h in seen:
            dup += 1
            continue
        seen.add(h)

        tunes.append(s)
        kept += 1

    corpus = "\n\n".join(tunes)
    OUT_PATH.write_text(corpus, encoding="utf-8")

    avg = (len(corpus) // kept) if kept else 0
    print(f"[clean] kept={kept} dropped={dropped} dup={dup}")
    print(f"[clean] chars={len(corpus):,} avg_chars_per_tune={avg:,}")
    print(f"[clean] wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()