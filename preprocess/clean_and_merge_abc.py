"""Step 3: Clean individual ABC files and merge into a single corpus for training.

Cleaning steps (based on Gwern's GPT-2 music preprocessing):
- Remove comments (lines starting with %)
- Remove voice markers (V:) for single-voice output
- Remove lyrics (w: lines)
- Remove guitar chords ("..." annotations)
- Strip long/messy titles from MIDI paths
- Keep only essential headers: X, T, M, L, Q, K
- Filter non-printable characters
- Deduplicate by content hash
"""
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
MAX_LEN = 4096  # Maximum characters per tune (truncate very long pieces)

ALLOWED = set(string.printable)  # Only keep printable ASCII characters
NOTE_RE = re.compile(r"[A-Ga-g]")  # Pattern to detect music notes
KEY_RE = re.compile(r"(?m)^\s*K:")  # Pattern to detect key signature header

# Headers to keep (essential ABC metadata)
KEEP_HEADERS = {"X:", "T:", "M:", "L:", "Q:", "K:"}
# Headers to skip (voice, lyrics, guitar chords, etc.)
SKIP_PREFIXES = ("V:", "w:", "W:", "%%", "I:", "N:", "H:", "R:", "B:", "D:", "F:", "G:", "O:", "P:", "S:", "Z:")


def clean_text(s: str) -> str:
    """Remove comments, voice markers, lyrics, and enforce length limits."""
    lines = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue
        # Skip comments
        if t.startswith("%"):
            continue
        # Skip voice markers, lyrics, and other non-essential headers
        if t.startswith(SKIP_PREFIXES):
            continue
        # Clean up messy titles (from MIDI file paths)
        if t.startswith("T:"):
            # Simplify long path-based titles
            if "/Users/" in t or "/home/" in t or len(t) > 60:
                t = "T:Untitled"
        # Remove inline guitar chords like "Cm" "G7" etc in quotes
        t = re.sub(r'"[^"]*"', '', t)
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