"""Step 1: Convert Lakh MIDI files to ABC notation using midi2abc CLI tool."""
from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import subprocess
import random
import re
import os


ROOT = Path(__file__).resolve().parent.parent
LMD_DIR = ROOT / "data" / "lmd_full"  # Source: Lakh MIDI Dataset
ABC_DIR = ROOT / "data" / "raw"        # Output: Converted ABC files

MAX_FILES: int = 120000     # Target number of MIDI files to convert
N_WORKERS: int = max(4, (os.cpu_count() or 8) - 1)  # Parallel workers for faster conversion
CHUNK_SIZE: int = 8         # Files per worker batch

MIN_ABC_SIZE: int = 150     # Skip files smaller than this (likely metadata only)
MAX_ABC_SIZE: int = 300_000 # Truncate to avoid pathological blowups (see Gwern)


def out_key_for(midi_path: Path) -> str:
    """Generate unique output filename from MIDI path."""
    rel = midi_path.relative_to(LMD_DIR)
    return f"{rel.parts[0]}_{midi_path.stem}"


def clean_abc_content(raw: str) -> str:
    """
    Clean ABC content:
    - Remove comment lines (% but not %%)
    - Remove lyrics (w:)
    - Remove error/warning lines
    - Remove copyright notices
    - Optionally remove spaces in non-MIDI-directive lines
    """
    lines = []
    for line in raw.splitlines():
        # Skip various noise
        if line.startswith("% ") and not line.startswith("%%"):
            continue
        if line.startswith("w:"):  # lyrics
            continue
        if "Missing time signature" in line:
            continue
        if "Error " in line:
            continue
        if "Copyright" in line or "All rights reserved" in line:
            continue
        # Remove standalone % comments (but keep %% MIDI directives)
        if re.match(r"^%[^%]", line):
            continue
        lines.append(line)
    
    return "\n".join(lines)


def convert_one(midi_path_str: str) -> bool:
    """Convert a single MIDI file to ABC using midi2abc CLI."""
    midi_path = Path(midi_path_str)
    key = out_key_for(midi_path)
    abc_path = ABC_DIR / f"{key}.abc"

    try:
        # Use Gwern's recommended flags:
        # -bpl 999999: avoid excessive newlines (bars per line)
        # -nogr: no grace notes (cleaner output)
        result = subprocess.run(
            ["midi2abc", str(midi_path), "-bpl", "999999", "-nogr"],
            capture_output=True,
            timeout=60,
            text=True,
        )
        
        if result.returncode != 0:
            return False
        
        raw_abc = result.stdout
        
        # Check for pathological blowup
        if len(raw_abc) > MAX_ABC_SIZE:
            raw_abc = raw_abc[:MAX_ABC_SIZE]
        
        # Clean the content
        cleaned = clean_abc_content(raw_abc)
        
        # Validate: must have minimum size and required headers
        if len(cleaned) < MIN_ABC_SIZE:
            return False
        if "X:" not in cleaned and "K:" not in cleaned:
            return False
        
        # Write cleaned ABC
        abc_path.write_text(cleaned, encoding="utf-8")
        return True
        
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def check_midi2abc() -> bool:
    """Check if midi2abc is installed."""
    try:
        result = subprocess.run(["midi2abc", "-h"], capture_output=True, timeout=5)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return True  # Assume installed if other error


def main(max_files: int | None = None) -> None:
    print("=== Lakh MIDI -> ABC ===")
    print("LMD:", LMD_DIR)
    print("OUT:", ABC_DIR)

    if not check_midi2abc():
        raise RuntimeError(
            "midi2abc not found. Install with: brew install abcmidi"
        )

    if not LMD_DIR.exists():
        raise FileNotFoundError(f"Missing LMD directory: {LMD_DIR}")

    ABC_DIR.mkdir(parents=True, exist_ok=True)

    all_midis = list(LMD_DIR.rglob("*.mid")) + list(LMD_DIR.rglob("*.MID"))
    print(f"Found MIDI: {len(all_midis)}")

    already_done = {p.stem for p in ABC_DIR.glob("*.abc")}
    print(f"Already ABC: {len(already_done)}")

    midi_to_convert: list[Path] = []
    for p in all_midis:
        if out_key_for(p) not in already_done:
            midi_to_convert.append(p)

    print(f"Remaining: {len(midi_to_convert)}")

    random.shuffle(midi_to_convert)
    if max_files is not None:
        budget = max(0, max_files - len(already_done))
        midi_to_convert = midi_to_convert[:budget]
        print(f"Using subset: {len(midi_to_convert)} (budget={budget})")

    if not midi_to_convert:
        print("Nothing to do.")
        return

    midi_paths = [str(p) for p in midi_to_convert]

    print(f"Workers: {N_WORKERS} | chunk: {CHUNK_SIZE}")

    converted = 0
    failed = 0

    with mp.Pool(processes=N_WORKERS, maxtasksperchild=128) as pool:
        for ok in tqdm(
            pool.imap_unordered(convert_one, midi_paths, chunksize=CHUNK_SIZE),
            total=len(midi_paths),
            desc="Converting",
        ):
            if ok:
                converted += 1
            else:
                failed += 1

    print(f"Done. converted={converted} failed={failed} out={ABC_DIR.resolve()}")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()