#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from music21 import converter, environment
from tqdm import tqdm
import multiprocessing as mp
import warnings
import random
import os


ROOT = Path(__file__).resolve().parent.parent
LMD_DIR = ROOT / "data" / "lmd_full"
ABC_DIR = ROOT / "data" / "raw"

MAX_FILES: int | None = 120_000          # Cap number of MIDI files to convert
N_WORKERS: int = max(4, (os.cpu_count() or 8) - 1)  # Worker processes for CPU parallelism
CHUNK_SIZE: int = 8                     # Batch size per worker to reduce overhead

warnings.filterwarnings("ignore")
us = environment.UserSettings()
try:
    us["warnings"] = 0
except Exception:
    pass


def out_key_for(midi_path: Path) -> str:
    rel = midi_path.relative_to(LMD_DIR)
    return f"{rel.parts[0]}_{midi_path.stem}"


def convert_one(midi_path_str: str) -> bool:
    midi_path = Path(midi_path_str)
    key = out_key_for(midi_path)
    abc_path = ABC_DIR / f"{key}.abc"

    try:
        score = converter.parse(midi_path)
        score.write("abc", fp=str(abc_path))
        return True
    except Exception:
        return False


def main() -> None:
    print("=== Lakh MIDI -> ABC ===")
    print("LMD:", LMD_DIR)
    print("OUT:", ABC_DIR)

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
    if MAX_FILES is not None:
        midi_to_convert = midi_to_convert[:MAX_FILES]
        print(f"Using subset: {len(midi_to_convert)}")

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