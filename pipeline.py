from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys


@dataclass
class PipelineConfig:
    root: Path
    lmd_dir: Path
    abc_raw_dir: Path
    processed_dir: Path

    # preprocessing targets
    target_abc_files: int | None = 100_000

    # dataset split
    train_frac: float = 0.98
    val_frac: float = 0.01
    test_frac: float = 0.01

    # tokenization choice
    tokenization: str = "char"


ORDER = ["convert", "clean", "split", "tokenize"]


def default_config() -> PipelineConfig:
    root = Path(__file__).resolve().parent
    return PipelineConfig(
        root=root,
        lmd_dir=root / "data" / "lmd_full",
        abc_raw_dir=root / "data" / "raw",
        processed_dir=root / "data" / "processed",
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def count_files(folder: Path, pattern: str) -> int:
    return sum(1 for _ in folder.glob(pattern)) if folder.exists() else 0


def step_convert(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Convert Lakh MIDI → ABC.
    Designed to be resumable and skip already converted files.
    """
    from preprocess import convert_lmd_to_abc as conv

    ensure_dir(cfg.abc_raw_dir)
    already = count_files(cfg.abc_raw_dir, "*.abc")
    print(f"[convert] ABC files present: {already}")

    if (not force) and cfg.target_abc_files is not None and already >= cfg.target_abc_files:
        print(f"[convert] target reached ({cfg.target_abc_files}), skipping")
        return

    conv.main()


def step_clean_merge(cfg: PipelineConfig, force: bool = False) -> Path:
    """
    Clean individual ABC files and merge into a single corpus.
    """
    from preprocess import clean_and_merge_abc as cm

    ensure_dir(cfg.processed_dir)
    out_path = cfg.processed_dir / "all_abc.txt"

    if out_path.exists() and not force:
        print(f"[clean] {out_path.name} exists, skipping")
        return out_path

    cm.main()
    return out_path


def step_split(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Split merged corpus into train/val/test text files.
    """
    from preprocess import build_dataset as bd

    train_txt = cfg.processed_dir / "train.txt"
    val_txt = cfg.processed_dir / "val.txt"
    test_txt = cfg.processed_dir / "test.txt"

    if (not force) and train_txt.exists() and val_txt.exists() and test_txt.exists():
        print("[split] train/val/test already exist, skipping")
        return

    bd.main()


def step_tokenize(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Tokenize text splits into numpy arrays for model training.
    """
    from preprocess import tokenize as tok

    train_npy = cfg.processed_dir / "train.npy"
    val_npy = cfg.processed_dir / "val.npy"
    test_npy = cfg.processed_dir / "test.npy"
    vocab = cfg.processed_dir / "vocab.json"

    if (
        not force
        and train_npy.exists()
        and val_npy.exists()
        and test_npy.exists()
        and vocab.exists()
    ):
        print("[tokenize] tokenized data exists, skipping")
        return

    tok.main()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_step", default="convert", choices=ORDER)
    ap.add_argument("--to", dest="to_step", default="tokenize", choices=ORDER)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = default_config()

    start = ORDER.index(args.from_step)
    end = ORDER.index(args.to_step)
    if start > end:
        print("Invalid pipeline range")
        sys.exit(1)

    steps = ORDER[start : end + 1]
    print(f"Running pipeline steps: {steps} (force={args.force})")

    for step in steps:
        if step == "convert":
            step_convert(cfg, args.force)
        elif step == "clean":
            step_clean_merge(cfg, args.force)
        elif step == "split":
            step_split(cfg, args.force)
        elif step == "tokenize":
            step_tokenize(cfg, args.force)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()