"""Step 4: Split merged corpus into train/val/test sets for model training."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

IN_PATH = PROC / "all_abc.txt"   # Input: Merged corpus from clean step
TRAIN_PATH = PROC / "train.txt"  # Output: Training set (98%)
VAL_PATH = PROC / "val.txt"      # Output: Validation set (1%)
TEST_PATH = PROC / "test.txt"    # Output: Test set (1%)

TRAIN_FRAC = 0.98  # 98% for training
VAL_FRAC = 0.01    # 1% for validation
TEST_FRAC = 0.01   # 1% for testing


def main() -> None:
    """Split the merged corpus into train/val/test by character count."""
    if abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing merged corpus: {IN_PATH}")

    text = IN_PATH.read_text(encoding="utf-8", errors="ignore")
    n = len(text)
    if n == 0:
        raise ValueError("Empty corpus")

    # Calculate split boundaries
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    n_test = n - n_train - n_val

    # Split text sequentially (not shuffled to preserve tune structure)
    train = text[:n_train]
    val = text[n_train:n_train + n_val]
    test = text[n_train + n_val:]

    # Write split files
    TRAIN_PATH.write_text(train, encoding="utf-8")
    VAL_PATH.write_text(val, encoding="utf-8")
    TEST_PATH.write_text(test, encoding="utf-8")

    print(f"[split] n={n:,} train={len(train):,} val={len(val):,} test={len(test):,}")
    print(f"[split] wrote: {TRAIN_PATH.name}, {VAL_PATH.name}, {TEST_PATH.name}")


if __name__ == "__main__":
    main()