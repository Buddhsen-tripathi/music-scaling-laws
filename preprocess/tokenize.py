"""Step 5: Tokenize text splits into numpy arrays for efficient model training."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# Input: Text splits from build_dataset
TRAIN_TXT = PROC / "train.txt"
VAL_TXT = PROC / "val.txt"
TEST_TXT = PROC / "test.txt"

# Output: Tokenized numpy arrays and vocabulary mapping
VOCAB_JSON = PROC / "vocab.json"  # Character-to-index mapping
TRAIN_NPY = PROC / "train.npy"    # Tokenized training data
VAL_NPY = PROC / "val.npy"        # Tokenized validation data
TEST_NPY = PROC / "test.npy"      # Tokenized test data


def build_vocab(train_text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-level vocabulary from training text."""
    chars = sorted(set(train_text))  # All unique characters
    stoi = {ch: i for i, ch in enumerate(chars)}  # String to index
    itos = {i: ch for ch, i in stoi.items()}      # Index to string
    return stoi, itos


def encode(text: str, stoi: dict[str, int], unk_id: int = 0) -> np.ndarray:
    """Encode text to token IDs. Unknown chars map to unk_id."""
    ids = np.fromiter((stoi.get(c, unk_id) for c in text), dtype=np.int32)
    return ids


def main() -> None:
    for p in (TRAIN_TXT, VAL_TXT, TEST_TXT):
        if not p.exists():
            raise FileNotFoundError(f"Missing split: {p}")

    train_text = TRAIN_TXT.read_text(encoding="utf-8", errors="ignore")
    val_text = VAL_TXT.read_text(encoding="utf-8", errors="ignore")
    test_text = TEST_TXT.read_text(encoding="utf-8", errors="ignore")

    stoi, itos = build_vocab(train_text)

    vocab_obj = {
        "vocab_size": len(stoi),
        "stoi": stoi,
        "itos": itos,
    }
    VOCAB_JSON.write_text(json.dumps(vocab_obj, ensure_ascii=False), encoding="utf-8")

    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)
    test_ids = encode(test_text, stoi)

    np.save(TRAIN_NPY, train_ids)
    np.save(VAL_NPY, val_ids)
    np.save(TEST_NPY, test_ids)

    print(f"[tok] vocab_size={len(stoi)}")
    print(f"[tok] train={train_ids.size:,} val={val_ids.size:,} test={test_ids.size:,}")
    print(f"[tok] wrote: {VOCAB_JSON.name}, train/val/test.npy")


if __name__ == "__main__":
    main()
