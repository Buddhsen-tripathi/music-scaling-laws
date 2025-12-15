from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def iter_tunes_from_abc_file(path: Path):
    current: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line.strip():
                if current:
                    yield "\n".join(current).strip()
                    current = []
                continue

            current.append(line)

    if current:
        yield "\n".join(current).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default="data/processed/all_abc.txt",
        help="Path to merged ABC corpus (defaults to data/processed/all_abc.txt)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="report/length_hist.png",
        help="Output PNG path (defaults to report/length_hist.png)",
    )
    ap.add_argument("--bins", type=int, default=100, help="Histogram bins")
    ap.add_argument(
        "--max-len",
        type=int,
        default=4096,
        help="Reference maximum length in characters (used for optional filtering and plot annotation)",
    )
    ap.add_argument(
        "--exclude-over-max-len",
        action="store_true",
        help="Exclude tunes with length > max-len from the histogram and report how many were excluded",
    )
    ap.add_argument(
        "--log-y",
        action="store_true",
        help="Use a log-scaled y-axis (often clearer due to the spike near max length)",
    )
    ap.add_argument(
        "--no-max-len-line",
        action="store_true",
        help="Do not draw a vertical reference line at max-len",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        fallback = Path("data/processed/train.txt")
        if fallback.exists():
            input_path = fallback
        else:
            raise FileNotFoundError(f"Missing input corpus: {args.input} (and fallback train.txt not found)")

    lengths: list[int] = []
    for tune in iter_tunes_from_abc_file(input_path):
        if tune:
            lengths.append(len(tune))

    excluded = 0
    if args.exclude_over_max_len:
        kept: list[int] = []
        for l in lengths:
            if l <= args.max_len:
                kept.append(l)
            else:
                excluded += 1
        lengths = kept

    if not lengths:
        raise RuntimeError(f"No tunes found in {input_path}. Expected records starting with 'X:'.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=args.bins, color="#4C78A8", edgecolor="white")
    plt.title("ABC Tune Length Distribution (characters)")
    plt.xlabel("Tune length (characters)")
    plt.ylabel("Count")

    if not args.no_max_len_line:
        plt.axvline(
            args.max_len,
            color="#E45756",
            linestyle="--",
            linewidth=1.5,
            label=f"MAX_LEN={args.max_len}",
        )
        plt.legend(loc="upper right")

    if args.log_y:
        plt.yscale("log")
        plt.ylabel("Count (log scale)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    median = lengths_sorted[n // 2] if n % 2 == 1 else (lengths_sorted[n // 2 - 1] + lengths_sorted[n // 2]) / 2
    mean = sum(lengths_sorted) / n

    print(f"Input: {input_path}")
    print(f"Tunes: {n}")
    print(f"Min length: {min(lengths_sorted)}")
    print(f"Median length: {median}")
    print(f"Mean length: {mean:.2f}")
    print(f"Max length: {max(lengths_sorted)}")
    if args.exclude_over_max_len:
        print(f"Excluded > MAX_LEN ({args.max_len}): {excluded}")
    print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
