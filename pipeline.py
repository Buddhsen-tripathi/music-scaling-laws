"""
Main pipeline for Scaling Laws for Music Language Models.

Pipeline steps:
    1. convert   - MIDI → ABC using midi2abc
    2. verify    - Validate ABC files and remove corrupted ones
    3. clean     - Clean and merge ABC files into single corpus
    4. split     - Split corpus into train/val/test (98/1/1)
    5. tokenize  - Convert text to numpy arrays for training
    6. train     - Train transformer and LSTM models (scaling experiment)
    7. evaluate  - Generate scaling plots and analysis
    8. generate  - Generate music samples from best model

Usage:
    python pipeline.py                        # Run full pipeline
    python pipeline.py --from clean           # Start from clean step
    python pipeline.py --to tokenize          # Stop after tokenize (preprocessing only)
    python pipeline.py --from train --to train  # Only run training
    python pipeline.py --force                # Force re-run all steps
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""
    root: Path
    lmd_dir: Path
    abc_raw_dir: Path
    processed_dir: Path

    # preprocessing targets
    target_abc_files: int | None = 120000

    # dataset split
    train_frac: float = 0.98
    val_frac: float = 0.01
    test_frac: float = 0.01

    # tokenization choice
    tokenization: str = "char"


# Pipeline execution order
ORDER = ["convert", "verify", "clean", "split", "tokenize", "train", "evaluate", "generate"]


def default_config() -> PipelineConfig:
    """Create default configuration with standard directory structure."""
    root = Path(__file__).resolve().parent
    return PipelineConfig(
        root=root,
        lmd_dir=root / "data" / "lmd_full",      # Lakh MIDI Dataset
        abc_raw_dir=root / "data" / "raw",        # Converted ABC files
        processed_dir=root / "data" / "processed", # Final tokenized data
    )


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def count_files(folder: Path, pattern: str) -> int:
    """Count files matching pattern in folder."""
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

    conv.main(max_files=cfg.target_abc_files)


def step_verify(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Verify ABC files are valid before cleaning.
    Removes invalid files that would pollute the corpus.
    """
    from preprocess import verify_abc as va

    abc_count = count_files(cfg.abc_raw_dir, "*.abc")
    if abc_count == 0:
        print("[verify] No ABC files to verify")
        return

    print(f"[verify] Verifying {abc_count} ABC files...")
    va.main(strict=False, delete_invalid=True)


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


def step_train(cfg: PipelineConfig, force: bool = False, model_type: str = "both") -> None:
    """
    Train transformer and LSTM models for scaling experiment.
    Trains 5 transformer sizes and 4 LSTM sizes, each for exactly 1 epoch.
    
    Args:
        model_type: "transformer", "lstm", or "both"
    """
    from train.scaling_experiment import run_scaling_experiment

    results_path = cfg.root / "scaling_results.json"
    checkpoints_dir = cfg.root / "checkpoints"

    import json

    all_results = []

    # Train transformers (5 sizes)
    if model_type in ("transformer", "both"):
        print("\n" + "="*60)
        print("TRANSFORMER SCALING EXPERIMENT")
        print("="*60)
        transformer_results = run_scaling_experiment(
            model_type="transformer",
            sizes=["tiny", "small", "medium", "large", "xl"],
            data_dir=cfg.processed_dir,
            save_dir=checkpoints_dir,
            # batch_size and block_size use defaults from config/constants.py
            max_iters=None,  # 1 epoch
            force=force,
        )
        all_results.extend(transformer_results)

    # Train LSTMs (4 sizes)
    if model_type in ("lstm", "both"):
        print("\n" + "="*60)
        print("LSTM SCALING EXPERIMENT")
        print("="*60)
        lstm_results = run_scaling_experiment(
            model_type="lstm",
            sizes=["tiny", "small", "medium", "large"],
            data_dir=cfg.processed_dir,
            save_dir=checkpoints_dir,
            # batch_size and block_size use defaults from config/constants.py
            max_iters=None,  # 1 epoch
            force=force,
        )
        all_results.extend(lstm_results)

    # Save results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[train] Results saved to {results_path}")


def step_evaluate(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Generate scaling plots and analysis from training results.
    """
    from eval.scaling_analysis import main as run_analysis

    results_path = cfg.root / "scaling_results.json"
    report_dir = cfg.root / "report"

    if not results_path.exists():
        print("[evaluate] No scaling_results.json found, skipping")
        return

    report_dir.mkdir(parents=True, exist_ok=True)
    run_analysis(str(results_path), str(report_dir))
    print(f"[evaluate] Plots saved to {report_dir}")


def step_generate(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Generate music samples from the best trained model.
    """
    from eval.generate import main as run_generate

    checkpoints_dir = cfg.root / "checkpoints"
    samples_dir = cfg.root / "samples"

    # Find best checkpoint (prefer larger transformer, use _final.pt)
    for size in ["xl", "large", "medium", "small", "tiny"]:
        ckpt = checkpoints_dir / "transformer" / f"transformer_{size}_final.pt"
        if ckpt.exists():
            best_ckpt = ckpt
            break
    else:
        print("[generate] No checkpoints found, skipping")
        return

    samples_dir.mkdir(parents=True, exist_ok=True)
    print(f"[generate] Using checkpoint: {best_ckpt}")
    run_generate(str(best_ckpt), str(samples_dir), num_samples=10)
    print(f"[generate] Samples saved to {samples_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="Scaling Laws for Music Language Models Pipeline")
    ap.add_argument("--from", dest="from_step", default="convert", choices=ORDER,
                    help="Start from this step")
    ap.add_argument("--to", dest="to_step", default="generate", choices=ORDER,
                    help="Stop after this step")
    ap.add_argument("--force", action="store_true", help="Force re-run all steps")
    ap.add_argument("--model-type", choices=["transformer", "lstm", "both"], default="both",
                    help="Model type to train (for train step)")
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
        elif step == "verify":
            step_verify(cfg, args.force)
        elif step == "clean":
            step_clean_merge(cfg, args.force)
        elif step == "split":
            step_split(cfg, args.force)
        elif step == "tokenize":
            step_tokenize(cfg, args.force)
        elif step == "train":
            step_train(cfg, args.force, args.model_type)
        elif step == "evaluate":
            step_evaluate(cfg, args.force)
        elif step == "generate":
            step_generate(cfg, args.force)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()