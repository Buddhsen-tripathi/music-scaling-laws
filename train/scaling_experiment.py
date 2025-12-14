"""
Run scaling experiments for both Transformer and LSTM models.
Trains models of varying sizes and collects statistics for scaling law analysis.
"""
from __future__ import annotations

from pathlib import Path
import json
import argparse

import numpy as np

from train.trainer import TrainConfig, train
from config.constants import BATCH_SIZE, BLOCK_SIZE


def run_scaling_experiment(
    model_type: str,
    sizes: list[str],
    data_dir: Path,
    save_dir: Path,
    batch_size: int = BATCH_SIZE,
    block_size: int = BLOCK_SIZE,
    max_iters: int | None = None,
    force: bool = False,
) -> list[dict]:
    """Run scaling experiment for a model type across multiple sizes.
    
    Args:
        force: If True, retrain models even if final checkpoint exists
    """
    results = []
    
    for size in sizes:
        print(f"\n{'#'*60}")
        print(f"# {model_type.upper()} - {size.upper()}")
        print(f"{'#'*60}")
        
        config = TrainConfig(
            model_type=model_type,
            model_size=size,
            batch_size=batch_size,
            block_size=block_size,
            max_iters=max_iters,
            data_dir=data_dir,
            save_dir=save_dir / model_type,
        )

        if model_type == "transformer" and size == "xl":
            train_tokens = int(np.load(data_dir / "train.npy", mmap_mode="r").shape[0])
            config.max_tokens = train_tokens
            print(f"[override] transformer/xl max_tokens set to full epoch: {train_tokens:,}")
        
        try:
            stats = train(config, force=force)
            results.append(stats)
        except Exception as e:
            print(f"Error training {model_type}/{size}: {e}")
            results.append({
                "model_type": model_type,
                "model_size": size,
                "error": str(e),
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run scaling experiments")
    parser.add_argument("--model-type", choices=["transformer", "lstm", "both"], default="both")
    parser.add_argument("--sizes", nargs="+", default=["tiny", "small", "medium", "large", "xl"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--max-iters", type=int, default=None, help="Training iterations per model (None = 1 epoch)")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="scaling_results.json")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    
    all_results = []
    
    if args.model_type in ["transformer", "both"]:
        transformer_results = run_scaling_experiment(
            "transformer", args.sizes, data_dir, save_dir, args.batch_size, args.block_size, args.max_iters
        )
        all_results.extend(transformer_results)
    
    if args.model_type in ["lstm", "both"]:
        lstm_results = run_scaling_experiment(
            "lstm", args.sizes, data_dir, save_dir, args.batch_size, args.block_size, args.max_iters
        )
        all_results.extend(lstm_results)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SCALING EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Model':<12} {'Size':<8} {'Params':>12} {'Val Loss':>10} {'Time (s)':>10}")
    print("-"*60)
    for r in all_results:
        if "error" not in r:
            print(f"{r['model_type']:<12} {r['model_size']:<8} {r['num_params']:>12,} {r['val_loss']:>10.4f} {r['train_time_sec']:>10.1f}")


if __name__ == "__main__":
    main()
