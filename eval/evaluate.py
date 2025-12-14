"""
Model evaluation: perplexity, validity metrics, and qualitative analysis.
"""
from __future__ import annotations

from pathlib import Path
import json
import argparse
import math

import torch
import numpy as np
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.transformer import TransformerLM, TransformerConfig
from models.lstm import LSTMLM, LSTMConfig
from train.trainer import MusicDataset


def load_model(checkpoint_path: Path, device: str = "cpu"):
    """Load a trained model from checkpoint."""
    import __main__ as _main

    try:
        from train.trainer import TrainConfig

        if not hasattr(_main, "TrainConfig"):
            setattr(_main, "TrainConfig", TrainConfig)
    except Exception:
        pass
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]
    
    if isinstance(model_config, TransformerConfig):
        model = TransformerLM(model_config)
    else:
        model = LSTMLM(model_config)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, model_config, ckpt


@torch.no_grad()
def compute_perplexity(
    model,
    data_path: Path,
    block_size: int,
    batch_size: int = 32,
    device: str = "cpu",
    max_batches: int = 50,
) -> float:
    """Compute perplexity on a dataset."""
    data = np.load(data_path, mmap_mode="r")
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"Dataset too small for block_size={block_size}: {data_path}")

    total_loss = 0.0
    total_tokens = 0

    for _ in range(max_batches):
        starts = np.random.randint(0, max_start, size=(batch_size,))
        x_np = np.stack([data[i : i + block_size] for i in starts]).astype(np.int64, copy=False)
        y_np = np.stack([data[i + 1 : i + 1 + block_size] for i in starts]).astype(np.int64, copy=False)

        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate_model(
    checkpoint_path: Path,
    data_dir: Path,
    device: str = "cpu",
    batch_size: int = 32,
    max_batches: int = 50,
) -> dict:
    """
    Evaluate a model on test set.
    
    Returns:
        dict with test_loss, test_perplexity, num_params, etc.
    """
    model, config, ckpt = load_model(checkpoint_path, device)
    
    block_size = config.block_size
    
    # Test perplexity
    test_ppl = compute_perplexity(
        model,
        data_dir / "test.npy",
        block_size,
        batch_size=batch_size,
        device=device,
        max_batches=max_batches,
    )
    
    # Val perplexity (for comparison)
    val_ppl = compute_perplexity(
        model,
        data_dir / "val.npy",
        block_size,
        batch_size=batch_size,
        device=device,
        max_batches=max_batches,
    )
    
    results = {
        "checkpoint": str(checkpoint_path),
        "model_type": "transformer" if isinstance(config, TransformerConfig) else "lstm",
        "num_params": model.num_params(),
        "val_loss": ckpt.get("final_val_loss", -1),
        "val_perplexity": val_ppl,
        "test_perplexity": test_ppl,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint to evaluate")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory with all checkpoints")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=50)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    results = []
    
    if args.checkpoint:
        # Evaluate single checkpoint
        r = evaluate_model(
            Path(args.checkpoint),
            data_dir,
            device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
        results.append(r)
    else:
        # Evaluate all checkpoints in directory
        ckpt_dir = Path(args.checkpoints_dir)
        for ckpt_path in sorted(ckpt_dir.rglob("*_final.pt")):
            print(f"\nEvaluating: {ckpt_path}")
            try:
                r = evaluate_model(
                    ckpt_path,
                    data_dir,
                    device,
                    batch_size=args.batch_size,
                    max_batches=args.max_batches,
                )
                results.append(r)
                print(f"  Test PPL: {r['test_perplexity']:.2f}")
            except Exception as e:
                print(f"  Error: {e}")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Model':<12} {'Params':>12} {'Val PPL':>10} {'Test PPL':>10}")
    print("-"*60)
    for r in sorted(results, key=lambda x: x["num_params"]):
        name = f"{r['model_type']}"
        print(f"{name:<12} {r['num_params']:>12,} {r['val_perplexity']:>10.2f} {r['test_perplexity']:>10.2f}")


if __name__ == "__main__":
    main()
