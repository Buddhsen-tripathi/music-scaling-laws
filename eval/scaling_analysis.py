"""
Scaling law analysis and visualization.
Fits power law: L = a * N^(-alpha) + c
"""
from __future__ import annotations

from pathlib import Path
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power_law(N: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    """Power law: L = a * N^(-alpha) + c"""
    return a * np.power(N, -alpha) + c


def fit_scaling_law(params: np.ndarray, losses: np.ndarray) -> tuple[tuple, np.ndarray]:
    """
    Fit power law to scaling data.
    
    Returns:
        (a, alpha, c), fitted_losses
    """
    # Initial guesses
    p0 = [1.0, 0.1, min(losses)]
    
    try:
        popt, _ = curve_fit(
            power_law, params, losses,
            p0=p0,
            bounds=([0, 0, 0], [np.inf, 2, np.inf]),
            maxfev=10000,
        )
        fitted = power_law(params, *popt)
        return tuple(popt), fitted
    except Exception as e:
        print(f"Warning: curve fitting failed: {e}")
        return (0, 0, 0), losses


def plot_scaling_laws(
    results: list[dict],
    output_path: Path,
    title: str = "Scaling Laws for Music Language Models",
):
    """
    Create scaling law plots.
    
    Args:
        results: List of training results with num_params, val_loss, model_type
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Separate by model type
    transformer_data = [(r["num_params"], r["val_loss"]) for r in results 
                        if r.get("model_type") == "transformer" and "error" not in r]
    lstm_data = [(r["num_params"], r["val_loss"]) for r in results 
                 if r.get("model_type") == "lstm" and "error" not in r]
    
    colors = {"transformer": "#2ecc71", "lstm": "#e74c3c"}
    
    # Plot 1: Individual scaling curves
    ax1 = axes[0]
    
    for name, data, color in [("Transformer", transformer_data, colors["transformer"]),
                               ("LSTM", lstm_data, colors["lstm"])]:
        if not data:
            continue
        params = np.array([d[0] for d in data])
        losses = np.array([d[1] for d in data])
        
        # Sort by params
        idx = np.argsort(params)
        params, losses = params[idx], losses[idx]
        
        # Fit power law
        (a, alpha, c), fitted = fit_scaling_law(params, losses)
        
        # Plot
        ax1.scatter(params, losses, s=100, c=color, label=f"{name} (α={alpha:.3f})", zorder=5)
        
        # Plot fitted curve
        params_smooth = np.logspace(np.log10(params.min()), np.log10(params.max()), 100)
        fitted_smooth = power_law(params_smooth, a, alpha, c)
        ax1.plot(params_smooth, fitted_smooth, c=color, linestyle="--", alpha=0.7)
        
        print(f"\n{name} Scaling Law:")
        print(f"  L = {a:.4f} * N^(-{alpha:.4f}) + {c:.4f}")
        print(f"  Exponent α = {alpha:.4f}")
    
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Parameters", fontsize=12)
    ax1.set_ylabel("Validation Loss", fontsize=12)
    ax1.set_title("Scaling Laws (Log Scale)", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined comparison
    ax2 = axes[1]
    
    for name, data, color in [("Transformer", transformer_data, colors["transformer"]),
                               ("LSTM", lstm_data, colors["lstm"])]:
        if not data:
            continue
        params = np.array([d[0] for d in data])
        losses = np.array([d[1] for d in data])
        idx = np.argsort(params)
        params, losses = params[idx], losses[idx]
        
        ax2.plot(params, losses, "o-", c=color, label=name, markersize=8, linewidth=2)
    
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Parameters", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.set_title("Transformer vs LSTM Scaling", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def plot_training_curves(
    results: list[dict],
    checkpoints_dir: Path,
    output_path: Path,
):
    """Plot training loss curves for all models."""
    import torch
    import __main__ as _main

    try:
        from train.trainer import TrainConfig

        if not hasattr(_main, "TrainConfig"):
            setattr(_main, "TrainConfig", TrainConfig)
    except Exception:
        pass
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    
    for ax, model_type in zip(axes, ["transformer", "lstm"]):
        ax.set_title(f"{model_type.capitalize()} Training Curves", fontsize=14)
        
        for i, size in enumerate(["tiny", "small", "medium", "large", "xl"]):
            ckpt_path = checkpoints_dir / model_type / f"{model_type}_{size}_final.pt"
            if not ckpt_path.exists():
                continue
            
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            train_losses = ckpt.get("train_losses", [])
            
            if train_losses:
                # Smooth with moving average
                window = max(1, len(train_losses) // 100)
                smoothed = np.convolve(train_losses, np.ones(window)/window, mode="valid")
                ax.plot(smoothed, label=size, color=colors[i], alpha=0.8)
        
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Training Loss", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to: {output_path}")
    plt.close()


def create_results_table(results: list[dict]) -> str:
    """Create a markdown table of results."""
    lines = [
        "| Model | Size | Parameters | Val Loss | Train Time (s) | Tokens/sec |",
        "|-------|------|------------|----------|----------------|------------|",
    ]
    
    for r in sorted(results, key=lambda x: (x.get("model_type", ""), x.get("num_params", 0))):
        if "error" in r:
            continue
        lines.append(
            f"| {r['model_type']} | {r['model_size']} | {r['num_params']:,} | "
            f"{r['val_loss']:.4f} | {r['train_time_sec']:.1f} | {r['tokens_per_sec']:.0f} |"
        )
    
    return "\n".join(lines)


def main(results_path: str = "scaling_results.json", output_dir: str = "report", 
         checkpoints_dir: str = "checkpoints"):
    """Run scaling analysis. Can be called from pipeline or CLI."""
    results_path = Path(results_path)
    checkpoints_dir = Path(checkpoints_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_path) as f:
        results = json.load(f)
    
    # Generate plots
    plot_scaling_laws(results, output_dir / "scaling_laws.png")
    plot_training_curves(results, checkpoints_dir, output_dir / "training_curves.png")
    
    # Print table
    print("\n" + "="*60)
    print("RESULTS TABLE (Markdown)")
    print("="*60)
    table = create_results_table(results)
    print(table)
    
    # Save table to file
    (output_dir / "results_table.md").write_text(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze scaling experiment results")
    parser.add_argument("--results", type=str, default="scaling_results.json")
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="report")
    args = parser.parse_args()
    main(args.results, args.output_dir, args.checkpoints)
