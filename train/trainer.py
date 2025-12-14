"""
Training script for music language models.
Supports both Transformer and LSTM architectures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.transformer import TransformerLM, TransformerConfig, TRANSFORMER_CONFIGS
from models.lstm import LSTMLM, LSTMConfig, LSTM_CONFIGS


# Import constants for defaults
from config.constants import (
    BATCH_SIZE, BLOCK_SIZE, MAX_TOKENS, LEARNING_RATE, MIN_LR,
    WARMUP_ITERS, WEIGHT_DECAY, GRAD_CLIP, CHECKPOINT_INTERVAL,
    EVAL_INTERVAL, DATA_DIR, CHECKPOINTS_DIR
)


@dataclass
class TrainConfig:
    # Data
    data_dir: Path = Path(DATA_DIR)
    
    # Model
    model_type: Literal["transformer", "lstm"] = "transformer"
    model_size: str = "small"  # tiny, small, medium, large, xl
    
    # Training (defaults from config/constants.py)
    batch_size: int = BATCH_SIZE
    block_size: int = BLOCK_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    max_iters: int | None = None
    max_tokens: int = MAX_TOKENS
    warmup_iters: int = WARMUP_ITERS
    lr_decay_iters: int | None = None
    min_lr: float = MIN_LR
    grad_clip: float = GRAD_CLIP
    
    # Logging & Checkpoints
    eval_interval: int = EVAL_INTERVAL
    log_interval: int = 100
    save_dir: Path = Path(CHECKPOINTS_DIR)
    checkpoint_interval: int = CHECKPOINT_INTERVAL
    resume: bool = True  # Resume from checkpoint if exists
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    compile_model: bool = False 


class MusicDataset(Dataset):
    def __init__(self, data_path: Path, block_size: int):
        self.data = np.load(data_path, mmap_mode="r")
        self.block_size = block_size
        self.max_start = len(self.data) - block_size - 1

    def __len__(self):
        return max(1, self.max_start)

    def __getitem__(self, idx):
        i = np.random.randint(0, self.max_start)

        x = torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64, copy=False))
        y = torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64, copy=False))
        return x, y


def get_lr(it: int, config: TrainConfig) -> float:
    lr_decay_iters = config.lr_decay_iters or config.max_iters or 10000
    
    # Warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # After decay
    if it > lr_decay_iters:
        return config.min_lr
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model: nn.Module, dataloader: DataLoader, device: str, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def train(config: TrainConfig, force: bool = False) -> dict:
    """
    Train a model and return training statistics.
    
    Args:
        config: Training configuration
        force: If True, retrain even if final checkpoint exists
    
    Returns:
        dict with keys: model_type, model_size, num_params, train_loss, val_loss,
                       train_time, tokens_per_sec, gpu_memory_mb
    """
    print(f"\n{'='*60}")
    print(f"Training {config.model_type} ({config.model_size})")
    print(f"{'='*60}")
    
    # Load vocab
    vocab_path = config.data_dir / "vocab.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]
    print(f"Vocab size: {vocab_size}")
    
    # Create model
    if config.model_type == "transformer":
        model_config = TRANSFORMER_CONFIGS[config.model_size]
        model_config.vocab_size = vocab_size
        model_config.block_size = config.block_size
        model = TransformerLM(model_config)
    else:
        model_config = LSTM_CONFIGS[config.model_size]
        model_config.vocab_size = vocab_size
        model_config.block_size = config.block_size
        model = LSTMLM(model_config)
    
    model = model.to(config.device)
    num_params = model.num_params()
    print(f"Model parameters: {num_params:,}")
    
    if config.compile_model and hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Load data
    train_dataset = MusicDataset(config.data_dir / "train.npy", config.block_size)
    val_dataset = MusicDataset(config.data_dir / "val.npy", config.block_size)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Calculate iterations based on token budget (capped at max_tokens)
    tokens_per_iter = config.batch_size * config.block_size
    total_tokens = len(train_dataset.data)
    iters_per_epoch = total_tokens // tokens_per_iter
    iters_for_budget = config.max_tokens // tokens_per_iter
    
    if config.max_iters is None:
        config.max_tokens = total_tokens
        config.max_iters = iters_per_epoch
    if config.lr_decay_iters is None:
        config.lr_decay_iters = config.max_iters
    
    print(f"Device: {config.device}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Block size: {config.block_size}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Token budget: {config.max_tokens:,}")
    print(f"Tokens per iter: {tokens_per_iter:,}")
    print(f"Max iterations: {config.max_iters:,} ({config.max_iters * tokens_per_iter / 1e6:.1f}M tokens)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Checkpoint paths: _last for resuming, _final for finished artifact
    config.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = config.save_dir / f"{config.model_type}_{config.model_size}_last.pt"
    ckpt_final = config.save_dir / f"{config.model_type}_{config.model_size}_final.pt"
    
    train_losses = []
    best_val_loss = float("inf")
    iter_num = 0
    tokens_processed = 0
    
    # Check if final checkpoint exists (training complete) - skip unless force
    if ckpt_final.exists() and not force:
        print(f"[SKIP] Final checkpoint exists: {ckpt_final}")
        print(f"       Use --force or delete checkpoint to retrain")
        import __main__ as _main
        if not hasattr(_main, "TrainConfig"):
            setattr(_main, "TrainConfig", TrainConfig)
        ckpt = torch.load(ckpt_final, map_location=config.device, weights_only=False)
        return {
            "model_type": config.model_type,
            "model_size": config.model_size,
            "num_params": num_params,
            "train_loss": ckpt.get("final_train_loss", 0),
            "val_loss": ckpt.get("final_val_loss", 0),
            "train_time_sec": ckpt.get("train_time_sec", 0),
            "tokens_per_sec": ckpt.get("tokens_per_sec", 0),
            "gpu_memory_mb": ckpt.get("gpu_memory_mb", 0),
        }
    
    # If force, remove existing checkpoints
    if force and ckpt_final.exists():
        print(f"[FORCE] Removing existing checkpoint: {ckpt_final}")
        ckpt_final.unlink()
    if force and ckpt_last.exists():
        ckpt_last.unlink()
    
    # Resume from _last checkpoint if exists
    if config.resume and ckpt_last.exists():
        print(f"Resuming from checkpoint: {ckpt_last}")
        import __main__ as _main
        if not hasattr(_main, "TrainConfig"):
            setattr(_main, "TrainConfig", TrainConfig)
        ckpt = torch.load(ckpt_last, map_location=config.device, weights_only=False)
        
        # Validate config compatibility
        old_cfg = ckpt.get("train_config", None)
        if old_cfg is not None:
            if hasattr(old_cfg, "block_size") and old_cfg.block_size != config.block_size:
                raise ValueError(f"block_size mismatch: checkpoint={old_cfg.block_size}, current={config.block_size}")
            if hasattr(old_cfg, "batch_size") and old_cfg.batch_size != config.batch_size:
                raise ValueError(f"batch_size mismatch: checkpoint={old_cfg.batch_size}, current={config.batch_size}")
        
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        iter_num = ckpt.get("iter_num", 0)
        tokens_processed = ckpt.get("tokens_processed", 0)
        train_losses = ckpt.get("train_losses", [])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at iter {iter_num}, best_val_loss={best_val_loss:.4f}")
        
        if iter_num >= config.max_iters:
            print("Training already complete!")
            return {
                "model_type": config.model_type,
                "model_size": config.model_size,
                "num_params": num_params,
                "train_loss": ckpt.get("final_train_loss", train_losses[-1] if train_losses else 0),
                "val_loss": ckpt.get("final_val_loss", best_val_loss),
                "train_time_sec": ckpt.get("train_time_sec", 0),
                "tokens_per_sec": ckpt.get("tokens_per_sec", 0),
                "gpu_memory_mb": ckpt.get("gpu_memory_mb", 0),
            }
    
    # Training loop
    model.train()
    start_time = time.time()
    
    pbar = tqdm(total=config.max_iters, initial=iter_num, desc="Training")
    
    for epoch in range(100):  # Max epochs (will break on max_iters)
        for x, y in train_loader:
            if iter_num >= config.max_iters:
                break
            
            # Update learning rate
            lr = get_lr(iter_num, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            x, y = x.to(config.device), y.to(config.device)
            
            # Forward pass
            _, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            train_losses.append(loss.item())
            tokens_processed += tokens_per_iter
            iter_num += 1
            pbar.update(1)
            
            # Logging (every ~1% of training)
            log_every = max(1, config.max_iters // 100)
            if iter_num % log_every == 0:
                avg_loss = sum(train_losses[-log_every:]) / len(train_losses[-log_every:])
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
            
            # Evaluation & Checkpoint
            if iter_num % config.eval_interval == 0 or iter_num == config.max_iters:
                val_loss = estimate_loss(model, val_loader, config.device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                # Safe train loss calculation (avg_loss may not exist yet)
                train_loss_print = sum(train_losses[-100:]) / max(1, min(100, len(train_losses)))
                print(f"\n[iter {iter_num}] train_loss={train_loss_print:.4f} val_loss={val_loss:.4f}")
            
            # Save checkpoint periodically to _last
            if iter_num % config.checkpoint_interval == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config,
                    "train_config": config,
                    "iter_num": iter_num,
                    "tokens_processed": tokens_processed,
                    "train_losses": train_losses,
                    "best_val_loss": best_val_loss,
                }, ckpt_last)
        
        if iter_num >= config.max_iters:
            break
    
    pbar.close()
    train_time = time.time() - start_time
    
    # Final evaluation
    final_train_loss = sum(train_losses[-100:]) / min(100, len(train_losses))
    final_val_loss = estimate_loss(model, val_loader, config.device, max_batches=100)
    
    # GPU memory
    gpu_memory_mb = 0
    if config.device == "cuda":
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    elif config.device == "mps":
        # MPS doesn't have easy memory tracking
        gpu_memory_mb = -1
    
    # Save final checkpoint to _final (separate from _last)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "train_config": config,
        "iter_num": iter_num,
        "tokens_processed": tokens_processed,
        "train_losses": train_losses,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "train_time_sec": train_time,
        "tokens_per_sec": tokens_processed / train_time,
        "gpu_memory_mb": gpu_memory_mb,
    }, ckpt_final)
    print(f"Saved final checkpoint: {ckpt_final}")
    
    # Clean up _last checkpoint after successful completion
    if ckpt_last.exists():
        ckpt_last.unlink()
        print(f"Removed resume checkpoint: {ckpt_last}")
    
    stats = {
        "model_type": config.model_type,
        "model_size": config.model_size,
        "num_params": num_params,
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "train_time_sec": train_time,
        "tokens_per_sec": tokens_processed / train_time,
        "gpu_memory_mb": gpu_memory_mb,
    }
    
    print(f"\nResults:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train music language model")
    parser.add_argument("--model-type", choices=["transformer", "lstm"], default="transformer")
    parser.add_argument("--model-size", choices=["tiny", "small", "medium", "large", "xl"], default="small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    
    config = TrainConfig(
        model_type=args.model_type,
        model_size=args.model_size,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.lr,
        max_iters=args.max_iters,
        data_dir=Path(args.data_dir),
        save_dir=Path(args.save_dir),
        compile_model=args.compile,
    )
    
    train(config)


if __name__ == "__main__":
    main()