"""
Music sample generation from trained models.
Supports both unconditional and conditional (prompted) generation.
"""
from __future__ import annotations

from pathlib import Path
import json
import argparse
import re

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.transformer import TransformerLM, TransformerConfig
from models.lstm import LSTMLM, LSTMConfig


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
    
    return model, model_config


def load_vocab(vocab_path: Path) -> tuple[dict, dict]:
    """Load vocabulary mappings."""
    with open(vocab_path) as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    return stoi, itos


def encode(text: str, stoi: dict) -> torch.Tensor:
    """Encode text to token IDs."""
    ids = [stoi.get(c, 0) for c in text]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def decode(ids: torch.Tensor, itos: dict) -> str:
    """Decode token IDs to text."""
    return "".join(itos.get(i, "?") for i in ids.squeeze().tolist())


def is_valid_abc(text: str) -> bool:
    """Check if text is valid ABC notation."""
    # Must have X: (index) or K: (key)
    has_header = bool(re.search(r"[XK]:", text))
    # Should have some note content
    has_notes = bool(re.search(r"[A-Ga-g]", text))
    return has_header and has_notes


def generate_samples(
    model,
    stoi: dict,
    itos: dict,
    num_samples: int = 10,
    max_length: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    prompt: str | None = None,
    device: str = "cpu",
) -> list[dict]:
    """
    Generate music samples.
    
    Returns:
        List of dicts with keys: text, is_valid, prompt
    """
    samples = []
    
    for i in range(num_samples):
        if prompt:
            idx = encode(prompt, stoi).to(device)
            gen_length = max_length - idx.size(1)
        else:
            # Start with X:1 (ABC tune index)
            start_text = "X:1\n"
            idx = encode(start_text, stoi).to(device)
            gen_length = max_length - idx.size(1)
        
        with torch.no_grad():
            output = model.generate(
                idx,
                max_new_tokens=gen_length,
                temperature=temperature,
                top_k=top_k,
            )
        
        text = decode(output, itos)
        valid = is_valid_abc(text)
        
        samples.append({
            "id": i + 1,
            "text": text,
            "is_valid": valid,
            "prompt": prompt,
            "temperature": temperature,
        })
        
        print(f"\n--- Sample {i+1} (valid={valid}) ---")
        print(text[:200] + "..." if len(text) > 200 else text)
    
    return samples


def abc_to_midi(abc_text: str, output_path: Path) -> bool:
    """Convert ABC to MIDI using music21."""
    try:
        from music21 import converter
        score = converter.parse(abc_text, format="abc")
        score.write("midi", fp=str(output_path))
        return True
    except Exception as e:
        print(f"MIDI conversion failed: {e}")
        return False


def main(checkpoint_path: str = None, output_dir: str = "samples", num_samples: int = 10,
         vocab_path: str = "data/processed/vocab.json", temperature: float = 0.8,
         top_k: int = 50, prompt: str = None, to_midi: bool = False):
    """Generate music samples. Can be called from pipeline or CLI."""
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and vocab
    model, config = load_model(Path(checkpoint_path), device)
    stoi, itos = load_vocab(Path(vocab_path))
    
    print(f"Model: {type(model).__name__}")
    print(f"Parameters: {model.num_params():,}")
    print(f"Vocab size: {len(stoi)}")
    
    # Generate samples
    samples = generate_samples(
        model, stoi, itos,
        num_samples=num_samples,
        max_length=512,
        temperature=temperature,
        top_k=top_k,
        prompt=prompt,
        device=device,
    )
    
    # Save samples
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    with open(output_dir / "samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    
    # Save individual ABC files
    valid_count = 0
    midi_success = 0
    for s in samples:
        abc_path = output_dir / f"sample_{s['id']}.abc"
        abc_path.write_text(s["text"])
        
        if s["is_valid"]:
            valid_count += 1
            if to_midi:
                midi_path = output_dir / f"sample_{s['id']}.mid"
                if abc_to_midi(s["text"], midi_path):
                    midi_success += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")
    print(f"Valid ABC: {valid_count} ({100*valid_count/len(samples):.1f}%)")
    if to_midi:
        print(f"MIDI conversions: {midi_success}/{valid_count} ({100*midi_success/max(1,valid_count):.1f}%)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music samples")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default="data/processed/vocab.json")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt text")
    parser.add_argument("--output-dir", type=str, default="samples")
    parser.add_argument("--to-midi", action="store_true", help="Convert valid samples to MIDI")
    args = parser.parse_args()
    main(args.checkpoint, args.output_dir, args.num_samples, args.vocab, 
         args.temperature, args.top_k, args.prompt, args.to_midi)
