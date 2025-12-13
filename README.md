# Scaling Laws for Music Language Models

A study of transformer vs. LSTM scaling on symbolic music (ABC notation). Includes data preprocessing, model training, scaling-law plots, and music generation experiments.

**Course:** CS-GY 6923 – Machine Learning, NYU Tandon School of Engineering

## Project Goals
- Build a full preprocessing pipeline for ABC music (Lakh MIDI → ABC)
- Train transformers of varying sizes (1M–100M params)
- Train RNNs/LSTMs of similar sizes
- Derive scaling laws (loss vs model size): L = a·N^(-α) + c
- Generate symbolic music samples

## Repository Structure
```
music-scaling-laws/
├── pipeline.py              # Main entry point for preprocessing
├── preprocess/              # Data preprocessing scripts
│   ├── convert_lmd_to_abc.py   # Step 1: MIDI → ABC conversion (uses midi2abc)
│   ├── verify_abc.py           # Step 2: Validate ABC files, remove corrupted
│   ├── clean_and_merge_abc.py  # Step 3: Clean and merge into single corpus
│   ├── build_dataset.py        # Step 4: Train/val/test split (98/1/1)
│   └── tokenize.py             # Step 5: Character-level tokenization
├── models/                  # Model architectures
│   ├── transformer.py          # GPT-style decoder-only transformer
│   └── lstm.py                 # LSTM language model
├── train/                   # Training scripts
│   ├── trainer.py              # Single model training
│   └── scaling_experiment.py   # Run full scaling experiments
├── eval/                    # Evaluation and analysis
│   ├── evaluate.py             # Compute perplexity metrics
│   ├── scaling_analysis.py     # Fit power laws, generate plots
│   └── generate.py             # Music sample generation
├── data/                    # Data directory (not in git)
│   ├── lmd_full/               # Lakh MIDI dataset
│   ├── raw/                    # Converted ABC files
│   └── processed/              # Tokenized train/val/test splits
├── checkpoints/             # Saved model checkpoints
├── samples/                 # Generated music samples
└── report/                  # Report figures and outputs
```

## Preprocessing Pipeline

The pipeline converts raw MIDI files to tokenized training data in 5 steps:

| Step | Script | Description |
|------|--------|-------------|
| 1. convert | `convert_lmd_to_abc.py` | Convert MIDI → ABC using `midi2abc` CLI |
| 2. verify | `verify_abc.py` | Validate ABC files, remove corrupted ones |
| 3. clean | `clean_and_merge_abc.py` | Clean and merge into single corpus |
| 4. split | `build_dataset.py` | Split into train/val/test (98%/1%/1%) |
| 5. tokenize | `tokenize.py` | Character-level tokenization to numpy |

### Pipeline Statistics (Lakh MIDI Dataset)
- **Input MIDI files:** 178,561
- **Converted ABC files:** 117,222 (97.7% success rate)
- **Valid after verification:** 116,986 (99.8%)
- **Final corpus:** 823M characters, 116,379 tunes
- **Vocabulary size:** 97 characters
- **Training tokens:** 807M

## Setup

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install midi2abc for MIDI conversion
brew install abcmidi
```

### 2. Download Lakh MIDI Dataset
```bash
# Download from https://colinraffel.com/projects/lmd/
# Extract to data/lmd_full/
```

### 3. Run Preprocessing Pipeline
```bash
# Run full pipeline (convert → verify → clean → split → tokenize)
python pipeline.py

# Or run individual steps
python pipeline.py --from convert --to convert  # Only conversion
python pipeline.py --from clean --to tokenize   # Skip conversion
python pipeline.py --force                       # Force re-run all steps
```

## Training

### Current Transformer Results (120M-token cap, cleaned corpus, B=128, T=256)

| Model | Params | Train loss | Val loss | Time (hh:mm:ss) |
|-------|--------|------------|----------|-----------------|
| tiny  | 0.8M   | 0.8175     | 0.7309   | 0:28:03         |
| small | 4.2M   | 0.5944     | 0.5524   | 1:22:02         |
| medium | — | training | in progress | — |

Notes:
- Preprocessing now strips `V:` voice markers, lyrics (`w:`/`W:`), inline chord annotations, and overly long titles; max piece length capped at 4096 chars.
- Token budget capped at 120M tokens per model; batch=128, block=256 from `config/constants.py`.

### Train a Single Model
```bash
# Train a small transformer
python train/trainer.py --model-type transformer --model-size small

# Train a medium LSTM
python train/trainer.py --model-type lstm --model-size medium
```

### Run Scaling Experiments
```bash
# Train all model sizes for both architectures
python train/scaling_experiment.py --model-type both

# Train only transformers
python train/scaling_experiment.py --model-type transformer --sizes tiny small medium large xl
```

## Evaluation

### Compute Perplexity
```bash
python eval/evaluate.py --checkpoints-dir checkpoints
```

### Generate Scaling Plots
```bash
python eval/scaling_analysis.py --results scaling_results.json
```

### Generate Music Samples
```bash
# Unconditional generation
python eval/generate.py --checkpoint checkpoints/transformer/transformer_large.pt --num-samples 10

# Conditional generation with prompt
python eval/generate.py --checkpoint checkpoints/transformer/transformer_large.pt --prompt "X:1\nT:My Song\nM:4/4\nK:C\n"

# Convert to MIDI
python eval/generate.py --checkpoint checkpoints/transformer/transformer_large.pt --to-midi
```

## Model Configurations

### Transformers
| Size   | Layers | Heads | Embed | ~Params |
|--------|--------|-------|-------|---------|
| tiny   | 4      | 4     | 128   | ~1M     |
| small  | 6      | 6     | 192   | ~5M     |
| medium | 8      | 8     | 384   | ~20M    |
| large  | 12     | 12    | 512   | ~50M    |
| xl     | 16     | 16    | 768   | ~100M+  |

### LSTMs
| Size   | Layers | Embed | Hidden | ~Params |
|--------|--------|-------|--------|---------|
| tiny   | 1      | 128   | 256    | ~1M     |
| small  | 2      | 192   | 384    | ~5M     |
| medium | 3      | 256   | 640    | ~20M    |
| large  | 4      | 384   | 896    | ~50M    |
| xl     | 4      | 512   | 1280   | ~100M+  |

## References
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)
- [GPT-2 Folk Music](https://gwern.net/gpt-2-music) (Gwern Branwen) - Inspiration for MIDI→ABC conversion
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Reference implementation
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) - Primary data source
- [ABC Notation](https://abcnotation.com) - Text-based music format