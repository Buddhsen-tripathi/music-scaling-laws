# Scaling Laws for Music Language Models

A study of Transformer vs. LSTM scaling on symbolic music (ABC notation). Includes data preprocessing, model training, scaling-law plots, and music generation experiments.

**Course:** CS-GY 6923 – Machine Learning  
**Institution:** New York University

**Project Details:** [`cs_gy_6923_project.pdf`](cs_gy_6923_project.pdf)

**Report:** [`report/report.md`](report/report.md) (and generated [`report/report.pdf`](report/report.pdf))

## Project Goals
- Build a full preprocessing pipeline for ABC music (Lakh MIDI → ABC)
- Train transformers of varying sizes (1M–100M params)
- Train RNNs/LSTMs of similar sizes
- Derive scaling laws (loss vs model size): L = a·N^(-α) + c
- Generate symbolic music samples

## Quickstart

### 1) Create environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Install MIDI → ABC converter
This project uses the `midi2abc` CLI from the `abcmidi` package.

```bash
brew install abcmidi
```

### 3) Put the Lakh MIDI Dataset in the expected location
Download from https://colinraffel.com/projects/lmd/ and extract into:

```text
data/lmd_full/
```

### 4) Run the end-to-end pipeline
```bash
python pipeline.py
```

This runs:
`convert → verify → clean → split → tokenize → train → evaluate → generate`.

`pipeline.py` is a single end-to-end orchestration script: one command runs the full workflow (and it can resume/skip completed steps).

## Setup Notes

- The pipeline is designed to be **resumable**.
- Use `--force` to re-run steps even if outputs already exist.
- By default, conversion is capped (see `PipelineConfig.target_abc_files` in `pipeline.py`).

## Repository Structure
```
music-scaling-laws/
├── pipeline.py              # Main orchestration script
├── preprocess/              # Data preprocessing scripts
│   ├── convert_lmd_to_abc.py   
│   ├── verify_abc.py           
│   ├── clean_and_merge_abc.py  
│   ├── plot_length_hist.py     
│   ├── build_dataset.py        
│   └── tokenize.py           
├── models/                  # Model architectures
│   ├── transformer.py          
│   └── lstm.py                
├── train/                   # Training scripts
│   ├── trainer.py              
│   └── scaling_experiment.py  
├── eval/                    # Evaluation and analysis
│   ├── evaluate.py             
│   ├── scaling_analysis.py     
│   └── generate.py             
├── data/                    # Data directory (not in git)
│   ├── lmd_full/               
│   ├── raw/                    
│   └── processed/              
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

### Run preprocessing only
```bash
python pipeline.py --to tokenize
```

### Run specific steps / resume
```bash
python pipeline.py --from convert --to verify
python pipeline.py --from clean --to tokenize
python pipeline.py --from train --to train --model-type transformer
python pipeline.py --force
```

## Training

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

### Train via pipeline
```bash
python pipeline.py --from train --to train --model-type both
```

## Evaluation

### Compute Perplexity
```bash
python eval/evaluate.py --checkpoints-dir checkpoints --data-dir data/processed --output report/eval_results.json
```

### Generate Scaling Plots
```bash
python eval/scaling_analysis.py --results scaling_results.json --output-dir report --checkpoints checkpoints
```

### Generate Music Samples
```bash
# Unconditional generation
python eval/generate.py --checkpoint checkpoints/transformer/transformer_xl_final.pt --num-samples 10 --output-dir samples

# Conditional generation with prompt
python eval/generate.py --checkpoint checkpoints/transformer/transformer_xl_final.pt --prompt "X:1\nT:My Song\nM:4/4\nK:C\n" --output-dir samples

# Convert to MIDI
python eval/generate.py --checkpoint checkpoints/transformer/transformer_xl_final.pt --to-midi --output-dir samples
```

## Outputs

- `data/raw/`: converted ABC files
- `data/processed/`: merged corpus + splits + tokenized arrays (`train.npy`, `val.npy`, `test.npy`, `vocab.json`)
- `checkpoints/<model_type>/`: training checkpoints (e.g. `transformer_xl_final.pt`)
- `scaling_results.json`: consolidated training results
- `report/`:
  - plots: `scaling_laws.png`, `training_curves.png`
  - tables: `results_table.md`
  - eval: `eval_results.json`
- `samples/`: generated `.abc`, optional `.mid`, plus `samples.json`

## Model Configurations

### Transformers
| Size | Layers | Heads | d_model | Parameters |
|--------|--------|-------|-------|---------|
| tiny | 4 | 4 | 128 | ~1M |
| small | 6 | 6 | 240 | ~5M |
| medium | 8 | 8 | 440 | ~20M |
| large | 12 | 10 | 620 | ~50M |
| xl | 14 | 12 | 768 | ~100M |

### LSTMs
| Size | Layers | Hidden | Parameters |
|--------|--------|--------|---------|
| tiny | 1 | 384 | ~1M |
| small | 2 | 512 | ~5M |
| medium | 3 | 1024 | ~20M |
| large | 4 | 1280 | ~50M |
| xl | 4 | 1792 | ~100M |

## References
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)
- [GPT-2 Folk Music](https://gwern.net/gpt-2-music) (Gwern Branwen) - Inspiration for MIDI→ABC conversion
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Reference implementation
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) - Primary data source
- [ABC Notation](https://abcnotation.com) - Text-based music format