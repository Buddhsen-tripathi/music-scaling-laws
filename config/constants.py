"""
Central configuration constants for the Scaling Laws project.
"""

# Training hyperparameters
BATCH_SIZE = 128
BLOCK_SIZE = 256

# Token budget
MAX_TOKENS = 120_000_000 # 120M tokens per model

# Learning rate schedule
LEARNING_RATE = 3e-4
MIN_LR = 1e-5
WARMUP_ITERS = 100
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Checkpointing
CHECKPOINT_INTERVAL = 400
EVAL_INTERVAL = 500

# Data paths (relative to project root)
DATA_DIR = "data/processed"
CHECKPOINTS_DIR = "checkpoints"
SAMPLES_DIR = "samples"
RESULTS_FILE = "scaling_results.json"
