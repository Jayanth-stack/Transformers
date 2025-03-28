"""
Configuration parameters for next token prediction project.
This file centralizes all hyperparameters and settings.
"""

# Dataset parameters
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
VOCAB_SIZE = 10000
MIN_FREQ = 5
SEQ_LEN = 8
BATCH_SIZE = 64

# Model parameters
# Common parameters
EMBEDDING_DIM = 128

# Feed-forward model parameters
FFNN_HIDDEN_DIMS = [256, 128]
FFNN_LEARNING_RATE = 0.001

# Transformer model parameters
TRANSFORMER_NHEAD = 4
TRANSFORMER_NHID = 512
TRANSFORMER_NLAYERS = 2
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_LEARNING_RATE = 0.0001

# Training parameters
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
DEVICE = "cuda"  # or "cpu" if no GPU available

# Evaluation parameters
TOP_K_VALUES = [1, 3, 5]  # For top-k accuracy

# Special tokens
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3 