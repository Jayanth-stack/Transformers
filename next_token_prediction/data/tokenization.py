"""
Tokenization and vocabulary management for next token prediction.
"""

import torch
from collections import Counter
from transformers import AutoTokenizer
import sys
import os

# Add the parent directory to the path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VOCAB_SIZE, MIN_FREQ, SPECIAL_TOKENS, UNK_IDX

def initialize_tokenizer():
    """
    Initialize the tokenizer for the project.
    
    Returns:
        AutoTokenizer: The initialized tokenizer
    """
    return AutoTokenizer.from_pretrained("bert-base-uncased")

def count_tokens(dataset, tokenizer):
    """
    Count the total number of tokens in a dataset.
    
    Args:
        dataset: The dataset to count tokens from
        tokenizer: The tokenizer to use
        
    Returns:
        int: Total number of tokens
    """
    total_tokens = 0
    for example in dataset:
        tokens = tokenizer.tokenize(example['text'])
        total_tokens += len(tokens)
    return total_tokens

def build_vocabulary(dataset, tokenizer, vocab_size=VOCAB_SIZE, min_freq=MIN_FREQ):
    """
    Build vocabulary from dataset tokens.
    
    Args:
        dataset: The dataset to build vocabulary from
        tokenizer: The tokenizer to use
        vocab_size (int): Maximum vocabulary size
        min_freq (int): Minimum frequency for tokens to be included
        
    Returns:
        dict: Mapping from tokens to indices
    """
    token_counter = Counter()
    
    # Tokenize the dataset and count token frequencies
    for example in dataset:
        tokens = tokenizer.tokenize(example['text'])
        token_counter.update(tokens)
    
    # Filter out rare tokens
    vocab = [token for token, freq in token_counter.most_common() 
             if freq >= min_freq]
    
    # Keep the top `vocab_size` most frequent tokens
    vocab = vocab[:vocab_size]
    
    # Add special tokens
    vocab = SPECIAL_TOKENS + vocab
    
    # Create token to index mapping
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    
    return token_to_idx

def tokens_to_indices(tokens, token_to_idx):
    """
    Convert tokens to their corresponding indices.
    
    Args:
        tokens (list): List of tokens to convert
        token_to_idx (dict): Token to index mapping
        
    Returns:
        list: List of indices
    """
    return [token_to_idx.get(token, UNK_IDX) for token in tokens]

def create_training_examples(dataset, tokenizer, token_to_idx, seq_len=8):
    """
    Create training examples for next token prediction.
    
    Args:
        dataset: The dataset to create examples from
        tokenizer: The tokenizer to use
        token_to_idx (dict): Token to index mapping
        seq_len (int): Input sequence length
        
    Returns:
        tuple: (inputs, targets) where inputs are sequences and targets are next tokens
    """
    inputs = []
    targets = []
    
    for example in dataset:
        tokens = tokenizer.tokenize(example['text'])
        token_indices = tokens_to_indices(tokens, token_to_idx)
        
        for i in range(len(token_indices) - seq_len):
            inputs.append(token_indices[i:i+seq_len])
            targets.append(token_indices[i+seq_len])
    
    return inputs, targets 