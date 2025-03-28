"""
Positional encoding for transformer model.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions for transformer model.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Embedding dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Create a vector of shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create a vector for division
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor: Embeddings with positional encoding added
        """
        # x is of shape (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :] 