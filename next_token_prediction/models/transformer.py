"""
Transformer model for next token prediction.
"""

import torch
import torch.nn as nn
import math
import sys
import os

# Add parent directory to path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_DIM, TRANSFORMER_NHEAD, TRANSFORMER_NHID, TRANSFORMER_NLAYERS, TRANSFORMER_DROPOUT
from models.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    """
    Transformer model for next token prediction.
    """
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, nhead=TRANSFORMER_NHEAD, 
                 nhid=TRANSFORMER_NHID, nlayers=TRANSFORMER_NLAYERS, dropout=TRANSFORMER_DROPOUT):
        """
        Initialize the transformer model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the token embeddings
            nhead (int): Number of heads in the multi-head attention
            nhid (int): Dimension of the feedforward network model
            nlayers (int): Number of transformer encoder layers
            dropout (float): Dropout probability
        """
        super(TransformerModel, self).__init__()
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Create transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True  # Input is [batch, seq_len, ...]
        )
        
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        # Final linear layer to predict next token
        self.decoder = nn.Linear(embedding_dim, vocab_size)
        
        # Store embedding dimension for scaling
        self.embedding_dim = embedding_dim
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """
        Initialize the model weights.
        """
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        
    def forward(self, src):
        """
        Forward pass through the model.
        
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, vocab_size)
        """
        # Create source padding mask (1 for padding, 0 for actual tokens)
        src_key_padding_mask = (src == 0)
        
        # Embed and apply positional encoding: [batch_size, seq_len, embedding_dim]
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        
        # Apply transformer: [batch_size, seq_len, embedding_dim]
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Get the last token's representation for next token prediction
        output = output[:, -1, :]  # [batch_size, embedding_dim]
        
        # Project to vocabulary: [batch_size, vocab_size]
        output = self.decoder(output)
        
        return output
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 