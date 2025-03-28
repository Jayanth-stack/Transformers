"""
Feed-forward neural network for next token prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_DIM, FFNN_HIDDEN_DIMS

class FeedForwardNN(nn.Module):
    """
    Feed-forward neural network for next token prediction.
    """
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dims=FFNN_HIDDEN_DIMS):
        """
        Initialize the feed-forward model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the token embeddings
            hidden_dims (list): Dimensions of the hidden layers
        """
        super(FeedForwardNN, self).__init__()
        
        # Embedding layer to convert token indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # First hidden layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dims[0])
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # Output layer
        self.output = nn.Linear(hidden_dims[1], vocab_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, vocab_size)
        """
        # Embed the input tokens: [batch_size, seq_len, embedding_dim]
        x = self.embedding(x)
        
        # Combine embeddings by averaging: [batch_size, embedding_dim]
        x = x.mean(dim=1)
        
        # Pass through feed-forward layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        
        return x  # [batch_size, vocab_size]
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0) 