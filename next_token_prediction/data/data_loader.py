"""
Data loading and preparation for next token prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sys
import os

# Add the parent directory to the path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_NAME, DATASET_CONFIG, BATCH_SIZE, SEQ_LEN
from data.tokenization import create_training_examples

class SequenceDataset(Dataset):
    """
    PyTorch Dataset for next token prediction task.
    """
    def __init__(self, inputs, targets):
        """
        Initialize the dataset.
        
        Args:
            inputs (list): List of input sequences
            targets (list): List of target tokens
        """
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        """
        Get the total number of examples.
        
        Returns:
            int: Dataset size
        """
        return len(self.inputs)
    
    def __getitem__(self, idx):
        """
        Get a specific example.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (input sequence, target token)
        """
        return self.inputs[idx], self.targets[idx]

def load_and_prepare_data():
    """
    Load WikiText-2 dataset and split into train, validation, and test sets.
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_inputs, train_targets, val_inputs, val_targets, 
                       test_inputs, test_targets, batch_size=BATCH_SIZE):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        train_inputs (list): Training input sequences
        train_targets (list): Training target tokens
        val_inputs (list): Validation input sequences
        val_targets (list): Validation target tokens
        test_inputs (list): Test input sequences
        test_targets (list): Test target tokens
        batch_size (int): Batch size
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create dataset objects
    train_data = SequenceDataset(train_inputs, train_targets)
    val_data = SequenceDataset(val_inputs, val_targets)
    test_data = SequenceDataset(test_inputs, test_targets)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def prepare_data_pipeline(tokenizer, token_to_idx, seq_len=SEQ_LEN):
    """
    Full data preparation pipeline from loading to creating dataloaders.
    
    Args:
        tokenizer: The tokenizer to use
        token_to_idx (dict): Token to index mapping
        seq_len (int): Input sequence length
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_and_prepare_data()
    
    # Create training examples
    train_inputs, train_targets = create_training_examples(
        train_dataset, tokenizer, token_to_idx, seq_len)
    val_inputs, val_targets = create_training_examples(
        val_dataset, tokenizer, token_to_idx, seq_len)
    test_inputs, test_targets = create_training_examples(
        test_dataset, tokenizer, token_to_idx, seq_len)
    
    # Create dataloaders
    return create_dataloaders(
        train_inputs, train_targets, 
        val_inputs, val_targets,
        test_inputs, test_targets
    ) 