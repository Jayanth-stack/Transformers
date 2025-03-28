"""
Metrics for evaluating next token prediction models.
"""

import torch
import numpy as np
from collections import defaultdict

def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss (float): Cross-entropy loss
        
    Returns:
        float: Perplexity
    """
    return torch.exp(torch.tensor(loss)).item()

def calculate_top_k_accuracy(outputs, targets, k):
    """
    Calculate top-k accuracy.
    
    Args:
        outputs (Tensor): Model predictions (logits)
        targets (Tensor): Ground truth labels
        k (int): K value for top-k accuracy
        
    Returns:
        float: Top-k accuracy
    """
    # Get top-k predictions
    _, top_k_indices = torch.topk(outputs, k, dim=1)
    
    # Expand targets to match top_k_indices shape
    targets_expanded = targets.unsqueeze(1).expand_as(top_k_indices)
    
    # Check if target is in top-k predictions
    correct = torch.eq(top_k_indices, targets_expanded).any(dim=1).sum().item()
    
    # Calculate accuracy
    accuracy = correct / targets.size(0)
    
    return accuracy

def calculate_metrics_by_frequency(model, dataloader, token_to_idx, idx_to_token, token_freqs, device):
    """
    Calculate metrics grouped by token frequency.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        token_to_idx (dict): Mapping from tokens to indices
        idx_to_token (dict): Mapping from indices to tokens
        token_freqs (dict): Token frequencies
        device: Device to run the model on
        
    Returns:
        dict: Metrics by frequency group
    """
    # Define frequency groups
    freq_groups = {
        'rare': (1, 10),      # 1-10 occurrences
        'uncommon': (11, 100),  # 11-100 occurrences
        'common': (101, 1000),  # 101-1000 occurrences
        'very_common': (1001, float('inf'))  # >1000 occurrences
    }
    
    # Initialize counters
    correct_by_group = defaultdict(int)
    total_by_group = defaultdict(int)
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            # Calculate metrics by frequency
            for i, target in enumerate(targets):
                # Get token
                token_idx = target.item()
                token = idx_to_token.get(token_idx, '<UNK>')
                
                # Get token frequency
                freq = token_freqs.get(token, 0)
                
                # Determine frequency group
                group = None
                for group_name, (min_freq, max_freq) in freq_groups.items():
                    if min_freq <= freq <= max_freq:
                        group = group_name
                        break
                
                if group is None:
                    continue
                
                # Increment counters
                total_by_group[group] += 1
                if predictions[i].item() == token_idx:
                    correct_by_group[group] += 1
    
    # Calculate accuracies by group
    accuracies = {}
    for group in freq_groups:
        if total_by_group[group] > 0:
            accuracies[group] = correct_by_group[group] / total_by_group[group]
        else:
            accuracies[group] = 0.0
    
    return accuracies 