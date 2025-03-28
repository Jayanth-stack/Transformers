"""
Evaluation functions for next token prediction models.
"""

import torch
import time
import sys
import os

# Add parent directory to path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K_VALUES

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run the model on
        
    Returns:
        tuple: (loss, perplexity, top-1 accuracy, top-3 accuracy, top-5 accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = {k: 0 for k in TOP_K_VALUES}
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Calculate top-k accuracy
            for k in TOP_K_VALUES:
                # Get top-k predictions
                _, top_k_indices = torch.topk(outputs, k, dim=1)
                
                # Check if target is in top-k predictions
                targets_expanded = targets.unsqueeze(1).expand_as(top_k_indices)
                correct_k = torch.eq(top_k_indices, targets_expanded).any(dim=1).sum().item()
                correct[k] += correct_k
            
            total += targets.size(0)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Calculate accuracies
    accuracies = {k: correct[k] / total for k in TOP_K_VALUES}
    
    return avg_loss, perplexity, accuracies[1], accuracies[3], accuracies[5]

def measure_inference_time(model, dataloader, device, num_samples=1000):
    """
    Measure the average inference time per prediction.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for test data
        device: Device to run the model on
        num_samples (int): Number of samples to use for measurement
        
    Returns:
        float: Average inference time per prediction in milliseconds
    """
    model.eval()
    total_time = 0.0
    sample_count = 0
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            # Only use the specified number of samples
            if sample_count >= num_samples:
                break
                
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # Measure inference time
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            
            # Calculate inference time for this batch
            batch_time = end_time - start_time
            
            # Add to total time
            total_time += batch_time
            sample_count += batch_size
    
    # Calculate average inference time per prediction in milliseconds
    avg_inference_time = (total_time / sample_count) * 1000
    
    return avg_inference_time 