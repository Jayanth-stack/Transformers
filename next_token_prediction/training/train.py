"""
Training functions for next token prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

# Add parent directory to path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FFNN_LEARNING_RATE, TRANSFORMER_LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run the model on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f'Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s')
            start_time = time.time()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train(model, train_loader, val_loader, criterion, device, model_type="ffnn", 
          num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE):
    """
    Train the model with early stopping.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run the model on
        model_type (str): Type of model ('ffnn' or 'transformer')
        num_epochs (int): Maximum number of epochs to train for
        patience (int): Number of epochs to wait for improvement before stopping
        
    Returns:
        tuple: (trained model, training losses, validation losses, validation perplexities)
    """
    # Set learning rate based on model type
    if model_type.lower() == "ffnn":
        learning_rate = FFNN_LEARNING_RATE
    else:  # transformer
        learning_rate = TRANSFORMER_LEARNING_RATE
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_perplexities = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model = None
    counter = 0
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        from training.evaluate import evaluate_model
        val_loss, val_perplexity, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Perplexity: {val_perplexity:.2f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            
        # Early stopping
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, train_losses, val_losses, val_perplexities 