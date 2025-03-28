"""
Visualization functions for next token prediction.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curves(ffnn_train_losses, ffnn_val_losses, ffnn_val_perplexities,
                         transformer_train_losses, transformer_val_losses, transformer_val_perplexities,
                         save_dir=None):
    """
    Plot training and validation curves for both models.
    
    Args:
        ffnn_train_losses (list): Training losses for feed-forward model
        ffnn_val_losses (list): Validation losses for feed-forward model
        ffnn_val_perplexities (list): Validation perplexities for feed-forward model
        transformer_train_losses (list): Training losses for transformer model
        transformer_val_losses (list): Validation losses for transformer model
        transformer_val_perplexities (list): Validation perplexities for transformer model
        save_dir (str, optional): Directory to save the plots. If None, plots are displayed only.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(ffnn_train_losses, 'b-', label='FFNN Train')
    plt.plot(ffnn_val_losses, 'b--', label='FFNN Val')
    plt.plot(transformer_train_losses, 'r-', label='Transformer Train')
    plt.plot(transformer_val_losses, 'r--', label='Transformer Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation perplexity
    plt.subplot(1, 2, 2)
    plt.plot(ffnn_val_perplexities, 'b-', label='FFNN')
    plt.plot(transformer_val_perplexities, 'r-', label='Transformer')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    
    plt.show()

def plot_accuracy_comparison(ffnn_accuracies, transformer_accuracies, k_values=[1, 3, 5], save_dir=None):
    """
    Plot accuracy comparison between the two models.
    
    Args:
        ffnn_accuracies (list): Accuracies for feed-forward model
        transformer_accuracies (list): Accuracies for transformer model
        k_values (list): K values for top-k accuracy
        save_dir (str, optional): Directory to save the plots. If None, plots are displayed only.
    """
    plt.figure(figsize=(10, 6))
    
    # Bar positions
    bar_width = 0.35
    index = np.arange(len(k_values))
    
    # Create bars
    plt.bar(index, ffnn_accuracies, bar_width, label='FFNN', color='blue', alpha=0.7)
    plt.bar(index + bar_width, transformer_accuracies, bar_width, label='Transformer', color='red', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Top-K')
    plt.ylabel('Accuracy')
    plt.title('Top-K Accuracy Comparison')
    plt.xticks(index + bar_width / 2, [f'Top-{k}' for k in k_values])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(ffnn_accuracies):
        plt.text(i - 0.05, v + 0.02, f'{v:.2f}', color='blue', fontweight='bold')
    
    for i, v in enumerate(transformer_accuracies):
        plt.text(i + bar_width - 0.05, v + 0.02, f'{v:.2f}', color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300)
    
    plt.show()

def visualize_attention_weights(attention_weights, tokens, save_dir=None):
    """
    Visualize attention weights from the transformer model.
    
    Args:
        attention_weights (Tensor): Attention weights matrix
        tokens (list): List of tokens corresponding to the weights
        save_dir (str, optional): Directory to save the plots. If None, plots are displayed only.
    """
    # Ensure attention_weights is a numpy array
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    
    # Set labels
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    
    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.title('Attention Weights')
    
    plt.tight_layout()
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'attention_weights.png'), dpi=300)
    
    plt.show()

def plot_model_comparison_table(metrics_dict, save_dir=None):
    """
    Plot a table of model comparison metrics.
    
    Args:
        metrics_dict (dict): Dictionary of metrics for both models
        save_dir (str, optional): Directory to save the plots. If None, plots are displayed only.
    """
    plt.figure(figsize=(12, 6))
    
    # Turn off axes
    plt.axis('off')
    plt.axis('tight')
    
    # Get metrics
    metrics = list(metrics_dict.keys())
    ffnn_values = [metrics_dict[metric]['ffnn'] for metric in metrics]
    transformer_values = [metrics_dict[metric]['transformer'] for metric in metrics]
    
    # Create data for the table
    table_data = []
    for i in range(len(metrics)):
        if isinstance(ffnn_values[i], float):
            table_data.append([metrics[i], f"{ffnn_values[i]:.4f}", f"{transformer_values[i]:.4f}"])
        else:
            table_data.append([metrics[i], str(ffnn_values[i]), str(transformer_values[i])])
    
    # Create table
    table = plt.table(cellText=table_data,
                     colLabels=['Metric', 'Feed-Forward', 'Transformer'],
                     colWidths=[0.4, 0.3, 0.3],
                     loc='center',
                     cellLoc='center')
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.title('Model Comparison', fontsize=16, pad=20)
    
    plt.tight_layout()
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'model_comparison_table.png'), dpi=300)
    
    plt.show() 