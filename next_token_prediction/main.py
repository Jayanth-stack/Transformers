"""
Main script for Next Token Prediction project.
"""

import torch
import torch.nn as nn
import argparse
import time
import os
import sys

# Import modules from the project
from data.tokenization import initialize_tokenizer, count_tokens, build_vocabulary
from data.data_loader import load_and_prepare_data, prepare_data_pipeline
from models.feed_forward import FeedForwardNN
from models.transformer import TransformerModel
from training.train import train
from training.evaluate import evaluate_model, measure_inference_time
from utils.visualization import (plot_training_curves, plot_accuracy_comparison, 
                               plot_model_comparison_table)
from config import (DEVICE, EMBEDDING_DIM, FFNN_HIDDEN_DIMS, TRANSFORMER_NHEAD, 
                  TRANSFORMER_NHID, TRANSFORMER_NLAYERS, TRANSFORMER_DROPOUT, 
                  NUM_EPOCHS)

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Next Token Prediction with Transformers')
    
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the outputs')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--skip_ffnn', action='store_true',
                        help='Skip training the feed-forward model')
    parser.add_argument('--skip_transformer', action='store_true',
                        help='Skip training the transformer model')
    
    return parser.parse_args()

def create_output_dir(output_dir):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir (str): Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    return models_dir, plots_dir

def main():
    """
    Main function to run the entire pipeline.
    """
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    models_dir, plots_dir = create_output_dir(args.output_dir)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_and_prepare_data()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = initialize_tokenizer()
    
    # Count tokens in each dataset
    print("Counting tokens...")
    train_tokens = count_tokens(train_dataset, tokenizer)
    val_tokens = count_tokens(val_dataset, tokenizer)
    test_tokens = count_tokens(test_dataset, tokenizer)
    
    print(f"Number of tokens in train dataset: {train_tokens}")
    print(f"Number of tokens in validation dataset: {val_tokens}")
    print(f"Number of tokens in test dataset: {test_tokens}")
    
    # Build vocabulary
    print("Building vocabulary...")
    token_to_idx = build_vocabulary(train_dataset, tokenizer)
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    vocab_size = len(token_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = prepare_data_pipeline(tokenizer, token_to_idx)
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Variables to store results
    metrics = {}
    
    # Initialize lists for training curves
    ffnn_train_losses = []
    ffnn_val_losses = []
    ffnn_val_perplexities = []
    transformer_train_losses = []
    transformer_val_losses = []
    transformer_val_perplexities = []
    
    # Train and evaluate feed-forward model
    if not args.skip_ffnn:
        print("\n===== Feed-Forward Model =====")
        
        # Initialize model
        ffnn_model = FeedForwardNN(vocab_size=vocab_size, 
                                  embedding_dim=EMBEDDING_DIM, 
                                  hidden_dims=FFNN_HIDDEN_DIMS).to(device)
        ffnn_model.init_weights()
        
        # Count parameters
        ffnn_params = ffnn_model.count_parameters()
        print(f"Feed-Forward model size: {ffnn_params} parameters")
        
        # Train model
        print("Training Feed-Forward model...")
        start_time = time.time()
        ffnn_model, ffnn_train_losses, ffnn_val_losses, ffnn_val_perplexities = train(
            ffnn_model, train_loader, val_loader, criterion, device, 
            model_type="ffnn", num_epochs=args.epochs
        )
        ffnn_train_time = time.time() - start_time
        
        # Save model
        torch.save(ffnn_model.state_dict(), os.path.join(models_dir, 'ffnn_model.pt'))
        
        # Evaluate on test set
        print("Evaluating Feed-Forward model on test set...")
        test_loss, test_perplexity, test_acc_1, test_acc_3, test_acc_5 = evaluate_model(
            ffnn_model, test_loader, criterion, device
        )
        
        # Measure inference time
        ffnn_inference_time = measure_inference_time(ffnn_model, test_loader, device)
        
        # Store metrics
        metrics['ffnn'] = {
            'test_loss': test_loss,
            'test_perplexity': test_perplexity,
            'test_acc_1': test_acc_1,
            'test_acc_3': test_acc_3,
            'test_acc_5': test_acc_5,
            'inference_time': ffnn_inference_time,
            'parameters': ffnn_params,
            'training_time': ffnn_train_time
        }
        
        print(f"Feed-Forward Test Loss: {test_loss:.4f}")
        print(f"Feed-Forward Test Perplexity: {test_perplexity:.2f}")
        print(f"Feed-Forward Test Acc@1: {test_acc_1:.4f}")
        print(f"Feed-Forward Test Acc@3: {test_acc_3:.4f}")
        print(f"Feed-Forward Test Acc@5: {test_acc_5:.4f}")
        print(f"Feed-Forward Avg. Inference Time: {ffnn_inference_time:.2f} ms per prediction")
    
    # Train and evaluate transformer model
    if not args.skip_transformer:
        print("\n===== Transformer Model =====")
        
        # Initialize model
        transformer_model = TransformerModel(
            vocab_size=vocab_size, 
            embedding_dim=EMBEDDING_DIM, 
            nhead=TRANSFORMER_NHEAD, 
            nhid=TRANSFORMER_NHID, 
            nlayers=TRANSFORMER_NLAYERS, 
            dropout=TRANSFORMER_DROPOUT
        ).to(device)
        
        # Count parameters
        transformer_params = transformer_model.count_parameters()
        print(f"Transformer model size: {transformer_params} parameters")
        
        # Train model
        print("Training Transformer model...")
        start_time = time.time()
        transformer_model, transformer_train_losses, transformer_val_losses, transformer_val_perplexities = train(
            transformer_model, train_loader, val_loader, criterion, device, 
            model_type="transformer", num_epochs=args.epochs
        )
        transformer_train_time = time.time() - start_time
        
        # Save model
        torch.save(transformer_model.state_dict(), os.path.join(models_dir, 'transformer_model.pt'))
        
        # Evaluate on test set
        print("Evaluating Transformer model on test set...")
        test_loss, test_perplexity, test_acc_1, test_acc_3, test_acc_5 = evaluate_model(
            transformer_model, test_loader, criterion, device
        )
        
        # Measure inference time
        transformer_inference_time = measure_inference_time(transformer_model, test_loader, device)
        
        # Store metrics
        metrics['transformer'] = {
            'test_loss': test_loss,
            'test_perplexity': test_perplexity,
            'test_acc_1': test_acc_1,
            'test_acc_3': test_acc_3,
            'test_acc_5': test_acc_5,
            'inference_time': transformer_inference_time,
            'parameters': transformer_params,
            'training_time': transformer_train_time
        }
        
        print(f"Transformer Test Loss: {test_loss:.4f}")
        print(f"Transformer Test Perplexity: {test_perplexity:.2f}")
        print(f"Transformer Test Acc@1: {test_acc_1:.4f}")
        print(f"Transformer Test Acc@3: {test_acc_3:.4f}")
        print(f"Transformer Test Acc@5: {test_acc_5:.4f}")
        print(f"Transformer Avg. Inference Time: {transformer_inference_time:.2f} ms per prediction")
    
    # Plot training curves if both models were trained
    if not args.skip_ffnn and not args.skip_transformer:
        print("\n===== Visualizing Results =====")
        
        # Plot training curves
        plot_training_curves(
            ffnn_train_losses, ffnn_val_losses, ffnn_val_perplexities,
            transformer_train_losses, transformer_val_losses, transformer_val_perplexities,
            save_dir=plots_dir
        )
        
        # Plot accuracy comparison
        plot_accuracy_comparison(
            [metrics['ffnn']['test_acc_1'], metrics['ffnn']['test_acc_3'], metrics['ffnn']['test_acc_5']],
            [metrics['transformer']['test_acc_1'], metrics['transformer']['test_acc_3'], metrics['transformer']['test_acc_5']],
            save_dir=plots_dir
        )
        
        # Create comparison table
        comparison_metrics = {
            'Test Perplexity': {'ffnn': metrics['ffnn']['test_perplexity'], 
                               'transformer': metrics['transformer']['test_perplexity']},
            'Test Accuracy@1': {'ffnn': metrics['ffnn']['test_acc_1'], 
                               'transformer': metrics['transformer']['test_acc_1']},
            'Test Accuracy@3': {'ffnn': metrics['ffnn']['test_acc_3'], 
                               'transformer': metrics['transformer']['test_acc_3']},
            'Test Accuracy@5': {'ffnn': metrics['ffnn']['test_acc_5'], 
                               'transformer': metrics['transformer']['test_acc_5']},
            'Inference Time (ms)': {'ffnn': metrics['ffnn']['inference_time'], 
                                   'transformer': metrics['transformer']['inference_time']},
            'Parameters': {'ffnn': metrics['ffnn']['parameters'], 
                          'transformer': metrics['transformer']['parameters']},
            'Training Time (s)': {'ffnn': metrics['ffnn']['training_time'], 
                                 'transformer': metrics['transformer']['training_time']}
        }
        
        plot_model_comparison_table(comparison_metrics, save_dir=plots_dir)
        
        # Print final comparison
        print("\n===== Model Comparison =====")
        print(f"Metric                  | Feed-Forward       | Transformer")
        print(f"------------------------|-------------------|------------------")
        print(f"Test Perplexity         | {metrics['ffnn']['test_perplexity']:.2f}              | {metrics['transformer']['test_perplexity']:.2f}")
        print(f"Test Accuracy@1         | {metrics['ffnn']['test_acc_1']:.4f}            | {metrics['transformer']['test_acc_1']:.4f}")
        print(f"Test Accuracy@3         | {metrics['ffnn']['test_acc_3']:.4f}            | {metrics['transformer']['test_acc_3']:.4f}")
        print(f"Test Accuracy@5         | {metrics['ffnn']['test_acc_5']:.4f}            | {metrics['transformer']['test_acc_5']:.4f}")
        print(f"Inference Time (ms)     | {metrics['ffnn']['inference_time']:.2f}              | {metrics['transformer']['inference_time']:.2f}")
        print(f"Parameters              | {metrics['ffnn']['parameters']}          | {metrics['transformer']['parameters']}")
        print(f"Training Time (s)       | {metrics['ffnn']['training_time']:.2f}            | {metrics['transformer']['training_time']:.2f}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 