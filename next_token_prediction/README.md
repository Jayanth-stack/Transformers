# Next Token Prediction with Transformers

This project implements and compares two approaches to next token prediction:
1. A baseline model using word embeddings and feed-forward layers
2. A multi-head Transformer-based model

## Project Structure

```
next_token_prediction/
│
├── data/                       # Dataset handling
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading and preparation
│   └── tokenization.py         # Tokenization and vocabulary building
│
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── feed_forward.py         # Baseline feed-forward model
│   ├── transformer.py          # Transformer-based model
│   └── positional_encoding.py  # Positional encoding implementation
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics
│   └── visualization.py        # Plotting and visualization
│
├── training/                   # Training functions
│   ├── __init__.py
│   ├── train.py                # Training loops
│   └── evaluate.py             # Evaluation functions
│
├── config.py                   # Configuration parameters
├── main.py                     # Main script to run the pipeline
└── requirements.txt            # Dependencies
```

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Code

You can run the entire pipeline using the main script:

```bash
python main.py
```

### Optional Arguments

- `--device`: Device to run on (cuda or cpu), default is "cuda" if available
- `--output_dir`: Directory to save the outputs, default is "output"
- `--epochs`: Number of training epochs, default is 5
- `--skip_ffnn`: Skip training the feed-forward model
- `--skip_transformer`: Skip training the transformer model

Example:

```bash
python main.py --device cpu --epochs 3 --output_dir my_results
```

Or to run only the transformer model:

```bash
python main.py --skip_ffnn
```

## Implementation Details

### Data Preprocessing

- Tokenization using BERT tokenizer
- Building vocabulary with special tokens
- Creating training examples for next token prediction

### Feed-Forward Model

- Embedding layer (dimension: 128)
- Two feed-forward layers (dimensions: [256, 128])
- ReLU activation
- Output layer for token prediction

### Transformer Model

- Embedding layer (dimension: 128)
- Positional encoding using sine/cosine functions
- Two transformer encoder layers with 4 attention heads
- Feed-forward dimension: 512
- Dropout rate: 0.1

### Evaluation Metrics

- Perplexity
- Top-1, Top-3, and Top-5 accuracy
- Inference time
- Model size (number of parameters)

## Results

The output of the main script includes:

1. Trained model files saved in the `output/models/` directory
2. Visualization plots saved in the `output/plots/` directory
3. Detailed comparison of model performances in the console output

## Customization

You can modify the hyperparameters and configurations in the `config.py` file to experiment with different settings. 