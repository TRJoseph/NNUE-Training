import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

### This file was most generated with Grok. It was prompted to help me do dataset analysis for hyperparameter tuning. ###

print("Starting script...")
start_time = time.time()

# Define the path to your CSV file
file_path = "Data/chessdata.csv"

# Load the dataset
print(f"Loading dataset from {file_path}...")
try:
    # Read the CSV file - assuming it has columns: "fen", "eval"
    df = pd.read_csv(file_path, names=["fen", "eval"], encoding='utf-8')
    print(f"Dataset loaded. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Process evaluations
print("Processing evaluations...")

# Function to clean and convert evaluation strings to float
def process_eval(eval_str):
    # Handle mate scores
    if "#" in str(eval_str):
        return -10000 if "-" in str(eval_str) else 10000
    
    # Remove any "+" signs and convert to float
    try:
        return float(str(eval_str).replace('+', ''))
    except ValueError:
        print(f"Warning: Could not convert '{eval_str}' to float. Using 0.")
        return 0

# Apply the processing to the eval column
df['raw_eval'] = df['eval'].apply(process_eval)

# Clip extreme values for visualization purposes
df['clipped_eval'] = np.clip(df['raw_eval'], -1000, 1000)

# Standard tanh normalization
def tanh_normalize(x, scale=400):
    """Normalize chess evaluation to [-1, 1] range using tanh scaling."""
    return np.tanh(x / scale)

# Custom sigmoid normalization (similar to previous script)
def sigmoid_normalize(x, k=0.004):
    """
    Sigmoid normalization with adjustable slope parameter k
    - Higher k values create sharper transition
    - Lower k values make a more gradual curve
    Maps to [-1, 1] range using the formula 2/(1+exp(-k*x))-1
    """
    return 2.0 / (1.0 + np.exp(-k * x)) - 1.0

# PyTorch-style sigmoid normalization (maps to [0, 1] range)
def torch_sigmoid(x, k=0.01):
    """
    PyTorch-style sigmoid normalization, which maps to [0, 1] range
    Formula: 1/(1+exp(-k*x))
    """
    return 1.0 / (1.0 + np.exp(-k * x))

# Normalized to [-1, 1] range using PyTorch-style sigmoid 
def torch_sigmoid_scaled(x, k=0.01):
    """
    PyTorch sigmoid scaled to [-1, 1] range
    Formula: 2/(1+exp(-k*x))-1 
    This is just a rescaled version of sigmoid
    """
    return 2.0 * torch_sigmoid(x, k) - 1.0

# Apply normalizations
df['tanh_norm'] = df['raw_eval'].apply(tanh_normalize)
df['sigmoid_norm'] = df['raw_eval'].apply(sigmoid_normalize)
df['torch_sigmoid'] = df['raw_eval'].apply(torch_sigmoid)
df['torch_sigmoid_scaled'] = df['raw_eval'].apply(torch_sigmoid_scaled)

# Different parameter values for torch.sigmoid()
k_values = [0.001, 0.005, 0.01, 0.02]
for k in k_values:
    df[f'torch_sigmoid_k{k}'] = df['raw_eval'].apply(lambda x: torch_sigmoid(x, k=k))
    df[f'torch_sigmoid_scaled_k{k}'] = df['raw_eval'].apply(lambda x: torch_sigmoid_scaled(x, k=k))

# Create visualizations
plt.figure(figsize=(12, 8))

# Plot torch.sigmoid() histogram (original [0,1] range)
plt.subplot(2, 1, 1)
plt.hist(df['torch_sigmoid'], bins=50, alpha=0.7, color='blue')
plt.title('PyTorch Sigmoid Normalized Chess Evaluations [0, 1]')
plt.xlabel('Normalized Evaluation (torch.sigmoid)')
plt.ylabel('Frequency')
plt.xlim(0, 1)
plt.axvline(0.5, color='red', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

# Plot scaled torch.sigmoid() histogram (scaled to [-1,1])
plt.subplot(2, 1, 2)
plt.hist(df['torch_sigmoid_scaled'], bins=50, alpha=0.7, color='purple')
plt.title('PyTorch Sigmoid Normalized Chess Evaluations (Scaled to [-1, 1])')
plt.xlabel('Normalized Evaluation (scaled torch.sigmoid)')
plt.ylabel('Frequency')
plt.xlim(-1, 1)
plt.axvline(0, color='red', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('torch_sigmoid_normalizations.png', dpi=300)
print("PyTorch sigmoid visualization saved as 'torch_sigmoid_normalizations.png'")

# Compare different k values for torch.sigmoid() scaled version
plt.figure(figsize=(15, 10))
for i, k in enumerate(k_values):
    plt.subplot(len(k_values), 1, i+1)
    plt.hist(df[f'torch_sigmoid_scaled_k{k}'], bins=50, alpha=0.7, 
             color=plt.cm.viridis(i/len(k_values)))
    plt.title(f'Scaled PyTorch Sigmoid Normalization (k={k})')
    plt.xlabel('Normalized Evaluation')
    plt.ylabel('Frequency')
    plt.xlim(-1, 1)
    plt.axvline(0, color='red', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('torch_sigmoid_parameter_comparison.png', dpi=300)
print("Parameter comparison saved as 'torch_sigmoid_parameter_comparison.png'")

# Create a figure showing the mapping functions
plt.figure(figsize=(12, 6))
x = np.linspace(-1000, 1000, 1000)
plt.plot(x, tanh_normalize(x), 'g-', linewidth=2, label='tanh(x/400)')
plt.plot(x, sigmoid_normalize(x), 'r-', linewidth=2, label='custom sigmoid 2/(1+exp(-0.004x))-1')
plt.plot(x, torch_sigmoid(x), 'b-', linewidth=2, label='torch.sigmoid(0.01x)')
plt.plot(x, torch_sigmoid_scaled(x), 'm-', linewidth=2, label='scaled torch.sigmoid ([-1,1])')

plt.title('Normalization Functions Comparison')
plt.xlabel('Raw Evaluation (centipawns)')
plt.ylabel('Normalized Value')
plt.xlim(-1000, 1000)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('normalization_functions.png', dpi=300)
print("Normalization functions comparison saved as 'normalization_functions.png'")

print(f"\nScript completed in {time.time() - start_time:.2f} seconds")