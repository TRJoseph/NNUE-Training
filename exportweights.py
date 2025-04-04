import torch
import numpy as np
import json

### This script is responsible for generating binary and json weight + bias files for use in my chess engine ###

def save_quantized_weights(pt_file, output_file="nnue_weightsQuantized.bin"):
    # Load PyTorch checkpoint
    model_data = torch.load(pt_file, map_location="cpu")

    # Extract weights and biases as numpy arrays (ensure they are in FP16 or INT8 format)
    weights = [
        model_data["ft.weight"].detach().transpose(0,1).numpy().astype(np.int16),
        model_data["l1.weight"].detach().transpose(0,1).numpy().astype(np.int8),
        model_data["l2.weight"].detach().transpose(0,1).numpy().astype(np.int8),
        model_data["l3.weight"].detach().transpose(0,1).numpy().astype(np.int8),
    ]
    
    biases = [
        model_data["ft.bias"].numpy().astype(np.int16),
        model_data["l1.bias"].numpy().astype(np.int32),
        model_data["l2.bias"].numpy().astype(np.int32),
        model_data["l3.bias"].numpy().astype(np.int32),
    ]

    # Save all weights and biases as a binary file
    with open(output_file, "wb") as f:
        for w in weights:
            w.tofile(f)
        for b in biases:
            b.tofile(f)

def save_quantized_weights_json(pt_file, output_file="nnue_weightsQuantized.json"):
    # Load PyTorch checkpoint
    model_data = torch.load(pt_file, map_location="cpu")

    # Convert tensors to lists of int16
    weights = {
        "ft.weight": model_data["ft.weight"].detach().transpose(0,1).numpy().astype(np.int16).tolist(),
        "l1.weight": model_data["l1.weight"].detach().transpose(0,1).numpy().astype(np.int8).tolist(),
        "l2.weight": model_data["l2.weight"].detach().transpose(0,1).numpy().astype(np.int8).tolist(),
        "l3.weight": model_data["l3.weight"].detach().transpose(0,1).numpy().astype(np.int8).tolist(),

        "ft.bias": model_data["ft.bias"].numpy().astype(np.int16).tolist(),
        "l1.bias": model_data["l1.bias"].numpy().astype(np.int32).tolist(),
        "l2.bias": model_data["l2.bias"].numpy().astype(np.int32).tolist(),
        "l3.bias": model_data["l3.bias"].numpy().astype(np.int32).tolist(),
    }

    # Save as JSON
    with open(output_file, "w") as f:
        json.dump(weights, f)


def save_normal_weights(pt_file, output_file="nnue_weightsNormal.bin"):
    # Load PyTorch checkpoint
    model_data = torch.load(pt_file, map_location="cpu")

    # Extract weights and biases as numpy arrays (in float32 format for normal weights)
    weights = [
        model_data["ft.weight"].detach().transpose(0, 1).numpy().astype(np.float32),
        model_data["l1.weight"].detach().transpose(0, 1).numpy().astype(np.float32),
        model_data["l2.weight"].detach().transpose(0, 1).numpy().astype(np.float32),
        model_data["l3.weight"].detach().transpose(0, 1).numpy().astype(np.float32),
    ]
    
    biases = [
        model_data["ft.bias"].detach().numpy().astype(np.float32),
        model_data["l1.bias"].detach().numpy().astype(np.float32),
        model_data["l2.bias"].detach().numpy().astype(np.float32),
        model_data["l3.bias"].detach().numpy().astype(np.float32),
    ]

    # Save all weights and biases as a binary file
    with open(output_file, "wb") as f:
        for w in weights:
            w.tofile(f)
        for b in biases:
            b.tofile(f)

def save_normal_weights_json(pt_file, output_file="nnue_weightsNormal.json"):
    # Load PyTorch checkpoint
    model_data = torch.load(pt_file, map_location="cpu")
    
    # Convert tensors to lists of float32 (normal weights)
    weights = {
        "ft.weight": model_data["ft.weight"].detach().transpose(0,1).numpy().astype(np.float32).tolist(),
        "l1.weight": model_data["l1.weight"].detach().transpose(0,1).numpy().astype(np.float32).tolist(),
        "l2.weight": model_data["l2.weight"].detach().transpose(0,1).numpy().astype(np.float32).tolist(),
        "l3.weight": model_data["l3.weight"].detach().transpose(0,1).numpy().astype(np.float32).tolist(),

        "ft.bias": model_data["ft.bias"].detach().numpy().astype(np.float32).tolist(),
        "l1.bias": model_data["l1.bias"].detach().numpy().astype(np.float32).tolist(),
        "l2.bias": model_data["l2.bias"].detach().numpy().astype(np.float32).tolist(),
        "l3.bias": model_data["l3.bias"].detach().numpy().astype(np.float32).tolist(),
    }

    # Save as JSON
    with open(output_file, "w") as f:
        json.dump(weights, f)

def count_zeros(lst):
    return len(lst)
save_quantized_weights("weights/nnue_weightsQuantized.pt")
save_quantized_weights_json("weights/nnue_weightsQuantized.pt")


save_normal_weights("weights/nnue_weightsNormal.pt")
save_normal_weights_json("weights/nnue_weightsNormal.pt")



