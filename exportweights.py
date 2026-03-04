"""
exportweights.py — export trained NNUE weights to binary and JSON formats.
Weights are transposed [in, out] for column-major accumulator access.
"""

import torch
import numpy as np
import json
import os


def _load(pt_file: str) -> dict:
    return torch.load(pt_file, map_location="cpu")



def save_quantized_binary(pt_file: str, output_file: str = "weights/nn_weightsQuantized.nnue"):
    """Save quantized weights as a packed binary file."""
    d = _load(pt_file)

    weights = [
        d["ft.weight"].numpy().astype(np.int16).T,
        d["l1.weight"].numpy().astype(np.int8).T,
        d["l2.weight"].numpy().astype(np.int8).T,
        d["l3.weight"].numpy().astype(np.int8).T,
    ]
    biases = [
        d["ft.bias"].numpy().astype(np.int16),   # int16: matches accumulator dtype for SIMD
        d["l1.bias"].numpy().astype(np.int32),
        d["l2.bias"].numpy().astype(np.int32),
        d["l3.bias"].numpy().astype(np.int32),
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        for arr in weights + biases:
            arr.tofile(f)

    print(f"Saved quantized binary → {output_file}")


def save_quantized_json(pt_file: str, output_file: str = "weights/nnue_weightsQuantized.json"):
    """Save quantized weights as JSON (useful for debugging / JS engines)."""
    d = _load(pt_file)

    data = {
        "ft.weight": d["ft.weight"].numpy().astype(np.int16).T.tolist(),
        "l1.weight": d["l1.weight"].numpy().astype(np.int8).T.tolist(),
        "l2.weight": d["l2.weight"].numpy().astype(np.int8).T.tolist(),
        "l3.weight": d["l3.weight"].numpy().astype(np.int8).T.tolist(),
        "ft.bias":   d["ft.bias"].numpy().astype(np.int16).tolist(),
        "l1.bias":   d["l1.bias"].numpy().astype(np.int32).tolist(),
        "l2.bias":   d["l2.bias"].numpy().astype(np.int32).tolist(),
        "l3.bias":   d["l3.bias"].numpy().astype(np.int32).tolist(),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f)

    print(f"Saved quantized JSON → {output_file}")



def save_normal_binary(pt_file: str, output_file: str = "weights/nn_weightsNormal.nnue"):
    """Save float32 weights as a packed binary file."""
    d = _load(pt_file)

    weights = [
        d["ft.weight"].numpy().astype(np.float32).T,
        d["l1.weight"].numpy().astype(np.float32).T,
        d["l2.weight"].numpy().astype(np.float32).T,
        d["l3.weight"].numpy().astype(np.float32).T,
    ]
    biases = [
        d["ft.bias"].numpy().astype(np.float32),
        d["l1.bias"].numpy().astype(np.float32),
        d["l2.bias"].numpy().astype(np.float32),
        d["l3.bias"].numpy().astype(np.float32),
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        for arr in weights + biases:
            arr.tofile(f)

    print(f"Saved normal binary → {output_file}")


def save_normal_json(pt_file: str, output_file: str = "weights/nn_weightsNormal.json"):
    """Save float32 weights as JSON."""
    d = _load(pt_file)

    data = {
        "ft.weight": d["ft.weight"].numpy().astype(np.float32).T.tolist(),
        "l1.weight": d["l1.weight"].numpy().astype(np.float32).T.tolist(),
        "l2.weight": d["l2.weight"].numpy().astype(np.float32).T.tolist(),
        "l3.weight": d["l3.weight"].numpy().astype(np.float32).T.tolist(),
        "ft.bias":   d["ft.bias"].numpy().astype(np.float32).tolist(),
        "l1.bias":   d["l1.bias"].numpy().astype(np.float32).tolist(),
        "l2.bias":   d["l2.bias"].numpy().astype(np.float32).tolist(),
        "l3.bias":   d["l3.bias"].numpy().astype(np.float32).tolist(),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f)

    print(f"Saved normal JSON → {output_file}")



def main():
    quantized_pt = "weights/nnue_weightsQuantized.pt"
    normal_pt    = "weights/nnue_weightsNormal.pt"

    save_quantized_binary(quantized_pt)
    save_quantized_json(quantized_pt)

    save_normal_binary(normal_pt)
    save_normal_json(normal_pt)


if __name__ == "__main__":
    main()
