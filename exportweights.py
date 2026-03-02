"""
exportweights.py
────────────────
Export trained NNUE weights to binary (.nnue) and JSON formats for use in the
chess engine.

Binary layout (weights then biases, in order ft → l1 → l2):
    Quantized  : ft [int16], l1 [int8], l2 [int8], ft_b [int32], l1_b [int32], l2_b [int32]
    Normal     : ft [float32], l1 [float32], l2 [float32], ft_b [float32], l1_b [float32], l2_b [float32]

Weight matrices are transposed (shape [in, out]) so the engine can use
column-major access patterns when building the accumulator.

Chess engine inference note:
    centipawns = out_q >> 12   (integer output divided by QB² = 64² = 4096)
"""

import torch
import numpy as np
import json
import os


def _load(pt_file: str) -> dict:
    return torch.load(pt_file, map_location="cpu")


# ── Quantized (integer) exports ───────────────────────────────────────────────

def save_quantized_binary(pt_file: str, output_file: str = "weights/nn_weightsQuantized.nnue"):
    """Save quantized weights as a packed binary file."""
    d = _load(pt_file)

    weights = [
        d["ft.weight"].numpy().astype(np.int16).T,   # [768, HIDDEN]
        d["l1.weight"].numpy().astype(np.int8).T,    # [2*HIDDEN, 128]
        d["l2.weight"].numpy().astype(np.int8).T,    # [128, 1]
    ]
    biases = [
        d["ft.bias"].numpy().astype(np.int16),       # [HIDDEN] int16: matches accumulator dtype for SIMD
        d["l1.bias"].numpy().astype(np.int32),       # [128]
        d["l2.bias"].numpy().astype(np.int32),       # [1]
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
        "ft.bias":   d["ft.bias"].numpy().astype(np.int16).tolist(),
        "l1.bias":   d["l1.bias"].numpy().astype(np.int32).tolist(),
        "l2.bias":   d["l2.bias"].numpy().astype(np.int32).tolist(),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f)

    print(f"Saved quantized JSON → {output_file}")


# ── Float (normal) exports ────────────────────────────────────────────────────

def save_normal_binary(pt_file: str, output_file: str = "weights/nn_weightsNormal.nnue"):
    """Save float32 weights as a packed binary file."""
    d = _load(pt_file)

    weights = [
        d["ft.weight"].numpy().astype(np.float32).T,
        d["l1.weight"].numpy().astype(np.float32).T,
        d["l2.weight"].numpy().astype(np.float32).T,
    ]
    biases = [
        d["ft.bias"].numpy().astype(np.float32),
        d["l1.bias"].numpy().astype(np.float32),
        d["l2.bias"].numpy().astype(np.float32),
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
        "ft.bias":   d["ft.bias"].numpy().astype(np.float32).tolist(),
        "l1.bias":   d["l1.bias"].numpy().astype(np.float32).tolist(),
        "l2.bias":   d["l2.bias"].numpy().astype(np.float32).tolist(),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f)

    print(f"Saved normal JSON → {output_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    quantized_pt = "weights/nnue_weightsQuantized.pt"
    normal_pt    = "weights/nnue_weightsNormal.pt"

    save_quantized_binary(quantized_pt)
    save_quantized_json(quantized_pt)

    save_normal_binary(normal_pt)
    save_normal_json(normal_pt)


if __name__ == "__main__":
    main()
