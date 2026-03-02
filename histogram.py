"""
histogram.py
────────────
Dataset analysis tool. Loads chessData.csv and produces visualizations of
the evaluation distribution and normalization functions.

Output images are saved to DatasetAnalysis/.

Run:  python histogram.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from training import process_eval, QO

DATASET_PATH = "Data/chessData.csv"
OUTPUT_DIR   = "DatasetAnalysis"


def load_dataset(path: str) -> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path, names=["fen", "eval"], encoding="utf-8")
    df["eval_cp"] = df["eval"].apply(process_eval)
    print(f"  {len(df):,} positions loaded")
    print(f"  Mean : {df['eval_cp'].mean():+.1f} cp")
    print(f"  Std  : {df['eval_cp'].std():.1f} cp")
    return df


def plot_raw_distribution(df: pd.DataFrame, output_dir: str):
    """Histogram of raw centipawn evaluations (clipped to ±1000 for readability)."""
    clipped = df["eval_cp"].clip(-1000, 1000)

    plt.figure(figsize=(12, 4))
    plt.hist(clipped, bins=100, color="steelblue", alpha=0.8)
    plt.axvline(0, color="red", linestyle="--", alpha=0.6, label="0 cp")
    plt.title("Raw Evaluation Distribution (centipawns, clipped ±1000)")
    plt.xlabel("Evaluation (cp)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "eval_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_wdl_distribution(df: pd.DataFrame, output_dir: str):
    """Histogram of WDL (win probability) targets used during training."""
    wdl = 1.0 / (1.0 + np.exp(-df["eval_cp"] / QO))

    plt.figure(figsize=(12, 4))
    plt.hist(wdl, bins=100, color="seagreen", alpha=0.8)
    plt.axvline(0.5, color="red", linestyle="--", alpha=0.6, label="0.5 (equal)")
    plt.title(f"WDL Target Distribution  (sigmoid(cp / {QO}))")
    plt.xlabel("Win Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "wdl_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_normalization_functions(output_dir: str):
    """Compare different normalization curves over centipawn range."""
    x = np.linspace(-1500, 1500, 2000)

    sigmoid_qo  = 1.0 / (1.0 + np.exp(-x / QO))
    sigmoid_400 = 1.0 / (1.0 + np.exp(-x / 400))
    tanh_400    = np.tanh(x / 400) * 0.5 + 0.5   # shifted to [0,1]

    plt.figure(figsize=(12, 5))
    plt.plot(x, sigmoid_qo,  label=f"sigmoid(cp/{QO})  ← training target", linewidth=2)
    plt.plot(x, sigmoid_400, label="sigmoid(cp/400)", linewidth=2, linestyle="--")
    plt.plot(x, tanh_400,    label="tanh(cp/400) scaled to [0,1]", linewidth=2, linestyle=":")
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    plt.axvline(0,   color="gray", linestyle="--", alpha=0.4)
    plt.title("Normalization Functions — centipawns → win probability")
    plt.xlabel("Evaluation (centipawns)")
    plt.ylabel("Win probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "normalization_functions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset(DATASET_PATH)

    print("\nGenerating plots...")
    plot_raw_distribution(df, OUTPUT_DIR)
    plot_wdl_distribution(df, OUTPUT_DIR)
    plot_normalization_functions(OUTPUT_DIR)

    print(f"\nDone in {time.time() - t0:.1f}s  — images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
