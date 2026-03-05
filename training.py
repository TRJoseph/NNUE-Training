import torch
import chess
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# ── Configuration ───────────────────────────────────────────────────────────
DATASET_FILE = "training_data.csv" # this needs to be in the /Data subdirectory

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE = 512
LEARNING_RATE = 0.0012
NUM_EPOCHS = 20
DATASET_SAMPLE_SIZE = None

# ── Quantization constants ────────────────────────────────────────────────────
# These must be kept in sync with the chess engine.
#
# Quantization scheme (engine integer inference):
#
#   FT layer  (input x is binary {0,1}):
#     acc_q  = ft_w_q @ x + ft_b_q   ≈  QA * acc_float      [int32]
#     h1_q   = clamp(acc_q, 0, QA)   ≈  QA * h1_float       [int16, range 0..QA]
#
#   L1 layer:
#     z1_q   = l1_w_q @ h1_q + l1_b_q  ≈  QA*QB * z1_float  [int32]
#     h2_q   = clamp(z1_q, 0, QA*QB)   ≈  QA*QB * h2_float  [int32, range 0..QA*QB]
#
#   L2 layer:
#     z2_q   = l2_w_q @ h2_q + l2_b_q  ≈  QA*QB² * z2_float [int32]
#     h3_q   = clamp(z2_q, 0, QA*QB²)  ≈  QA*QB² * h3_float [int32, range 0..QA*QB²]
#
#   L3 layer (output):
#     out_q  = l3_w_q @ h3_q + l3_b_q  ≈  QB²*QO * out_float [int32]
#
#   Chess engine final step:
#     centipawns = out_q >> 12          (divide by QB² = 4096)
#
QA = 255    # feature-transformer activation scale
QB = 64     # hidden-layer weight scale
QO = 410    # output scale (centipawns ≈ raw_output × QO)

# Clamp out-of-range mate scores to this value (centipawns)
DEFAULT_MATE_SCORE = 3000

INPUT_FEATURE_SIZE = 40960
HIDDEN_LAYER_SIZE = 1024
L1_SIZE = 64
L2_SIZE = 32

# ── Shared utilities ──────────────────────────────────────────────────────────
_PIECE_MAP = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4,
}


def board_to_features(board: chess.Board):
    """
    Convert a chess.Board to two INPUT_FEATURE_SIZE-element binary feature arrays and a STM flag.

    HalfKP layout: king_sq * 640 + piece_sq * 10 + piece_type_index
      - piece_type_index 0-4: own piece (P/N/B/R/Q)
      - piece_type_index 5-9: opponent piece (P/N/B/R/Q)
      - Kings are excluded as pieces; they are only used as the anchor square.
      - Black perspective mirrors both king and piece squares vertically.

    Returns:
        white_feature  np.ndarray float32 [INPUT_FEATURE_SIZE]
        black_feature  np.ndarray float32 [INPUT_FEATURE_SIZE]
        stm            int  1 = white to move, 0 = black to move
    """
    white_feature = np.zeros(INPUT_FEATURE_SIZE, dtype=np.float32)
    black_feature = np.zeros(INPUT_FEATURE_SIZE, dtype=np.float32)

    white_king_square_index = board.king(chess.WHITE)
    black_king_square_index = chess.square_mirror(board.king(chess.BLACK))

    for square, piece in board.piece_map().items():
        if piece.piece_type not in _PIECE_MAP:
            continue
        piece_idx = _PIECE_MAP[piece.piece_type]

        # "us"
        w_idx = white_king_square_index * 640 + square * 10 + piece_idx + (5 if piece.color == chess.BLACK else 0)
        white_feature[w_idx] = 1.0

        # "them"
        flipped = chess.square_mirror(square)
        b_idx = black_king_square_index * 640 + flipped * 10 + piece_idx + (5 if piece.color == chess.WHITE else 0)
        black_feature[b_idx] = 1.0

    stm = 1 if board.turn == chess.WHITE else 0
    return white_feature, black_feature, stm


def process_eval(evaluation) -> float:
    """Convert a Stockfish evaluation string/value to a clipped centipawn float."""
    s = str(evaluation).strip()
    try:
        # Handle mate scores: "#3", "#-5", "+M3", "-M5" etc.
        if "#" in s or "M" in s.upper():
            return float(-DEFAULT_MATE_SCORE if "-" in s else DEFAULT_MATE_SCORE)
        return float(np.clip(float(s), -DEFAULT_MATE_SCORE, DEFAULT_MATE_SCORE))
    except (ValueError, TypeError):
        return 0.0


def cp_to_wdl(cp: torch.Tensor) -> torch.Tensor:
    """Sigmoid-normalise a centipawn evaluation into a win probability [0, 1]."""
    return torch.sigmoid(cp / QO)


# ── Dataset ───────────────────────────────────────────────────────────────────
class ChessDataset(Dataset):
    """Lazy-loading dataset. FEN features are computed on-the-fly in __getitem__.
    Use num_workers=0 on Windows to avoid spawn issues."""

    def __init__(self, chess_positions_file: str, start_idx: int = 0, end_idx=None):
        filepath = os.path.join("Data", chess_positions_file)
        df = pd.read_csv(filepath, names=["fen", "eval"], encoding="utf-8")

        if end_idx is None:
            end_idx = len(df)
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        df["eval"] = df["eval"].apply(process_eval)

        # Store only the raw strings and floats — features computed lazily.
        self.fens  = df["fen"].to_numpy()
        self.evals = df["eval"].to_numpy(dtype=np.float32)

        print(f"Dataset loaded: {len(self.fens):,} positions")
        print(f"  Mean eval : {self.evals.mean():+.1f} cp")
        print(f"  Std  eval : {self.evals.std():.1f} cp")

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        white_feat, black_feat, stm = board_to_features(board)

        raw_eval = float(self.evals[idx])
        wdl_label = cp_to_wdl(torch.tensor(raw_eval))

        return (
            torch.from_numpy(white_feat),
            torch.from_numpy(black_feat),
            torch.tensor([float(stm)]),
            wdl_label.reshape(1),
            torch.tensor([raw_eval]),
        )


# ── Model ─────────────────────────────────────────────────────────────────────
class ChessNNUE(nn.Module):
    """
    HalfKP-style NNUE with two symmetric accumulators (white + black perspective).

    Architecture: INPUT_FEATURE_SIZE → HIDDEN_LAYER_SIZE (×2 after perspective concat) → L1_SIZE → L2_SIZE → 1

    Training:
        Loss is computed on sigmoid(raw_output) vs cp_to_wdl(target_cp).
        Activations are Clipped ReLU: clamp(x, 0, 1.0) in float, which maps
        exactly to clamp(x_q, 0, QA) in integer arithmetic (x_q = QA * x_float).

    Inference (chess engine):
        See module-level comment for the full integer quantization scheme.
        centipawns ≈ out_q >> 12   (divide by QB² = 4096)
    """

    def __init__(self):
        super().__init__()
        self.ft = nn.Linear(INPUT_FEATURE_SIZE, HIDDEN_LAYER_SIZE)
        self.l1 = nn.Linear(2 * HIDDEN_LAYER_SIZE, L1_SIZE)
        self.l2 = nn.Linear(L1_SIZE, L2_SIZE)
        self.l3 = nn.Linear(L2_SIZE, 1)

    def forward(self, white_features, black_features, stm, inference=False):
        w = self.ft(white_features)
        b = self.ft(black_features)

        # STM perspective comes first in the concatenated accumulator.
        acc = stm * torch.cat([w, b], dim=1) + (1 - stm) * torch.cat([b, w], dim=1)

        h1 = torch.clamp(acc, 0.0, 1.0)  # Clipped ReLU
        h2 = torch.clamp(self.l1(h1), 0.0, 1.0)
        h3 = torch.clamp(self.l2(h2), 0.0, 1.0)
        raw = self.l3(h3)

        if inference:
            return raw * QO, raw

        return torch.sigmoid(raw), raw

    # ── Quantization ──────────────────────────────────────────────────────────
    def quantize(self):
        """Compute quantized weights for chess engine integer inference."""
        self.ft_weight_q = (self.ft.weight.data * QA).round().to(torch.int16)
        self.ft_bias_q   = (self.ft.bias.data   * QA).round().to(torch.int16)  # int16: matches accumulator dtype for SIMD

        self.l1_weight_q = (self.l1.weight.data * QB).round().to(torch.int8)
        self.l1_bias_q   = (self.l1.bias.data   * QA * QB).round().to(torch.int32)

        self.l2_weight_q = (self.l2.weight.data * QB).round().to(torch.int8)
        self.l2_bias_q   = (self.l2.bias.data   * QA * QB * QB).round().to(torch.int32)

        # l3_b_q uses QB²*QO so engine recovers centipawns with out_q >> 12
        self.l3_weight_q = (self.l3.weight.data * QO / QA).round().to(torch.int8)
        self.l3_bias_q   = (self.l3.bias.data   * QB * QB * QO).round().to(torch.int32)

        for name, tensor in [("l1.weight", self.l1_weight_q), ("l2.weight", self.l2_weight_q), ("l3.weight", self.l3_weight_q)]:
            sat = (tensor.abs() == 127).float().mean().item()
            if sat > 0.01:
                print(f"  [WARNING] {name}: {sat * 100:.1f}% of values saturated at ±127 — "
                      "consider reducing QB or clipping weights before quantization.")

    def save_weights(self, save_dir: str = "weights"):
        """Quantize and export both float and integer weights to `save_dir`."""
        os.makedirs(save_dir, exist_ok=True)
        self.quantize()

        torch.save({
            "ft.weight": self.ft.weight.data,
            "ft.bias":   self.ft.bias.data,
            "l1.weight": self.l1.weight.data,
            "l1.bias":   self.l1.bias.data,
            "l2.weight": self.l2.weight.data,
            "l2.bias":   self.l2.bias.data,
            "l3.weight": self.l3.weight.data,
            "l3.bias":   self.l3.bias.data,
        }, os.path.join(save_dir, "nnue_weightsNormal.pt"))

        torch.save({
            "ft.weight": self.ft_weight_q,
            "ft.bias":   self.ft_bias_q,
            "l1.weight": self.l1_weight_q,
            "l1.bias":   self.l1_bias_q,
            "l2.weight": self.l2_weight_q,
            "l2.bias":   self.l2_bias_q,
            "l3.weight": self.l3_weight_q,
            "l3.bias":   self.l3_bias_q,
        }, os.path.join(save_dir, "nnue_weightsQuantized.pt"))

        print(f"Weights saved to '{save_dir}/'")


# ── Logging ───────────────────────────────────────────────────────────────────
def _run_signature(model: nn.Module, n_train: int, n_test: int, run_id: str) -> str:
    total_params  = sum(p.numel() for p in model.parameters())
    dataset_label = "entire dataset" if DATASET_SAMPLE_SIZE is None else f"{DATASET_SAMPLE_SIZE:,}"
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = "═" * 54
    return (
        f"\n{bar}\n"
        f"  NNUE Training Run — {ts}  [{run_id}]\n"
        f"{bar}\n"
        f"  Weights   : weights/{run_id}/\n"
        f"  Dataset   : {dataset_label}  (train {n_train:,}  |  test {n_test:,})\n"
        f"  Epochs    : {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}\n"
        f"  QA={QA}  QB={QB}  QO={QO}  MATE_CAP={DEFAULT_MATE_SCORE} cp\n"
        f"\n"
        f"  Architecture:\n"
        f"    FT  : {INPUT_FEATURE_SIZE} → {HIDDEN_LAYER_SIZE}  (×2 perspectives → {2 * HIDDEN_LAYER_SIZE})\n"
        f"    L1  : {2 * HIDDEN_LAYER_SIZE} → {L1_SIZE}\n"
        f"    L2  : {L1_SIZE} → {L2_SIZE}\n"
        f"    OUT : {L2_SIZE} → 1\n"
        f"    Params: {total_params:,}\n"
        f"{bar}\n"
    )


# ── Training infrastructure ───────────────────────────────────────────────────
def train_loop(dataloader, model, loss_fn, optimizer, batch_size, epoch):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0

    for batch, (white_features, black_features, stm, target, raw_eval) in enumerate(dataloader):
        white_features = white_features.to("cuda")
        black_features = black_features.to("cuda")
        stm            = stm.to("cuda")
        target         = target.to("cuda")
        raw_eval       = raw_eval.to("cuda")

        pred_norm, pred_raw = model(white_features, black_features, stm)
        loss = loss_fn(pred_norm, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if batch % 100 == 0:
            current = batch * batch_size + len(white_features)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

        # Early-epoch sample printout for a sanity check.
        if epoch <= 1 and batch % 500 == 0:
            with torch.no_grad():
                pred_cp = pred_raw * QO
                for i in range(min(3, len(pred_cp))):
                    print(f"  Sample {i}: Pred CP: {pred_cp[i].item():.1f}, "
                          f"Actual CP: {raw_eval[i].item():.1f}")

    return running_loss / len(dataloader)


def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    num_samples = 0

    with torch.no_grad():
        for white_features, black_features, stm, target, raw_eval in dataloader:
            white_features = white_features.to("cuda")
            black_features = black_features.to("cuda")
            stm            = stm.to("cuda")
            target         = target.to("cuda")
            raw_eval       = raw_eval.to("cuda")

            pred_norm, pred_raw = model(white_features, black_features, stm)
            loss = loss_fn(pred_norm, target)

            total_loss += loss.item() * len(white_features)
            total_mae  += torch.abs(pred_norm - target).mean().item() * QO * len(white_features)
            num_samples += len(white_features)

    return total_loss / num_samples, total_mae / num_samples


def run_model(dataset, loss_fn=nn.HuberLoss(), lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    torch.manual_seed(42)
    run_id   = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir  = os.path.join("weights", run_id)
    log_path = os.path.join(run_dir, "training_log.txt")
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model     = ChessNNUE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_size = int(0.80 * len(dataset))
    test_size  = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=0)

    train_losses = []
    test_losses  = []
    test_maes    = []

    sig = _run_signature(model, train_size, test_size, run_id)
    print(sig)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(sig)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}\n{'-' * 31}")

        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, epoch)
        train_losses.append(train_loss)
        print(f"\nTrain loss (epoch {epoch + 1}): {train_loss:.6f}")

        test_loss, test_mae = test_loop(test_dataloader, model, loss_fn)
        test_losses.append(test_loss)
        test_maes.append(test_mae)
        print(f"Test loss (epoch {epoch + 1}): {test_loss:.6f},  MAE ≈ {test_mae:.1f} cp\n")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch + 1:>3}: Train = {train_loss:.6f}  "
                    f"Test = {test_loss:.6f}  MAE = {test_mae:.1f} cp\n")

        scheduler.step()

    best_epoch = int(np.argmin(test_losses)) + 1
    summary = (
        f"\n── Final Summary {'─' * 37}\n"
        f"  Final   train={train_losses[-1]:.6f}  test={test_losses[-1]:.6f}  MAE={test_maes[-1]:.1f} cp\n"
        f"  Best    test={min(test_losses):.6f}  MAE={test_maes[best_epoch-1]:.1f} cp  (epoch {best_epoch})\n"
        f"{'─' * 54}\n"
    )
    print(summary)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary)

    print("Training done!")
    return model, train_losses, test_losses, run_id


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_normalized_test_versus_train_loss(train_losses, test_losses):
    epoch_range = np.arange(1, NUM_EPOCHS + 1)
    train_arr = np.array(train_losses, dtype=np.float64)
    test_arr  = np.array(test_losses,  dtype=np.float64)

    plt.figure(figsize=(12, 5))
    plt.plot(epoch_range, train_arr / train_arr[0], marker='o', color='r', label="Training Loss")
    plt.plot(epoch_range, test_arr  / test_arr[0],  marker='o', color='g', label="Test Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss")
    plt.title("Normalized Loss vs. Epoch")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_normalized_loss_comparison(dataset):
    epoch_range = np.arange(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    for loss_fn, name, color in [
        (nn.MSELoss(),    "MSELoss",    'b'),
        (nn.SmoothL1Loss(), "SmoothL1Loss", 'r'),
        (nn.HuberLoss(),  "HuberLoss",  'g'),
    ]:
        _, _, test_losses = run_model(dataset, loss_fn)
        arr = np.array(test_losses)
        plt.plot(epoch_range, arr / arr[0], marker='o', color=color, label=name)

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss")
    plt.title("Loss Function Comparison")
    plt.grid()
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...\n")
    dataset = ChessDataset(DATASET_FILE, start_idx=0, end_idx=DATASET_SAMPLE_SIZE)
    print("Dataset ready.\n")

    model, train_losses, test_losses, run_id = run_model(dataset)
    model.save_weights(f"weights/{run_id}")
    plot_normalized_test_versus_train_loss(train_losses, test_losses)


if __name__ == "__main__":
    main()
