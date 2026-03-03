import torch
import chess

from training import (
    ChessNNUE,
    board_to_features,
    QA, QB, QO,
)

# Test FEN positions and their Stockfish evaluations (centipawns)
FEN_LIST = [
    "1nr3k1/p4p2/1p4p1/2qP1p1p/8/1QP2P2/P2N2PP/4RK2 w - - 0 31",
    "8/3kP3/1K4p1/8/2P2p1p/8/6PP/8 w - - 0 67",
    "8/3kP3/1p4p1/1K5p/2P2p2/8/6PP/8 w - - 1 66",
    "3r2r1/p2k1p2/1p1pp2p/8/3P4/2P5/P3KPPP/R6R w - - 2 21",
    "5r1k/p1pq2p1/1p1p3p/3B1b1Q/2P5/2P1P2n/PP5P/3R2RK b - - 0 27",
]
STOCKFISH_EVALS = ["0", "+5933", "+2440", "0", "-1019"]


def fen_to_tensors(fen: str):
    """Convert a FEN string to model-ready tensors (unsqueezed for batch dim=1)."""
    board = chess.Board(fen)
    white_feat, black_feat, stm = board_to_features(board)
    return (
        torch.from_numpy(white_feat).unsqueeze(0),   # [1, 768]
        torch.from_numpy(black_feat).unsqueeze(0),   # [1, 768]
        torch.tensor([[float(stm)]]),                # [1, 1]
    )


def _dequantize(weights: dict) -> dict:
    """Reconstruct float weights from the quantized integer weight dict."""
    return {
        "ft.weight": weights["ft.weight"].float() / QA,
        "ft.bias":   weights["ft.bias"].float()   / QA,
        "l1.weight": weights["l1.weight"].float() / QB,
        "l1.bias":   weights["l1.bias"].float()   / (QA * QB),
        "l2.weight": weights["l2.weight"].float() * QA / (QB * QO),
        "l2.bias":   weights["l2.bias"].float()   / (QB * QB * QO),
    }


def run_predictor(quantized: bool = False):
    model = ChessNNUE()

    if quantized:
        raw = torch.load("weights/nnue_weightsQuantized.pt", map_location="cpu")
        state_dict = _dequantize(raw)
        label = "Quantized"
    else:
        state_dict = torch.load("weights/nnue_weightsNormal.pt", map_location="cpu")
        label = "Normal"

    model.load_state_dict(state_dict)
    model.eval()
    print(f"\n{label} model loaded.\n")

    for i, (fen, sf_eval) in enumerate(zip(FEN_LIST, STOCKFISH_EVALS)):
        white_feat, black_feat, stm = fen_to_tensors(fen)
        with torch.no_grad():
            pred_cp, _ = model(white_feat, black_feat, stm, inference=True)
        print(f"Position {i + 1}  |  Stockfish: {sf_eval} cp  |  NNUE: {pred_cp.item():.1f} cp")


def main():
    run_predictor(quantized=False)
    run_predictor(quantized=True)


if __name__ == "__main__":
    main()
