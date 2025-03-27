import torch
import training
import chess
import numpy as np


model = training.ChessNNUE()
weights = torch.load("nnue_weights.pt") 
state_dict = {
    "ft.weight": weights["ft.weight"] / 255,
    "l1.weight": weights["l1.weight"] / 255,
    "l2.weight": weights["l2.weight"] / 255,
    "l3.weight": weights["l3.weight"] / 255,
    "ft.bias": weights["ft.bias"] / 255,   
    "l1.bias": weights["l1.bias"] / 255,
    "l2.bias": weights["l2.bias"] / 255,
    "l3.bias": weights["l3.bias"] / 255
}
model.load_state_dict(state_dict)
model.eval()
print("✅ Model loaded successfully!")
# Convert FEN to Input Tensor
def fen_to_tensor(fen):
    board = chess.Board(fen)

    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    white_feature = np.zeros(768, dtype=np.float32)
    black_feature = np.zeros(768, dtype=np.float32)
    for square, piece in board.piece_map().items():
        piece_idx = piece_map[piece.piece_type]
        
        # **WHITE PERSPECTIVE (Normal board view)**
        index = square * 12 + piece_idx + (6 if piece.color == chess.BLACK else 0)
        white_feature[index] = 1  # Includes both White and Black pieces

        # **BLACK PERSPECTIVE (Mirrored board view)**
        flipped_square = chess.square_mirror(square)
        index = flipped_square * 12 + piece_idx + (6 if piece.color == chess.WHITE else 0)
        black_feature[index] = 1  # Includes both White and Black pieces

    stm = 1 if board.turn == chess.WHITE else 0

    # white_feature, black_feature, stm
    return torch.tensor(white_feature).unsqueeze(0), torch.tensor(black_feature).unsqueeze(0), torch.tensor(stm).unsqueeze(0)

# Test FEN Position
fen = "r1bqkb1r/pp2p1pp/5p1n/3P4/3N4/2N5/PPP1BPPP/R1BQK2R w KQkq - 0 1"
white_feature, black_feature, stm = fen_to_tensor(fen)

# Make Prediction
with torch.no_grad():
    output = model(white_feature, black_feature, stm).item()

print(f"♟️ Chess NNUE Evaluation: {output:.3f}")