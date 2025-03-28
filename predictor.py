import torch
import training
import chess
import numpy as np


# activation range scaling factor
S_A = 127
# weight scaling factor
S_W = 64
# output scaling factor
S_O = 10000

model = training.ChessNNUE()
weights = torch.load("nnue_weightsQuantized.pt") 
state_dict = {
    "ft.weight": weights["ft.weight"] / S_A,  # Dequantize from 127
    "l1.weight": weights["l1.weight"] / S_W,  # Correct (64)
    "l2.weight": weights["l2.weight"] / S_W,  # Correct (64)
    "l3.weight": weights["l3.weight"] / (S_W * S_O / S_A),  # Reverse 64 * 10000 / 127
    "ft.bias": weights["ft.bias"] / S_A,  # Dequantize from 127
    "l1.bias": weights["l1.bias"] / (S_A * S_W),  # Correct (127 * 64)
    "l2.bias": weights["l2.bias"] / (S_A * S_W),  # Correct (127 * 64)
    "l3.bias": weights["l3.bias"] / (S_W * S_O),  # Reverse 64 * 10000
}

model.load_state_dict(state_dict)
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

# Test FEN Positions (~equal position, heavily winning for white, heaviley winning for white (slightly less), another ~equal positions)
fenList = ["1nr3k1/p4p2/1p4p1/2qP1p1p/8/1QP2P2/P2N2PP/4RK2 w - - 0 31", "8/3kP3/1K4p1/8/2P2p1p/8/6PP/8 w - - 0 67", "8/3kP3/1p4p1/1K5p/2P2p2/8/6PP/8 w - - 1 66", "3r2r1/p2k1p2/1p1pp2p/8/3P4/2P5/P3KPPP/R6R w - - 2 21", "5r1k/p1pq2p1/1p1p3p/3B1b1Q/2P5/2P1P2n/PP5P/3R2RK b - - 0 27"]
evaluations = ["0", "+5933", "+2440", "0", "-1019"]

#fenList = ["r1b3k1/pp1nb1pp/1q2p3/3pP3/3n4/P2B1P2/1PQBN2P/R3K2R w KQ - 0 16"]

for i in range(0, len(fenList)):
    white_feature, black_feature, stm = fen_to_tensor(fenList[i])
    # Make Prediction
    with torch.no_grad():
        output = model(white_feature, black_feature, stm).item()

    print(f"Position {i+1} with Stockfish evaluation: {evaluations[i]}\n")
    print(f"♟️ Chess NNUE Evaluation: {output:.3f}\n")

