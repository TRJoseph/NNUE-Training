import torch
import chess
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split


piece_map = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

class ChessDataset(Dataset):
    def __init__(self, chess_positions_file, transform=None, target_transform=None):
        self.chess_labels = pd.read_csv("./Data/" + chess_positions_file, names=["fen", "eval"])

        # mask for boolean array of every evaluation with checkmate in x number of moves
        mask = self.chess_labels["eval"].str.contains("#", na=False)



        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.chess_labels)
    
    def __getdataframe__(self):
        return self.chess_labels
    
    def __getitem__(self, idx):
        fen = self.chess_labels.iloc[idx, 0]
        evaluation = self.chess_labels.iloc[idx, 1]
        board = chess.Board(fen)
        features = [0] * 768
        for square, piece in chess.Board.piece_map(board).items():
            # Offset: 0-5 for white, 6-11 for black
            offset = 0 if piece.color == chess.WHITE else 6
            index = square * 12 + piece_map[piece.piece_type] + offset
            features[index] = 1
        tensor = torch.tensor(features, dtype=torch.float32), torch.tensor([float(evaluation)], dtype=torch.float32)
        return tensor

class ChessNNUE(nn.Module):
    def __init__(self):
        pass

def readData():
    with open("./Data/chessData.csv", "r") as f:
        next(f)
        line = f.readline().strip()
        fen, eval = line.split(",")
        print(f"FEN: {fen}, Eval: {eval}")
        return fen
    
def getDataframe():
    return pd.read_csv("./Data/chessData.csv", header=None, names=["fen", "eval"], sep=",")

def countLines():
    with open("./Data/chessData.csv", "r") as f:
        # Count all lines including header
        total_lines = sum(1 for line in f)
        print(f"Total number of lines: {total_lines}")
        return total_lines

def main():
    #fen = readData()
    #createFeatureVector(chess.Board(fen))

    dataset = ChessDataset("chessData.csv")
    df = dataset.__getdataframe__()
    print(df.shape)

    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])
    print(len(training_data), len(test_data))

    # prepares data with dataloader
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    for features, evals in train_dataloader:
        print("Train batch:", features.shape, evals.shape)
        break
    for features, evals in test_dataloader:
        print("Test batch:", features.shape, evals.shape)
        break

if __name__ == "__main__":
    main()