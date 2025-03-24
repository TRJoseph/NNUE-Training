import torch
import chess
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import Parallel, delayed


piece_map = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

BATCH_SIZE = 64
LEARNING_RATE = 0.001

class ChessDataset(Dataset):
    def __init__(self, chess_positions_file, device="cuda", transform=None, target_transform=None, start_idx=0, end_idx=None):
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

        # Load CSV only once
        self.chess_labels = pd.read_csv("./Data/" + chess_positions_file, names=["fen", "eval"], encoding='utf-8')

        def process_eval(evaluation):
            if isinstance(evaluation, str) and "#" in evaluation:
                return -10000 if "-" in evaluation else 10000
            return float(evaluation)

        self.chess_labels["eval"] = self.chess_labels["eval"].apply(process_eval)

        # Use start_idx and end_idx to slice the dataset
        if end_idx is None:
            end_idx = len(self.chess_labels)

        # Extract features and labels for the specific range
        features = []
        labels = []

        print(f"\nConverting {str(end_idx-start_idx)} board positions to feature arrays...\n")
        for idx in range(start_idx, end_idx):
            fen = self.chess_labels.iloc[idx, 0]
            evaluation = self.chess_labels.iloc[idx, 1]
            board = chess.Board(fen)
            print("Index " + str(idx))

            # Initialize a feature array for the board state
            feature = np.zeros(774, dtype=np.float32)

            for square, piece in board.piece_map().items():
                offset = 0 if piece.color == chess.WHITE else 6
                index = square * 12 + piece_map[piece.piece_type] + offset
                feature[index] = 1

            # encode the side to move (1 if black's turn, 0 if white's turn)
            feature[768] = 1 if board.turn == chess.BLACK else 0

            feature[769] = 1 if board.castling_rights & chess.BB_A1 else 0
            feature[770] = 1 if board.castling_rights & chess.BB_H1 else 0
            feature[771] = 1 if board.castling_rights & chess.BB_A8 else 0
            feature[772] = 1 if board.castling_rights & chess.BB_H8 else 0

            # en passant square available
            feature[773] = 1 if board.ep_square != None else 0

            features.append(feature)
            labels.append(evaluation)

        print("\nConverting lists to tensors...\n")
        # Convert lists to numpy arrays and then to tensors
        self.features = torch.tensor(np.array(features), dtype=torch.float32, device=self.device)
        self.labels = torch.tensor(np.array(labels).reshape(-1, 1), dtype=torch.float32, device=self.device)  # Reshape to ensure tensor format

    def __len__(self):
        return len(self.features)
    
    def __getdataframe__(self):
        return self.chess_labels
    
    def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.cat((self.relu(x), self.relu(-x)), dim=1)
class ChessNNUE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(774, 1024) 
        self.crelu = CReLU() 
        self.fc2 = nn.Linear(2048, 1)
        #self.sigmoid = nn.Sigmoid()

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(768, 1024),
        #     nn.Linear(2048, 1)
        # )
    def forward(self, x):
        hidden = self.fc1(x)          # [batch_size, 1024]
        combined = self.crelu(hidden) # [batch_size, 2048]
        return self.fc2(combined)     # [batch_size, 1]
    
def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()  # Set model to evaluation mode
    #size = len(dataloader.dataset)

    total_loss = 0.0
    count = 0
    num_batches = len(dataloader)

    with torch.no_grad():  # Disable gradient tracking for inference
        for X, y in dataloader:
            X, y = X.to("cuda"), y.to("cuda")
            pred = model(X)

            loss = loss_fn(pred, y)
            count += len(y)
            total_loss += loss

    # Average loss per batch instead of per sample
    total_loss /= count
    return total_loss


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

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = ChessNNUE().to("cuda")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    dataset = ChessDataset("chessData.csv", device="cuda", start_idx=0, end_idx=3000000)
    train_size = int(0.80 * dataset.__len__())
    test_size = int(dataset.__len__()) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])
    print(len(training_data), len(test_data))

    # prepares data with dataloader
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    for features, evals in train_dataloader:
        print("Train batch:", features.shape, evals.shape)
        break
    for features, evals in test_dataloader:
        print("Test batch:", features.shape, evals.shape)
        break

    epochs = 1000
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, BATCH_SIZE)
        print("\n\n TRAIN LOOP COMPLETE on epoch " + str(epoch+1) + "\n\n")
        loss = test_loop(test_dataloader, model, loss_fn)
        centipawn_loss = loss ** 0.5  # Convert to centipawns
        print("\n\n TEST LOOP COMPLETE on epoch " + str(epoch+1) + "\n\n")
        with open("averagelossresultswithenpassant.txt", "a") as f:
            f.write(f"Epoch {epoch+1}: Loss = {loss} (Centipawns: {centipawn_loss:.2f})\n")
    print("Done!")


    # Export for C#
    torch.save({
        "accumulator_weights": model.fc1.weight.t() * 255,
        "output_weights": model.fc2.weight.view(2048) * 255
    }, "nnue_weights.pt")


if __name__ == "__main__":
    main()