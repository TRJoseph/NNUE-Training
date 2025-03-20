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

        # this sets the matescore and adds a bias depending on how many moves away from mate it is
        # for example, "#+4" (mate-in-4 for white) = 10000 + (100/4) = 10025 whereas "#+1" (mate-in-1 for white) = 10000 + (100/1) = 10100
        # this will hopefully reflect to the NN that a quicker mate is a "better" position
        self.chess_labels.loc[mask, "eval"] = self.chess_labels.loc[mask, "eval"].apply(
           lambda x: 10500 if x == "#+0" else -10500 if x == "#-0" else (10000 + (100/int(x.replace("#", ""))) if int(x.replace("#", "")) > 0 else -10000 + (100/int(x.replace("#", ""))))
        )

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

class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.cat((self.relu(x), self.relu(-x)), dim=1)
class ChessNNUE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(768, 1024) 
        self.fc2 = nn.Linear(2048, 1)

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(768, 1024),
        #     nn.Linear(2048, 1)
        # )
    def forward(self, x):
        hidden = self.fc1(x)
        combined = torch.cat((hidden, hidden), dim=1)
        return self.fc2(combined)
    
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
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
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
    model = ChessNNUE().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)

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

    # run train and test loops
    train_loop(train_dataloader, model, loss_fn, optimizer, 64)
    print("\n\n TRAIN LOOP COMPLETE \n\n")
    test_loop(test_dataloader, model, loss_fn)
    print("\n\n TEST LOOP COMPLETE \n\n")


if __name__ == "__main__":
    main()