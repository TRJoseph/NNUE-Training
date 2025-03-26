import torch
import chess
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt

piece_map = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

BATCH_SIZE = 64
LEARNING_RATE = 0.0003
NUM_EPOCHS = 20
DATASET_SAMPLE_SIZE = 2000000

HIDDEN_LAYER_SIZE = 1024
SCALE = 400  
QA = 255
QB = 64

class ChessDataset(Dataset):
    def __init__(self, chess_positions_file, device="cuda", transform=None, target_transform=None, start_idx=0, end_idx=None):
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.feature_length = (end_idx - start_idx)
        # all pieces from white's perspective
        self.white_features = []
        # all pieces from black's perspective
        self.black_features = []
        # side to move
        self.stm = []

        # load dataset 
        self.chess_labels = pd.read_csv("./Data/" + chess_positions_file, names=["fen", "eval"], encoding='utf-8')

        def process_eval(evaluation):
            if "#" in evaluation:
                evaluation = -10000 if "-" in evaluation else 10000

            return float(evaluation)
            #return torch.sigmoid(int(evaluation) / SCALE)

        self.chess_labels["eval"] = self.chess_labels["eval"].apply(process_eval)

        # Use start_idx and end_idx to slice the dataset
        if end_idx is None:
            end_idx = len(self.chess_labels)

        # Extract features and labels for the specific range
        labels = []

        print(f"\nConverting {str(end_idx - start_idx)} board positions to feature arrays...\n")
        for idx in range(start_idx, end_idx):
            fen = self.chess_labels.iloc[idx, 0]
            evaluation = self.chess_labels.iloc[idx, 1]
            board = chess.Board(fen)
            print(f"Processing position {idx + 1}...")

            # initialize feature arrays for both perspectives
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

            self.white_features.append(white_feature)
            self.black_features.append(black_feature)
            labels.append(evaluation)
            self.stm.append(1 if board.turn == chess.WHITE else 0)  # 1 if White to move, 0 if Black to move

        print("\nConverting lists to tensors...\n")
        # Convert lists to tensors
        self.white_features = torch.tensor(np.array(self.white_features), dtype=torch.float32, device=self.device)
        self.black_features = torch.tensor(np.array(self.black_features), dtype=torch.float32, device=self.device)
        self.labels = torch.tensor(np.array(labels).reshape(-1, 1), dtype=torch.float32, device=self.device)  # Reshape for PyTorch
        self.stm = torch.tensor(np.array(self.stm), dtype=torch.float32, device=self.device)  # Convert side to move into tensor


    def __len__(self):
        return self.feature_length
    
    def __getdataframe__(self):
        return self.chess_labels
    
    def __getitem__(self, idx):
        white_features = self.white_features[idx] 
        black_features = self.black_features[idx]  
        stm = self.stm[idx] 
        label = cp_to_wdl(self.labels[idx])
        stm = stm.clone().detach().float().unsqueeze(0)
        
        return white_features, black_features, stm, label
    
def cp_to_wdl(cp):
    return torch.sigmoid(cp / SCALE)


    
class ChessNNUE(nn.Module):
    def __init__(self):
        super(ChessNNUE, self).__init__()

        self.ft = nn.Linear(768, HIDDEN_LAYER_SIZE)
        self.l1 = nn.Linear(2 * HIDDEN_LAYER_SIZE, 8)
        self.l2 = nn.Linear(8, 32)
        self.l3 = nn.Linear(32, 1)

    # The inputs are a whole batch!
    # `stm` indicates whether white is the side to move. 1 = true, 0 = false.
    def forward(self, white_features, black_features, stm):
        w = self.ft(white_features) # white's perspective
        b = self.ft(black_features) # black's perspective

        # Remember that we order the accumulators for 2 perspectives based on who is to move.
        # So we blend two possible orderings by interpolating between `stm` and `1-stm` tensors.
        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        # Run the linear layers and use clamp_ as ClippedReLU
        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        l3_x = torch.clamp(self.l2(l2_x), 0.0, 1.0)
        return self.l3(l3_x)
    
def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (white_features, black_features, stm, target) in enumerate(dataloader):
        white_features, black_features, stm, target = (
            white_features.to("cuda"),
            black_features.to("cuda"),
            stm.to("cuda"),
            target.to("cuda"),
        )
        # Forward pass
        pred = model(white_features, black_features, stm)
        loss = loss_fn(pred, target)

        # Backpropagation
        loss.backward()

        # Check for vanishing or exploding gradients
        # for param in model.parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"Gradient norm: {grad_norm}")
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(white_features)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():  # Disable gradient tracking for inference
        for (white_features, black_features, stm, target) in dataloader:
            white_features, black_features, stm, target = (
                white_features.to("cuda"),
                black_features.to("cuda"),
                stm.to("cuda"),
                target.to("cuda"),
            )
            pred = model(white_features, black_features, stm)
            loss = loss_fn(pred, target)

            total_loss += loss.item()  # Use .item() to get the scalar value

    # Compute average loss per batch
    avg_loss = total_loss / num_batches
    return avg_loss


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

    dataset = ChessDataset("chessData.csv", device="cuda", start_idx=0, end_idx=DATASET_SAMPLE_SIZE)
    train_size = int(0.80 * dataset.__len__())
    test_size = int(dataset.__len__()) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])
    print(len(training_data), len(test_data))

    # prepares data with dataloader
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    # Iterate over training batches
    for white_features, black_features, stm, evals in train_dataloader:
        print("Train batch:", white_features.shape, black_features.shape, stm.shape, evals.shape)
        break

    # Iterate over testing batches
    for white_features, black_features, stm, evals in test_dataloader:
        print("Test batch:", white_features.shape, black_features.shape, stm.shape, evals.shape)
        break
    loss_values = np.array([])

    epochs = NUM_EPOCHS
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, BATCH_SIZE)
        print("\n\n TRAIN LOOP COMPLETE on epoch " + str(epoch+1) + "\n\n")
        loss = test_loop(test_dataloader, model, loss_fn)
        loss_values = np.append(loss_values, loss)
        print("\n\n TEST LOOP COMPLETE on epoch " + str(epoch+1) + "\n\n")
        with open("AvgLossResults.txt", "a") as f:
            f.write(f"Epoch {epoch+1}: Loss = {loss}\n")
    print("Done!")

    epoch_range = np.arange(1, NUM_EPOCHS + 1)
    # Plot Loss vs Epoch
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, loss_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.grid()

    # Plot Loss vs Log(Epoch)
    plt.subplot(1, 2, 2)
    plt.plot(np.log(epoch_range), loss_values, marker='o', linestyle='-', color='r')
    plt.xlabel("log(Epoch)")
    plt.ylabel("Loss")
    plt.title("Loss vs. log(Epoch)")
    plt.grid()

    plt.tight_layout()
    plt.show()

    # export weights
    torch.save({
        "ft.weight": model.ft.weight.t() * 255,  # [768, 1024]
        "l1.weight": model.l1.weight.t() * 255,  # [2048, 8]
        "l2.weight": model.l2.weight.t() * 255,  # [8, 32]
        "l3.weight": model.l3.weight.t() * 255,  # [32, 1]
        "ft.bias": model.ft.bias.t() * 255,
        "l1.bias": model.l1.bias * 255, 
        "l2.bias": model.l2.bias * 255,
        "l3.bias": model.l3.bias * 255                        
    }, "nnue_weights.pt")


if __name__ == "__main__":
    main()