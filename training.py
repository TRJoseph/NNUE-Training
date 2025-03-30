import torch
import chess
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
DATASET_SAMPLE_SIZE = 100000

HIDDEN_LAYER_SIZE = 1024
QA = 255
QB = 64
# output scaling factor
QO = 400  


class ChessDataset(Dataset):
    def __init__(self, chess_positions_file, device="cuda", transform=None, target_transform=None, start_idx=0, end_idx=None):
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.feature_length = 0
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
                evaluation = -2000 if "-" in evaluation else 2000
            return float(evaluation)
            #return torch.sigmoid(torch.tensor(float(evaluation) / SCALE))

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
        self.white_features = torch.tensor(np.array(self.white_features), dtype=torch.float32, device=self.device)
        self.black_features = torch.tensor(np.array(self.black_features), dtype=torch.float32, device=self.device)
        self.labels = torch.tensor(np.array(labels).reshape(-1, 1), dtype=torch.float32, device=self.device)
        self.stm = torch.tensor(np.array(self.stm), dtype=torch.float32, device=self.device)  # converts side to move into tensor
        self.feature_length = len(self.white_features)

    def __len__(self):
        return self.feature_length
    
    def __getdataframe__(self):
        return self.chess_labels
    
    def __getitem__(self, idx):
        white_features = self.white_features[idx] 
        black_features = self.black_features[idx]  
        stm = self.stm[idx] 
        label = self.labels[idx]
        #label = cp_to_wdl(self.labels[idx])
        stm = stm.clone().detach().float().unsqueeze(0)
        
        return white_features, black_features, stm, label
    
def cp_to_wdl(cp):
    return torch.sigmoid(cp / QO)


class ChessNNUE(nn.Module):
    def __init__(self):
        super(ChessNNUE, self).__init__()

        self.ft = nn.Linear(768, HIDDEN_LAYER_SIZE)
        self.l1 = nn.Linear(2 * HIDDEN_LAYER_SIZE, 8)
        self.l2 = nn.Linear(8, 32)
        self.l3 = nn.Linear(32, 1)

        # activation range scaling factor (yoinked from stockfish quantization schema)
        self.s_A = 127

        # scaling factors for quantization and activation
        self.QA = 255 
        self.QB = 64
        self.QO = 400 

    # The inputs are a whole batch!
    # `stm` indicates whether white is the side to move. 1 = true, 0 = false.
    def forward(self, white_features, black_features, stm):
        w = self.ft(white_features)  # white's perspective
        b = self.ft(black_features)  # black's perspective

        # we order the accumulators for 2 perspectives based on who is to move.
        # So we blend two possible orderings by interpolating between `stm` and `1-stm` tensors.
        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        # runs the linear layers and use clamp as ClippedReLU
        # clip to 127.0 following stockfish schema
        l1_x = torch.clamp(accumulator, 0.0, self.s_A)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, self.s_A)
        l3_x = torch.clamp(self.l2(l2_x), 0.0, self.s_A)

        output = self.l3(l3_x)
        return output
    
    # I think this is fixed, now the weights and biases can be stored as shorts during inference in my chess engine
    # rounding causes some precision loss on the forward pass, but the gains in performance using FP8 and FP16 compute makes up for the loss
    def quantize_weights_and_biases(self):
        self.ft_weight_quantized = (self.ft.weight.data * QA).round().to(torch.int16)
        self.ft_bias_quantized = (self.ft.bias.data * QA).round().to(torch.int16)
        
        self.l1_weight_quantized = (self.l1.weight.data * QB).round().to(torch.int8)
        self.l1_bias_quantized = (self.l1.bias.data * QB).round().to(torch.int16)
        
        self.l2_weight_quantized = (self.l2.weight.data * QB).round().to(torch.int8)
        self.l2_bias_quantized = (self.l2.bias.data * QB).round().to(torch.int16)
        
        self.l3_weight_quantized = (self.l3.weight.data * QB).round().to(torch.int16)
        self.l3_bias_quantized = (self.l3.bias.data * QO).round().to(torch.int16)
    
def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    model.train()
    for batch, (white_features, black_features, stm, target) in enumerate(dataloader):
        white_features, black_features, stm, target = (
            white_features.to("cuda"),
            black_features.to("cuda"),
            stm.to("cuda"),
            target.to("cuda"),
        )

        pred = model(white_features, black_features, stm)
        loss = loss_fn(pred, target)
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
    model.eval()

    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():  # disable gradient tracking for inference
        for (white_features, black_features, stm, target) in dataloader:
            white_features, black_features, stm, target = (
                white_features.to("cuda"),
                black_features.to("cuda"),
                stm.to("cuda"),
                target.to("cuda"),
            )
            pred = model(white_features, black_features, stm)
            loss = loss_fn(pred, target)

            total_loss += loss.item()

    # computes average loss per batch
    avg_loss = total_loss / num_batches
    return avg_loss

    
def run_model(dataset, loss_fn=nn.HuberLoss(), lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    """Train the ChessNNUE model with the specified loss function, learning rate, batch_size and return the loss values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = ChessNNUE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_values = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        print("\n\n TRAIN LOOP COMPLETE on Epoch " + str(epoch+1) + "\n\n")
        loss = test_loop(test_dataloader, model, loss_fn)
        loss_values.append(loss)
        print(f"Test Loss on Epoch {epoch+1}: {loss}\n")
        with open("AvgLossResults.txt", "a") as f:
            f.write(f"Epoch {epoch+1}: Loss = {loss}\n")

    print("Done!")
    return model, np.array(loss_values)

def plot_normalized_loss_comparison(dataset):
    epoch_range = np.arange(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    for loss_fn, name, color in [
        (nn.MSELoss(), "MSELoss", 'b'),
        (nn.SmoothL1Loss(), "SmoothL1Loss", 'r'),
        (nn.HuberLoss(), "HuberLoss", 'g')
    ]:
        model, loss_values = run_model(dataset, loss_fn)
        normalized_losses = loss_values / loss_values[0]  # Normalize to start at 1
        plt.plot(epoch_range, normalized_losses, marker='o', linestyle='-', color=color, label=name)

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss")
    plt.title("Normalized Loss vs. Epoch")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return model

def main():
    print("Loading Dataset...\n")
    # torch.device("cuda" if torch.cuda.is_available() else "cpu"
    dataset = ChessDataset("chessData.csv", device="cpu", start_idx=0, end_idx=DATASET_SAMPLE_SIZE)
    print("Dataset Initialized!\n")   
    model = None

    #model = plot_normalized_loss_comparison(dataset)

    model, loss_values = run_model(dataset, nn.MSELoss())

    if model is not None:
        ## quantizes weights and biases for use as NNUE
        model.quantize_weights_and_biases()

        # export quantized weights
        torch.save({
            "ft.weight": model.ft.weight,  
            "l1.weight": model.l1.weight,
            "l2.weight": model.l2.weight,
            "l3.weight": model.l3.weight,

            "ft.bias": model.ft.bias,
            "l1.bias": model.l1.bias,
            "l2.bias": model.l2.bias,
            "l3.bias": model.l3.bias,
        }, "nnue_weightsNormal.pt")

        # Save quantized weights
        torch.save({
            "ft.weight": model.ft_weight_quantized,  
            "l1.weight": model.l1_weight_quantized,
            "l2.weight": model.l2_weight_quantized,
            "l3.weight": model.l3_weight_quantized,

            "ft.bias": model.ft_bias_quantized,
            "l1.bias": model.l1_bias_quantized,
            "l2.bias": model.l2_bias_quantized,
            "l3.bias": model.l3_bias_quantized,
        }, "nnue_weightsQuantized.pt")

if __name__ == "__main__":
    main()