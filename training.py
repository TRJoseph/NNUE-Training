import torch
import chess
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import os

# default model configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
NUM_EPOCHS = 10
DATASET_SAMPLE_SIZE = None

# mate score in centipawns, this may be worth looking at for increasing model performance
DEFAULT_MATE_SCORE = 5000

HIDDEN_LAYER_SIZE = 1024
QA = 127
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
                evaluation = -DEFAULT_MATE_SCORE if "-" in evaluation else DEFAULT_MATE_SCORE
            return float(evaluation)
            #return torch.sigmoid(torch.tensor(float(evaluation) / SCALE))

        self.chess_labels["eval"] = self.chess_labels["eval"].apply(process_eval)

        # this allows the program to grow or shrink the sample set from the dataset instead of using every single position found in the file
        if end_idx is None:
            end_idx = len(self.chess_labels)

        # keeps track of labels for later use
        labels = []

        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

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

        raw_label = self.labels[idx]
        sigmoid_label = cp_to_wdl(raw_label)
        stm = stm.clone().detach().float().unsqueeze(0)
        
        return white_features, black_features, stm, sigmoid_label, raw_label
    
def cp_to_wdl(cp):
    return torch.sigmoid(cp / QO)

def wdl_to_cp(wdl):
    # Avoid extreme values by clamping
    wdl_clamped = torch.clamp(wdl, 0.001, 0.999)
    return -QO * torch.log((1 / wdl_clamped) - 1)

class ChessNNUE(nn.Module):
    def __init__(self):
        super(ChessNNUE, self).__init__()

        self.ft = nn.Linear(768, HIDDEN_LAYER_SIZE)
        self.l1 = nn.Linear(2 * HIDDEN_LAYER_SIZE, 8)
        self.l2 = nn.Linear(8, 32)
        self.l3 = nn.Linear(32, 1)

        # activation range scaling factor (yoinked from stockfish quantization schema)
        self.QA = 127
        self.QB = 64
        self.QO = 400 

        nn.init.xavier_uniform_(self.ft.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l2.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l3.weight, gain=0.01)
        
        # Initialize biases to small values
        nn.init.zeros_(self.ft.bias)
        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)
        nn.init.zeros_(self.l3.bias)


    # The inputs are a whole batch!
    # `stm` indicates whether white is the side to move. 1 = true, 0 = false.
    def forward(self, white_features, black_features, stm, inference=False):
        w = self.ft(white_features)  # white's perspective
        b = self.ft(black_features)  # black's perspective

        # we order the accumulators for 2 perspectives based on who is to move.
        # So we blend two possible orderings by interpolating between `stm` and `1-stm` tensors.
        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        # runs the linear layers and use clamp as ClippedReLU
        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        l3_x = torch.clamp(self.l2(l2_x), 0.0, 1.0)


        raw_output = self.l3(l3_x)

        # during inference, scale output instead of passing evaluation through sigmoid
        if inference:
            # this output is with the scaling factor applied, in TigerEngine, the only code in the forward pass will be inference (obviously)
            output = raw_output * self.QO
            return output, raw_output
            
        output = torch.sigmoid(raw_output)

        return output, raw_output
    
    # I think this is fixed, now the weights and biases can be stored as shorts during inference in my chess engine
    # rounding causes some precision loss on the forward pass, but the gains in performance using FP8 and FP16 compute makes up for the loss
    def quantize_weights_and_biases(self):
        # Feature Transformer (ft)
        self.ft_weight_quantized = (self.ft.weight.data * 127).round().to(torch.int16)
        self.ft_bias_quantized = (self.ft.bias.data * 127).round().to(torch.int16)
        
        # Hidden Layer 1 (l1)
        self.l1_weight_quantized = (self.l1.weight.data * 64).round().to(torch.int8)
        self.l1_bias_quantized = (self.l1.bias.data * 127 * 64).round().to(torch.int32)
        
        # Hidden Layer 2 (l2)
        self.l2_weight_quantized = (self.l2.weight.data * 64).round().to(torch.int8)
        self.l2_bias_quantized = (self.l2.bias.data * 127 * 64).round().to(torch.int32)
        
        # Output Layer (l3)
        self.l3_weight_quantized = (self.l3.weight.data * (64 * 400 / 127)).round().to(torch.int8) 
        self.l3_bias_quantized = (self.l3.bias.data * 64 * 400).round().to(torch.int32) 

    
def train_loop(dataloader, model, loss_fn, optimizer, batch_size, epoch):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    for batch, (white_features, black_features, stm, target, raw_eval) in enumerate(dataloader):
        white_features, black_features, stm, target, raw_eval = (
            white_features.to("cuda"),
            black_features.to("cuda"),
            stm.to("cuda"),
            target.to("cuda"),
            raw_eval.to("cuda"),
        )

        pred_sigmoid, pred_raw = model(white_features, black_features, stm)
        loss = loss_fn(pred_sigmoid, target)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check for vanishing or exploding gradients
        # for param in model.parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"Gradient norm: {grad_norm}")
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(white_features)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

         # Early in training, print some evaluations
        if epoch <= 1 and batch % 500 == 0:
            with torch.no_grad():
                # For visualization, convert sigmoid back to centipawn
                pred_cp = wdl_to_cp(pred_sigmoid[:3])
                for i in range(3):
                    print(f"  Sample {i}: Pred CP: {pred_cp[i].item():.1f}, Actual CP: {raw_eval[i].item():.1f}")

    return running_loss / len(dataloader)
    

def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for (white_features, black_features, stm, target, raw_eval) in dataloader:
            white_features, black_features, stm, target, raw_eval = (
                white_features.to("cuda"),
                black_features.to("cuda"),
                stm.to("cuda"),
                target.to("cuda"),
                raw_eval.to("cuda"),
            )
            
            pred_sigmoid, pred_raw = model(white_features, black_features, stm)
            
            # loss is on the sigmoid values 
            loss = loss_fn(pred_sigmoid, target)
            total_loss += loss.item() * len(white_features)
            
            pred_cp = wdl_to_cp(pred_sigmoid)
            mae = torch.abs(pred_cp - raw_eval).mean().item()
            
            total_mae += mae * len(white_features)
            num_samples += len(white_features)

    # return loss and MAE
    return total_loss / num_samples, total_mae / num_samples

    
def run_model(dataset, loss_fn=nn.MSELoss(), lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # initialize model and optimizer
    model = ChessNNUE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # splits dataset into training and test sets
    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # lists to store losses for plotting
    train_losses = []
    test_losses = []

    # Training loop over epochs
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        
        # training loss
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, epoch)
        train_losses.append(train_loss)
        print(f"\nTRAIN LOOP COMPLETE on Epoch {epoch + 1}\nAverage Train Loss: {train_loss:.6f}\n")

        # test loss
        test_loss, test_mae = test_loop(test_dataloader, model, loss_fn)
        test_losses.append(test_loss)
        print(f"Validation after Epoch {epoch + 1}\nAverage Test Loss: {test_loss:.6f}, MAE: {test_mae:.6f}\n")

        with open("AvgLossResults.txt", "a") as f:
            f.write(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}, MAE = {test_mae:.6f}\n")

    print("Training Done!")
    return model, train_losses, test_losses

def plot_normalized_loss_comparison(dataset):
    epoch_range = np.arange(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    for loss_fn, name, color in [
        (nn.MSELoss(), "MSELoss", 'b'),
        (nn.SmoothL1Loss(), "SmoothL1Loss", 'r'),
        (nn.HuberLoss(), "HuberLoss", 'g')
    ]:
        model, train_losses, test_losses = run_model(dataset, loss_fn)
        normalized_losses = test_losses / test_losses[0]  # Normalize to start at 1
        plt.plot(epoch_range, normalized_losses, marker='o', linestyle='-', color=color, label=name)

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss")
    plt.title("Normalized Loss vs. Epoch")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return model

def plot_normalized_test_versus_train_loss(train_losses, test_losses):
    epoch_range = np.arange(1, NUM_EPOCHS + 1)

    train_losses = np.array(train_losses, dtype=np.float64)
    test_losses = np.array(test_losses, dtype=np.float64)

    # normalize to start at 1 to get an idea of the relative performance
    normalized_train_losses = train_losses / train_losses[0]
    normalized_test_losses =  test_losses / test_losses[0]
    plt.figure(figsize=(12, 5))
    plt.plot(epoch_range, normalized_train_losses, marker='o', linestyle='-', color='r', label="Training Loss")
    plt.plot(epoch_range, normalized_test_losses, marker='o', linestyle='-', color='g', label="Test Loss")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss")
    plt.title("Normalized Loss vs. Epoch")
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    print("Loading Dataset...\n")
    # torch.device("cuda" if torch.cuda.is_available() else "cpu"
    dataset = ChessDataset("chessData.csv", device="cpu", start_idx=0, end_idx=DATASET_SAMPLE_SIZE)
    print("Dataset Initialized!\n")   
    model = None

    model, train_losses, test_losses = run_model(dataset)

    # plots test and training loss to give an idea of model performance with the default global configuration (found at the top)
    #plot_normalized_test_versus_train_loss(train_losses, test_losses)

    # plots relative normalized losses with different loss functions, learning rates, and batch sizes
    #model = plot_normalized_loss_comparison(dataset)

    save_dir = "weights"

    if model is not None:
        ## quantizes weights and biases for use as NNUE
        model.quantize_weights_and_biases()

        # export normal weights
        torch.save({
            "ft.weight": model.ft.weight,  
            "l1.weight": model.l1.weight,
            "l2.weight": model.l2.weight,
            "l3.weight": model.l3.weight,

            "ft.bias": model.ft.bias,
            "l1.bias": model.l1.bias,
            "l2.bias": model.l2.bias,
            "l3.bias": model.l3.bias,
        }, os.path.join(save_dir, "nnue_weightsNormal.pt"))

        # export quantized weights
        torch.save({
            "ft.weight": model.ft_weight_quantized,  
            "l1.weight": model.l1_weight_quantized,
            "l2.weight": model.l2_weight_quantized,
            "l3.weight": model.l3_weight_quantized,

            "ft.bias": model.ft_bias_quantized,
            "l1.bias": model.l1_bias_quantized,
            "l2.bias": model.l2_bias_quantized,
            "l3.bias": model.l3_bias_quantized,
        }, os.path.join(save_dir, "nnue_weightsQuantized.pt"))

if __name__ == "__main__":
    main()
