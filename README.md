# ChessNNUETraining
This repository is a host for all data processing, PGNs, and python code related to the training of my NNUE (Efficiently Updatable Neural Network) for my personal Chess Engine, TigerEngine.

## Overview
NNUE, (ƎUИИ Efficiently Updatable Neural Networks) are a Neural Network architecture intended to replace traditional evaluation methods for zero-sum games such as chess and Shogi. This repository tracks the PyTorch model and training code that I have created in order to replace my personal Engine's (TigerEngine's) evaluation function.

### Training.py
In the training.py python file, one will find dataset setup code that converts a given board position into feature arrays of 768 neurons (corresponding to 64 squares * 12 pieces (6 white and 6 black)). It is well known now that
having the network consider the board perspective from both black and white's perspective is advantageous, hence the concatenation on the first hidden layer of the network. The actual network architecture itself resembles what Stockfish has been transitioning to slowly over the years (`1024x2->8->32->1`) [Read more about this here](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#consideration-of-networks-size-and-cost). Before training the model, it is important and standard practice to run the targets through a sigmoid activation function. This is is to minimize large gradients and to ensure the model is able to figure out intricacies between tense positions.

### Quantization
This code outputs both the standard and quantized weights and biases for inference. The quantization method follows the recommendations from the [Chess Programming Wiki](https://www.chessprogramming.org/NNUE#Quantization), enabling my evaluation code to utilize FP8 and FP16 computations instead of standard FP32. This optimization enhances efficiency when evaluating positions within the engine. One can technically get by without doing this step, but in all likelihood the minimax search tree would not be able to search nearly as deep in the same time frame, negatively impacting engine elo.
