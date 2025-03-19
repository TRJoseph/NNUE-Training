import chess
import chess.engine
import chess.pgn
import torch
import numpy
import os
import random
import zstandard as zstd
from io import StringIO

STOCKFISH_PATH = "C:/Users/trjos/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
PGN_DIR = "./PGNs"
PGN_FILES = []

def count_games_in_pgns():
    total_games = 0
    for pgn_file_name in PGN_FILES:
        full_path = os.path.join(PGN_DIR, pgn_file_name)
        with open(full_path) as pgn_file:
            while chess.pgn.read_game(pgn_file) is not None:
                print(total_games)
                total_games += 1
        print(f"Games in {pgn_file_name}: {total_games - (total_games - total_games)}")
    print(f"Total games across all files: {total_games}")
    return total_games

def start_stockfish_engine(stockfish_path: str) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    return engine

def get_evaluation(engine: chess.engine.SimpleEngine, fen: str, time: float = 1.0) -> float:
    """Get Stockfish evaluation in pawns for a FEN position."""
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(time=time))  # search for 'time' seconds
    score = info["score"].relative.score(mate_score=10000)  # centipawns or mate
    if score is None:
        return 0.0
    return score / 100.0
    
def get_pgns(path="."):
    for entry in os.listdir(path):
        PGN_FILES.append(entry)

def parse_pgn(engine: chess.engine.SimpleEngine):
    total_games_analyzed = 0
    total_positions_analyzed = 0
    for pgn_file_name in PGN_FILES:
        full_path = os.path.join(PGN_DIR, pgn_file_name)
        with open(full_path, "rb") as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                # Wrap the reader in a text stream for chess.pgn
                text_stream = StringIO()
                buffer = ""
                while total_positions_analyzed < 2000000:
                    chunk = reader.read(8192).decode("utf-8", errors="ignore")  # Read 8KB at a time
                    if not chunk:  # End of file
                        break
                    buffer += chunk
                    text_stream.seek(0)
                    text_stream.write(buffer)
                    text_stream.seek(0)
                    pgn = chess.pgn.read_game(text_stream)
                    if pgn is None:  # Incomplete game at buffer end
                        buffer = buffer[text_stream.tell():]  # Keep remainder
                        continue
                    buffer = buffer[text_stream.tell():]  # Clear processed part
                    # Process the game
                    board = pgn.board()
                    moves = list(pgn.mainline_moves())
                    num_moves = len(moves)
                    sample_indices = random.sample(range(10, min(40, num_moves)), min(5, max(0, num_moves - 10)))
                    for i, move in enumerate(moves):
                        if total_positions_analyzed >= 2000000:
                            break
                        board.push(move)
                        if i in sample_indices:
                            fen = board.fen()
                            eval = get_evaluation(engine, fen, 0.1)
                            with open("training_data.txt", "a") as data_file:
                                data_file.write(f"{fen},{eval}\n")
                            total_positions_analyzed += 1
                    total_games_analyzed += 1
                    print(f"Total Games Analyzed: {total_games_analyzed}")
                    print(f"Total Positions Analyzed: {total_positions_analyzed}")
                    if total_positions_analyzed >= 2000000:
                        break
    return total_games_analyzed

def main():
    get_pgns(PGN_DIR)
    #total_games = count_games_in_pgns()
    #print(f"Estimated positions with 5 samples/game: {total_games * 5}")
    engine = start_stockfish_engine(STOCKFISH_PATH)
    parse_pgn(engine)
    engine.quit()

if __name__ == "__main__":
    main()