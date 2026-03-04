"""
ParsePGNs.py
────────────
Generates a training data CSV from PGN files by sampling positions and
evaluating them with Stockfish.

Output format: one line per position: "fen,eval_centipawns"

Configuration:
  - Set STOCKFISH_PATH to your local Stockfish binary.
  - Put .pgn or .pgn.zst files into the PGNs/ directory.
  - Adjust MAX_POSITIONS, POSITIONS_PER_GAME, and EVAL_TIME as needed.
"""

import chess
import chess.engine
import chess.pgn
import os
import random
import zstandard as zstd
from io import StringIO

# ── Configuration ─────────────────────────────────────────────────────────────
STOCKFISH_PATH   = "C:/Users/trjos/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
PGN_DIR          = "./PGNs"
OUTPUT_FILE      = "./Data/training_data.csv"
MAX_POSITIONS    = 20_000_000   # stop after this many positions total
POSITIONS_PER_GAME = 5         # positions sampled per game
EVAL_TIME        = 0.1         # Stockfish analysis time per position (seconds)

# Only sample moves in this ply range (avoid openings and pure endgames)
SAMPLE_PLY_MIN = 10
SAMPLE_PLY_MAX = 40


# ── Stockfish helpers ─────────────────────────────────────────────────────────
def start_engine(path: str) -> chess.engine.SimpleEngine:
    return chess.engine.SimpleEngine.popen_uci(path)


def get_evaluation_cp(engine: chess.engine.SimpleEngine, fen: str) -> float:
    """
    Return Stockfish's evaluation of `fen` in centipawns (from the side to move).
    Mate scores are capped at ±10,000 cp.
    Returns 0.0 on any error.
    """
    board = chess.Board(fen)
    info  = engine.analyse(board, chess.engine.Limit(time=EVAL_TIME))
    score = info["score"].relative.score(mate_score=10_000)
    return float(score) if score is not None else 0.0


# ── PGN parsing ───────────────────────────────────────────────────────────────
def get_pgn_files(directory: str) -> list[str]:
    return [f for f in os.listdir(directory) if f.endswith((".pgn", ".pgn.zst", ".zst"))]


def _open_pgn_stream(full_path: str):
    """Return a text stream for a plain or zstd-compressed PGN file."""
    if full_path.endswith(".zst"):
        f = open(full_path, "rb")
        dctx = zstd.ZstdDecompressor()
        raw = dctx.stream_reader(f)
        return raw, f  # caller must close both
    else:
        f = open(full_path, encoding="utf-8", errors="ignore")
        return f, None


def parse_pgns(engine: chess.engine.SimpleEngine) -> int:
    """
    Walk every PGN file in PGN_DIR, sample positions, evaluate them with
    Stockfish, and append results to OUTPUT_FILE.

    Returns the number of positions written.
    """
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pgn_files = get_pgn_files(PGN_DIR)
    if not pgn_files:
        print(f"No PGN files found in {PGN_DIR}")
        return 0

    total_positions = 0
    total_games     = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for pgn_filename in pgn_files:
            full_path = os.path.join(PGN_DIR, pgn_filename)
            print(f"\nProcessing: {pgn_filename}")

            stream, extra = _open_pgn_stream(full_path)
            try:
                text_stream = StringIO()
                buffer      = ""

                while total_positions < MAX_POSITIONS:
                    chunk = stream.read(8192)
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8", errors="ignore")
                    if not chunk:
                        break

                    buffer += chunk
                    text_stream.seek(0)
                    text_stream.write(buffer)
                    text_stream.seek(0)

                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        buffer = buffer[text_stream.tell():]
                        continue
                    buffer = buffer[text_stream.tell():]

                    moves    = list(game.mainline_moves())
                    num_moves = len(moves)
                    hi        = min(SAMPLE_PLY_MAX, num_moves)
                    if hi <= SAMPLE_PLY_MIN:
                        continue
                    sample_indices = set(random.sample(
                        range(SAMPLE_PLY_MIN, hi),
                        min(POSITIONS_PER_GAME, hi - SAMPLE_PLY_MIN),
                    ))

                    board = game.board()
                    for i, move in enumerate(moves):
                        if total_positions >= MAX_POSITIONS:
                            break
                        board.push(move)
                        if i in sample_indices:
                            fen    = board.fen()
                            cp_val = get_evaluation_cp(engine, fen)
                            out.write(f"{fen},{cp_val:.0f}\n")
                            total_positions += 1

                    total_games += 1
                    if total_games % 100 == 0:
                        print(f"  Games: {total_games:,}  |  Positions: {total_positions:,}")

            finally:
                stream.close()
                if extra:
                    extra.close()

    print(f"\nDone. {total_positions:,} positions from {total_games:,} games → {OUTPUT_FILE}")
    return total_positions


def count_games(directory: str) -> int:
    """Quick count of games across all PGN files (for planning)."""
    pgn_files = get_pgn_files(directory)
    total = 0
    for fname in pgn_files:
        full_path = os.path.join(directory, fname)
        with open(full_path, encoding="utf-8", errors="ignore") as f:
            while chess.pgn.read_game(f) is not None:
                total += 1
    print(f"Total games: {total:,}  |  Est. positions at {POSITIONS_PER_GAME}/game: {total * POSITIONS_PER_GAME:,}")
    return total


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    engine = start_engine(STOCKFISH_PATH)
    try:
        parse_pgns(engine)
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
