"""
convert_lichess_eval.py
───────────────────────
Converts lichess_db_eval.jsonl.zst to the training CSV format used by training.py.
Output: one line per position — "fen,eval_centipawns"

Download lichess_db_eval.jsonl.zst from https://database.lichess.org/#evals
"""

import json
import os
import zstandard as zstd

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE    = "./Data/lichess_db_eval.jsonl.zst"
OUTPUT_FILE   = "./Data/training_data.csv"
MAX_POSITIONS = 10_000_000  # stop after writing this many positions
MATE_SCORE_CP = 3_000       # cp to use for mate scores (matches DEFAULT_MATE_SCORE)
MIN_DEPTH     = 20          # skip evals shallower than this (lower = noisier data)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _best_eval(evals: list) -> dict | None:
    """Highest-depth eval that meets MIN_DEPTH, or None."""
    candidates = [e for e in evals if e.get("depth", 0) >= MIN_DEPTH]
    return max(candidates, key=lambda e: e["depth"]) if candidates else None


def _pv_to_cp(pvs: list) -> int | None:
    """Extract centipawns from the first (best) PV. Returns None to skip."""
    if not pvs:
        return None
    pv = pvs[0]
    if "cp" in pv:
        return int(pv["cp"])
    if "mate" in pv:
        return MATE_SCORE_CP if pv["mate"] > 0 else -MATE_SCORE_CP
    return None


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    written = 0
    scanned = 0

    print(f"Reading {INPUT_FILE}")
    print(f"Target: {MAX_POSITIONS:,} positions  |  MIN_DEPTH: {MIN_DEPTH}\n")

    with (
        open(INPUT_FILE, "rb") as fh,
        zstd.ZstdDecompressor().stream_reader(fh) as reader,
        open(OUTPUT_FILE, "a", encoding="utf-8") as out,
    ):
        buf = b""
        while written < MAX_POSITIONS:
            chunk = reader.read(1 << 20)  # 1 MB at a time
            if not chunk:
                break
            buf += chunk

            while b"\n" in buf and written < MAX_POSITIONS:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                scanned += 1
                try:
                    pos = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ev = _best_eval(pos.get("evals", []))
                if ev is None:
                    continue

                cp = _pv_to_cp(ev.get("pvs", []))
                if cp is None:
                    continue

                # Lichess FENs have 4 fields; append halfmove/fullmove defaults
                # so chess.Board() in training.py can parse them.
                fen = pos["fen"].strip() + " 0 1"
                out.write(f"{fen},{cp}\n")
                written += 1

                if written % 100_000 == 0:
                    pct = written / MAX_POSITIONS * 100
                    print(f"  [{pct:5.1f}%]  scanned {scanned:,}  |  written {written:,}")

    print(f"\nDone. {written:,} positions → {OUTPUT_FILE}")
    if scanned > 0:
        print(f"Pass rate: {written/scanned*100:.1f}%  ({scanned - written:,} skipped, depth < {MIN_DEPTH})")


if __name__ == "__main__":
    main()
