# TigerEngine NNUE — Training Roadmap

## Current State

| Property | Value |
|---|---|
| Feature set | HalfKP (40,960 features per side) |
| Architecture | 40960 → 1024 (×2 concat) → 8 → 32 → 1 |
| Activation | Clipped ReLU [0, 1.0] float / [0, QA] integer |
| Loss | MSE on sigmoid WDL targets |
| Output scale | QO = 410 cp |
| Quantization | FT int16, hidden int8, biases int32, engine divides by QB²=4096 |
| Best benchmark | MAE ≈ 24.6 cp (1024/128 architecture, 2M positions) |

---

## ~~Phase 1 — Retrain with corrected architecture~~ ✓ Done

Completed fixes:
- Clamp corrected from `[0, 255]` (plain ReLU) to `[0, 1.0]` (true CReLU)
- `l2_bias` quantization corrected (`QB²×QO` not `QB×QO`)
- HalfKP opponent piece offset fixed (`+5` not `+6`)
- Architecture updated to 3 hidden layers matching Stockfish: `1024×2 → 8 → 32 → 1`
- ParsePGNs.py eval unit fixed (centipawns, not pawns)

---

## ~~Phase 4a — Reduce hidden layer to 1024~~ ✓ Done

`HIDDEN_LAYER_SIZE = 1024` already set.

---

## ~~Phase 4b — HalfKP features~~ ✓ Done

`board_to_features()` now encodes `king_sq × piece_sq × piece_type` (40,960 inputs).
`ChessNNUE.ft = nn.Linear(40960, HIDDEN_LAYER_SIZE)` updated.

---

## Phase 2 — More training data (do this now)

Current: 2M positions. Train loss 0.0013, test loss 0.0044 — 3.3× overfitting gap.
Target: 5–50M positions to close the gap.

### 2a. Increase MAX_POSITIONS in ParsePGNs.py

```python
MAX_POSITIONS = 10_000_000   # or higher if disk/time allows
```

Add more PGN files to cover more games. Each game yields ~5 positions
(`POSITIONS_PER_GAME = 5`), so 10M positions ≈ 2M games.

### 2b. Use a pre-labelled dataset (fastest)

Large, high-quality datasets already labelled with Stockfish evaluations:
- **Lichess database** — `lichess.org/api` has game exports with eval annotations.
- **fishpack / Stockfish training data** — the Stockfish team releases their
  training positions publicly on HuggingFace (`Ilyabarigou/stockfish-dataset`).
  These are already quiet positions, already in centipawns.

### 2c. Filter to quiet positions

Stockfish evaluations are most reliable when the position is quiet.
ParsePGNs.py already skips the opening (ply < 10). Optionally also skip checks:

```python
if board.is_check():
    continue
```

### 2d. Balance the evaluation distribution

If your dataset is heavily skewed toward draws (0 cp), the network will learn
to predict draws for everything. Check `histogram.py` output — ideally the WDL
distribution has a spread across [0.2, 0.8], not a spike at 0.5.

---

## Phase 3 — Mixed loss function

Pure WDL loss (sigmoid targets) is good at learning win/draw/loss but can lose
fine-grained centipawn precision. A blended loss improves accuracy:

```python
# In train_loop, replace the single loss line with:
wdl_loss  = loss_fn(pred_norm, target)                     # sigmoid space
cp_loss   = loss_fn(pred_raw / QO, raw_eval / QO)          # centipawn space (normalised)
loss = 0.7 * wdl_loss + 0.3 * cp_loss
```

The 0.7/0.3 split is a starting point — tune it based on your validation MAE.

You can also mix in game results (lambda interpolation):
```python
# game_result in {0.0, 0.5, 1.0}
target = lambda_ * sigmoid(cp / QO) + (1 - lambda_) * game_result
```
Stockfish uses lambda ≈ 0.5–0.8 depending on training stage.

---

## Phase 5 — Engine integration

### Binary file layout (written by exportweights.py)

```
ft.weight  [40960 × 1024]  int16
l1.weight  [2048 × 8]      int8
l2.weight  [8 × 32]        int8
l3.weight  [32 × 1]        int8
ft.bias    [1024]           int16
l1.bias    [8]              int32
l2.bias    [32]             int32
l3.bias    [1]              int32
```

### Integer inference in the engine

```
acc_q  = ft_weight_q[active_features] + ft_bias_q   // sum selected columns
h1_q   = clamp(acc_q, 0, QA)                        // QA = 255
z1_q   = l1_weight_q @ h1_q + l1_bias_q             // int8 × int16 → int32
h2_q   = clamp(z1_q >> 6, 0, QA)                    // scale down by QB=64
z2_q   = l2_weight_q @ h2_q + l2_bias_q
h3_q   = clamp(z2_q >> 6, 0, QA)
out_q  = l3_weight_q @ h3_q + l3_bias_q
cp     = out_q >> 12                                 // divide by QB² = 4096
```

### Blending with handcrafted eval

```cpp
int nnue_cp  = nnue_evaluate(pos);
int hce_cp   = handcrafted_evaluate(pos);

// Taper: full NNUE in endgame, blend in midgame
int phase    = game_phase(pos);   // 0=endgame, 24=opening
int eval     = (nnue_cp * (24 - phase) + hce_cp * phase) / 24;
```

Tune the blend weight with a self-play tournament.

---

## Milestone Summary

| Phase | Change | Status |
|---|---|---|
| 1 | Fixed clamp, quantization, HalfKP offset, 3-layer arch | ✓ Done |
| 2 | 5M+ quiet positions, centipawn labels | In progress — need more data |
| 3 | Mixed WDL + CP loss / game result lambda | Not started |
| 4a | HIDDEN_LAYER_SIZE 2048 → 1024 | ✓ Done |
| 4b | HalfKP features (40960 inputs) | ✓ Done |
| 5 | Engine integration + blend tuning | Not started |
