# TigerEngine NNUE — Training Roadmap for future changes

## Current State

| Property | Value |
|---|---|
| Feature set | Piece/square (768 features) — no king position |
| Architecture | 768 → 2048 (×2 concat) → 128 → 1 |
| Activation | Clipped ReLU [0, 1.0] float / [0, QA] integer |
| Loss | Huber on sigmoid WDL targets |
| Output scale | QO = 410 cp |
| Quantization | FT int16, hidden int8, biases int32, engine divides by QB²=4096 |

---

## Phase 1 — Retrain with current architecture (do this now)

The clamp was changed from `[0, 255]` (effectively plain ReLU) to `[0, 1.0]`
(true CReLU) and the `l2_bias` quantization was corrected (`QB²×QO` not `QB×QO`).
Any previous weights are no longer compatible — retrain from scratch.

**Checklist before running:**
- [ ] Verify `Data/chessData.csv` evaluations are in **centipawns** (not pawns).
      If you generated the data with the old `ParsePGNs.py`, the values were
      divided by 100 — you'll need to regenerate or multiply all evals by 100.
- [ ] Set `DATASET_SAMPLE_SIZE = None` in `training.py` to use the full dataset.
- [ ] Run `python histogram.py` first to inspect your eval distribution.
      Look for a roughly bell-shaped WDL distribution centred at 0.5.

**Quick wins during this run:**
- Add weight decay to reduce quantization error from large weights:
  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
  ```
- Switch from `StepLR` to cosine annealing for a smoother decay:
  ```python
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
  ```

---

## Phase 2 — Better training data

The most impactful thing after a correct training setup is more and cleaner data.

### 2a. Use a pre-labelled dataset (fastest)

Large, high-quality datasets already labelled with Stockfish evaluations:
- **Lichess database** — `lichess.org/api` has game exports with eval annotations.
- **fishpack / Stockfish training data** — the Stockfish team releases their
  training positions publicly on HuggingFace (`Ilyabarigou/stockfish-dataset`).
  These are already quiet positions, already in centipawns.

Target: **5–50 million positions** for this feature set.

### 2b. Filter to quiet positions

Stockfish evaluations are most reliable when the position is "quiet" (no hanging
pieces, not in the middle of a tactical sequence). Add a quietness filter to
`ParsePGNs.py`:

```python
# after board.push(move) in the sampling loop:
if board.is_check():
    continue
# Optional: skip if any piece is en prise (requires a SEE or quick eval delta)
```

### 2c. Balance the evaluation distribution

If your dataset is heavily skewed toward draws (0 cp), the network will learn
to predict draws for everything. Check `histogram.py` output — ideally the WDL
distribution has a spread across [0.2, 0.8], not a spike at 0.5.

Sample strategies:
- Undersample positions with |eval| < 50 cp
- Use opening books to seed more diverse game positions

### 2d. Fix the eval unit bug (if you used ParsePGNs.py)

The old `ParsePGNs.py` wrote evaluations in **pawns** (`score / 100.0`) while
`training.py` expects **centipawns**. If your CSV has values like `1.5` instead
of `150`, every eval is 100× too small. The file has been fixed — regenerate
your data with the new version.

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

---

## Phase 4 — Architecture improvements

### 4a. Reduce hidden layer size to 1024

`HIDDEN_LAYER_SIZE = 2048` with 768 input features is oversized. The 2×2048=4096
concatenated accumulator takes up 8 KB of int16 memory per thread, which hurts
cache performance in the engine. `1024` is a better trade-off:

- Change `HIDDEN_LAYER_SIZE = 1024` in `training.py`
- Update the engine to match
- Inference is ~4× faster; minimal quality loss at this feature set size

### 4b. King-relative features (HalfKP) — biggest long-term gain

The current encoding (`piece_type × square = 768`) does not know where the
kings are. A rook on e4 looks the same regardless of king safety. This is the
single biggest limitation of the current setup.

**HalfKP** encodes `king_square × piece_square × piece_type`:
- 64 king squares × 64 piece squares × 10 piece types (no kings) = 40,960 per side
- Input is still sparse (~30 active features per side) so it's fast to update
- The feature transformer grows from 768 → 40,960 inputs but the accumulator
  size stays the same (just the sparse lookup indices change)

This requires:
1. Rewriting `board_to_features()` to include king square in the index
2. Updating `ChessNNUE.ft = nn.Linear(40960, HIDDEN_LAYER_SIZE)`
3. Updating the engine's feature extraction to match
4. Retraining from scratch (different input dimensionality)

HalfKP is what gives Stockfish NNUE its king safety understanding. Expected
improvement: significant (50–150 Elo in a tuned engine).

---

## Phase 5 — Engine integration

### Using the quantized weights

The binary `.nnue` file layout (written by `exportweights.py`):
```
ft.weight  [768 × HIDDEN]   int16  (or [40960 × HIDDEN] for HalfKP)
l1.weight  [2*HIDDEN × 128] int8
l2.weight  [128 × 1]        int8
ft.bias    [HIDDEN]          int16
l1.bias    [128]             int32
l2.bias    [1]               int32
```

Integer inference in the engine:
```
acc_q  = ft_weight_q[active_features] + ft_bias_q  // sum selected columns
h1_q   = clamp(acc_q, 0, QA)                       // QA = 255
z1_q   = l1_weight_q @ h1_q + l1_bias_q            // int8 × int16 → int32
h2_q   = clamp(z1_q, 0, QA * QB)                   // QB = 64
out_q  = l2_weight_q @ h2_q + l2_bias_q            // int8 × int32 → int32
cp     = out_q >> 12                                // divide by QB² = 4096
```

### Blending with handcrafted eval

Since you're subsidising rather than replacing your eval function:

```cpp
int nnue_cp  = nnue_evaluate(pos) >> 12;
int hce_cp   = handcrafted_evaluate(pos);

// Taper: full NNUE in endgame, blend in midgame
int phase    = game_phase(pos);   // 0=endgame, 24=opening
int eval     = (nnue_cp * (24 - phase) + hce_cp * phase) / 24;
```

Or a fixed blend if phase tapering is complex:
```cpp
int eval = (nnue_cp * 60 + hce_cp * 40) / 100;
```

Tune the blend weight with a self-play tournament.

---

## Milestone Summary

| Phase | Change | Expected impact |
|---|---|---|
| 1 | Retrain with fixed clamp + quantization | Correct integer inference |
| 2 | 5M+ quiet positions, centipawn labels | Significant quality improvement |
| 3 | Mixed WDL + CP loss | Better centipawn precision |
| 4a | HIDDEN_LAYER_SIZE 2048 → 1024 | 4× faster engine inference |
| 4b | HalfKP features (40960 inputs) | King safety, largest Elo gain |
| 5 | Engine integration + blend tuning | Real Elo improvement in TigerEngine |
