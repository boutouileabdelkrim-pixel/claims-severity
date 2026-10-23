# Phase 4 — Feature engineering + splits

## Split strategy

- Unit: `IDpol` (one policy per row, already unique after Phase 2 merge)
- Stratification key: `ClaimNb > 0` (frequency positive class, ~5%)
- Sizes: 70 / 15 / 15 (from `SETTINGS`)
- Seed: 42 (from `SETTINGS`)
- One assignment per IDpol, shared across frequency and severity models
  so no policy appears in two different splits
- Reproducible: same seed + same merged.parquet = identical split

Exact row counts (train/val/test, freq_pos, sev_pos) are printed by
`make features` and pasted into `reports/phase4_features.txt`.

## Feature engineering

8 derived columns added, no target encoding, no frequency encoding.

| Column | Type | Formula | Why |
|--------|------|---------|-----|
| `VehGas_diesel` | int8 | `VehGas == 'Diesel'` | Binary, avoids one-hot |
| `Area_ord` | int8 | A→0, B→1, ..., F→5 | Area is ordinal by density |
| `BonusMalus_log` | float32 | `log(BonusMalus)` | BonusMalus ranges 50-230, right-skewed |
| `Density_log` | float32 | `log1p(Density)` | Density ranges 0-27k, huge scale |
| `VehAge_zero` | int8 | `VehAge == 0` | Brand-new vehicle risk pattern |
| `VehAge_over_20` | int8 | `VehAge > 20` | Old-vehicle risk pattern |
| `BonusMalus_gt100` | int8 | `BonusMalus > 100` | Penalty flag (malus > 1.0) |
| `DrivAge_bucket` | int8 | cut(15-25, 25-35, 35-55, 55-75, 75-101) | Non-linear age effect |

Preserved raw columns: `IDpol` (traceability, dropped as model input in
Phase 5), `ClaimNb`, `Exposure`, `ClaimAmount_sum`, `ClaimNb_sev`, all
original features.

**Explicitly NOT done**:
- Target encoding of `Region` / `VehBrand` — would leak the target.
  Trees handle categoricals natively (LightGBM) or via one-hot (Phase 5).
- Frequency encoding of `Region` — same leak risk.
- Interaction features (`VehAge * BonusMalus`) — deferred to Phase 6
  if SHAP shows they matter.

## Cap threshold (p99.5)

- Computed on **train severity only** (`split == 'train' AND ClaimNb_sev > 0`)
- Value: exact number in `reports/phase4_features.txt` (log line from `make features`)
- Applied as a **constant** to all splits (train/val/test)
- New column: `ClaimAmount_capped = min(ClaimAmount_sum, threshold)`
- Raw `ClaimAmount_sum` **preserved** for Tweedie-direct (Phase 6)

**Why on train only**: computing on full severity leaks the val/test
distribution into the cap choice, making val/test optimism artificially
higher. Standard practice: fit any preprocessing constant on train,
freeze it, apply everywhere.

## Output files

| File | Rows | Cols | Size |
|------|------|------|------|
| data/features/train.parquet | ~474,600 | 23 | ~7 MB |
| data/features/val.parquet   | ~101,700 | 23 | ~1.5 MB |
| data/features/test.parquet  | ~101,700 | 23 | ~1.5 MB |

Exact numbers are in `reports/phase4_features.txt` from `make features`.

Column count: 14 (merged) + 8 (new) + 1 (ClaimAmount_capped) = 23. `split` column is dropped
before saving (one file per split).

## Deferred to Phase 5

- One-hot encoding of `Region` and `VehBrand` for sklearn GLMs.
- Feature selection: drop `IDpol`, `ClaimNb_sev` from model inputs.
- Numeric float32 casts for reduced memory during GLM fit.
- `Exposure` handling: offset (Poisson) vs `sample_weight` — decided
  per model family.

## Correction note on Phase 3 docstring

`src/features/eda.py::gini_normalized` docstring says "freMTPL2 severity,
high values (0.7+) are expected". Actual Gini on our severity subset is
**0.6475** — still a strong heavy tail, but not 0.7+. The docstring is
slightly optimistic; the code is correct. Fixed at next natural edit
of that file.
