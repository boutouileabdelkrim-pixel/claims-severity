# Phase 2 — Acquisition + Cleaning + Merge

## Inputs (OpenML)

| Dataset | OpenML id | Rows | Cols | Sum ClaimAmount |
|---------|-----------|------|------|-----------------|
| freMTPL2freq | 41214 | 678,013 | 12 | — |
| freMTPL2sev  | 41215 |  26,639 |  2 | 60,697,931.00 |

## Pipeline

    data/raw/{freq,sev}.parquet
      -> validate_raw (schemas)
      -> clean_freq  (IDpol float->int64, VehGas "'X'"->"X")
      -> clean_sev   (aggregate per IDpol: sum amount, count claims)
      -> validate_clean
      -> merge_freq_sev (LEFT JOIN on IDpol, ClaimNb_sev fillna 0)
      -> validate_merged
      -> data/processed/{freq,sev_agg,merged}.parquet

## Critical decision: aggregate sev BEFORE join

Naive join of raw freq x raw sev = 18,061,588,307 rows. Aggregating
sev per IDpol brings it back to 678,013 rows. Documented in
reports/phase2_recon.txt.

## Data quirks observed during Phase 2

Three separate facts, all documented.

### 1. Claims without severity record (9,116 policies)

freq ClaimNb > 0 but no matching sev row. Historical fact of freMTPL2
(Noll et al., 2016): the sev file was truncated. These policies get
ClaimNb_sev = 0 and ClaimAmount_sum = NaN. Frequency model uses them;
severity model cannot (no target). Tweedie direct handles them natively.

### 2. Orphan sev IDpols (6 policies)

6 IDpols appear in sev but not in freq (2220367, 2227533, 2262511,
2277846, 2282134, + 1). Their ClaimAmount_sum (total 788,714.50) is
dropped by the LEFT JOIN. Sum check:

- sev_agg ClaimAmount_sum total:  60,697,931.00
- merged ClaimAmount_sum total:   59,909,216.50
- difference:                        788,714.50 (the 6 orphans)

### 3. Multi-year contracts (Exposure > 1)

Exposure is in years and can exceed 1 (real max observed: 2.01).
Schema enforces Exposure > 0 only, not <= 1.

### 4. VehGas quirk

Raw values ship as "'Regular'" / "'Diesel'" (embedded single quotes).
Stripped in clean_freq.

## Outputs

| File | Rows | Cols | Size |
|------|------|------|------|
| data/processed/freq.parquet     | 678,013 | 12 | 7.11 MB |
| data/processed/sev_agg.parquet  |  24,950 |  3 | 0.25 MB |
| data/processed/merged.parquet   | 678,013 | 14 | 7.30 MB |

## Next phase

Phase 3 — EDA + outlier policy:

- Distribution of ClaimAmount (heavy tail; max 66 claims per IDpol)
- Decision: cap / log / separate extreme claims
- Distribution of ClaimNb vs Exposure
- Modeling strategy for the 9,116 claims-without-severity
- Visual sanity of merged frame before Phase 4 feature engineering
