# Phase 3 — Outlier policy for ClaimAmount_sum

## Context

Severity (per-policy total claim amount) is heavy-tailed: a small number
of policies account for a disproportionate share of total claim cost.
This is the defining challenge of insurance severity modeling.

The exact numbers (Gini, quantiles, concentration, top-10) are printed by
`make eda` and saved to `reports/phase3_eda.txt`. This document decides
the policy; the file is the numeric evidence.

## Three candidate policies

### Policy A — No capping (raw target)

- **Pro**: keeps all signal. The tail is real cost, not noise.
- **Con**: a single catastrophic claim (500k EUR+) dominates gradients
  during training. GBM splits become unstable; RMSE explodes even when
  MAE stays reasonable. Gamma objective is particularly sensitive.
- **When to use**: Tweedie direct with proper deviance loss (Phase 6).
  Tweedie's log-link and compound Poisson-Gamma structure handle the tail
  natively.

### Policy B — Cap at p99.5 of per-policy severity

- **Pro**: preserves 99.5% of observations unmodified. Caps only the
  extreme tail, stabilizing gradient-based training without discarding
  the signal for 99.5% of policies.
- **Con**: ignores the top-0.5% tail signal. Requires documenting the cap
  value in the Model Card (regulatory traceability).
- **When to use**: GBM with Gamma objective (Phase 6), quantile regression
  (Phase 8), conformal prediction (Phase 8).

### Policy C — Separate catastrophic model

- **Pro**: models catastrophic claims (top 0.1%) as a separate rare-event
  problem; main model sees only "normal" claims.
- **Con**: complexity explodes (two models + threshold selection +
  combination logic). Critically, the catastrophic model would train on
  the top 0.1% of ~25,000 policies = ~25 rows. Not trainable.
- **When to use**: only when the top-tail has 1000+ observations. Not the
  case for freMTPL2.

## Decision

**Chosen policy: B (cap at p99.5) for the primary Gamma-objective GBM
and the conformal prediction pipeline (Phases 6, 8). Complement with a
Tweedie-direct model on the raw target (Phase 6) to preserve the tail
signal — this gives us both a "stabilized Gamma" view and a "raw tail
Tweedie" view to compare in Phase 9.**

**Rationale**:

1. Policy C is structurally infeasible: the top 0.1% of ~25,000 severity
   policies is roughly 25 rows. No learning algorithm trains on 25 rows.
2. Gamma objective (our primary severity GBM) is sensitive to extreme
   losses. Capping at p99.5 is the standard stabilizing treatment in
   actuarial practice on freMTPL2 (Noll, Salzmann, Wuthrich 2016 — the
   dataset's reference paper).
3. Keeping the raw target available via Tweedie direct means we never
   lose the catastrophic signal — we just don't let it dominate the
   Gamma model's gradients.

**Exact cap value**: see `reports/phase3_eda.txt`, section "Tail threshold
candidates". Phase 4 (feature engineering) will apply this cap to a NEW
column `ClaimAmount_capped` used by Gamma models. The merged frame's
`ClaimAmount_sum` stays untouched (raw values preserved for Tweedie).

## Impact on metrics — to be measured in Phase 6

- **MAE (capped target)**: expected to decrease vs. raw.
- **RMSE (capped target)**: expected to decrease hugely vs. raw, because
  RMSE is dominated by the tail. This is why RMSE is NOT our primary
  metric — see `docs/CONTRACT.md` (heavy-tail target section).
- **Gamma deviance (capped target)**: primary metric for the Gamma
  objective. Compare with/without cap in Phase 6.
- **Actual/Expected ratio by decile**: computed on raw target, so
  unaffected by capping. Reported in Phase 9.

## Deferred decisions

- **Frequency outliers** (ClaimNb > 5): NOT capped. Poisson/Tweedie
  handle the tail. Revisit in Phase 5 baselines.
- **Exposure outliers**: none observed. Max ~2 years = multi-year
  contracts. Kept as-is.
- **Zero-severity policies** (9,116 claims without sev record): frequency
  model uses them, severity model cannot (no target). Tweedie handles
  them natively. Documented in `reports/phase2_summary.md`.
