"""Print an EDA report for the freMTPL2 merged frame (Phase 3).

Output goes to stdout AND to reports/phase3_eda.txt. No plots — this is
the evidence for the outlier policy in reports/phase3_outliers.md.
"""

from __future__ import annotations

import pandas as pd

from src.features.eda import (
    concentration_stats,
    exposure_weighted_frequency,
    gini_normalized,
    quantile_summary,
)
from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

MERGED = PROCESSED_DIR / "merged.parquet"
OUT = REPORTS_DIR / "phase3_eda.txt"


def main() -> None:
    df = pd.read_parquet(MERGED)
    sev = df.loc[df["ClaimNb_sev"] > 0, "ClaimAmount_sum"].dropna()

    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    p("=" * 70)
    p("PHASE 3 — EDA REPORT (freMTPL2 merged)")
    p("=" * 70)
    p()

    p("--- Dataset shape ---")
    p(f"Policies:                {len(df):,}")
    p(f"Columns:                 {df.shape[1]}")
    p(f"Total exposure (years):  {df['Exposure'].sum():,.2f}")
    p()

    p("--- Frequency ---")
    p(f"Total claims (ClaimNb sum):   {int(df['ClaimNb'].sum()):,}")
    p(f"Policies with ClaimNb > 0:    {int((df['ClaimNb'] > 0).sum()):,}")
    p(f"Frequency per exposure year:  {exposure_weighted_frequency(df):.6f}")
    p()

    p("--- Severity (per-policy ClaimAmount_sum, ClaimNb_sev > 0) ---")
    p(f"Policies with severity:       {len(sev):,}")
    qs = quantile_summary(sev)
    for k, v in qs.items():
        if isinstance(v, float):
            p(f"  {k:>6s}: {v:>15,.2f}")
        else:
            p(f"  {k:>6s}: {v:>15,}")
    p()

    p("--- Concentration (severity) ---")
    p(f"Gini normalized:              {gini_normalized(sev):.4f}")
    cs = concentration_stats(sev, top_fracs=(0.10, 0.01, 0.001))
    for k, v in cs.items():
        p(f"  {k:>12s}: {v * 100:>6.2f}% of total severity")
    p()

    p("--- Tail threshold candidates (severity per policy) ---")
    for q in (0.99, 0.995, 0.999):
        threshold = sev.quantile(q)
        n_above = int((sev > threshold).sum())
        p(f"  p{q * 100:5.1f}: {threshold:>12,.2f} EUR   ({n_above:,} policies above)")
    p()

    p("--- Severity availability ---")
    p(f"Policies with ClaimNb_sev = 0:        {int((df['ClaimNb_sev'] == 0).sum()):,}")
    p(
        f"   of which ClaimNb > 0 (no sev row): {int(((df['ClaimNb_sev'] == 0) & (df['ClaimNb'] > 0)).sum()):,}"
    )
    p()

    p("--- Top-10 severity policies ---")
    top10 = df.nlargest(10, "ClaimAmount_sum")[
        ["IDpol", "ClaimNb", "ClaimNb_sev", "ClaimAmount_sum"]
    ]
    p(top10.to_string(index=False))
    p()
    p("=" * 70)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
