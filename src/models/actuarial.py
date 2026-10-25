"""Actuarial metrics: lift by decile, Gini, Lorenz (Phase 9).

Standard insurance-model diagnostics beyond deviance:
- lift_by_decile: does the model rank policies correctly? Compare
  actual/expected ratio per prediction decile.
- gini_normalized_predictions: ranking quality. 0 = random, 1 = perfect.
- lorenz_curve_predictions: same data as Gini, for plotting.

Uses np.trapz (numpy 1.x). np.trapezoid only exists in numpy 2.0+;
project pins numpy<2.0 in pyproject.toml.
"""

from __future__ import annotations

import numpy as np


def lift_by_decile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_deciles: int = 10,
) -> list[dict]:
    """Sort by predicted descending, group into n_deciles bins.

    Returns list of {decile, n, actual, predicted, ae_ratio}. Decile 1
    is the highest-predicted 10%.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if len(y_true) < n_deciles:
        raise ValueError(f"Need at least {n_deciles} rows")

    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    p_sorted = y_pred[order]

    out: list[dict] = []
    n = len(y_sorted)
    for i in range(n_deciles):
        start = i * n // n_deciles
        end = (i + 1) * n // n_deciles if i < n_deciles - 1 else n
        actual = float(y_sorted[start:end].sum())
        predicted = float(p_sorted[start:end].sum())
        out.append(
            {
                "decile": i + 1,
                "n": end - start,
                "actual": actual,
                "predicted": predicted,
                "ae_ratio": (actual / predicted) if predicted > 0 else 0.0,
            }
        )
    return out


def gini_normalized_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ranking Gini for pure premium.

    Sort policies by prediction descending. Cumulative actual loss should
    jump fast for a good model. Gini = 2 * AUC - 1 where AUC is area
    under the Lorenz curve (pop_share vs cumulative loss share).

    0 = random ranking, 1 = perfect ranking.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    total = y_true.sum()
    if total <= 0:
        return 0.0

    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    cum = np.cumsum(y_sorted) / total
    n = len(y_sorted)

    pop_curve = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    loss_curve = np.concatenate([[0.0], cum])

    auc = float(np.trapz(loss_curve, pop_curve))
    return float(2.0 * auc - 1.0)


def lorenz_curve_predictions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pop_share, loss_share) for plotting."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total = y_true.sum()
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    if total > 0:
        cum = np.cumsum(y_sorted) / total
    else:
        cum = np.zeros(len(y_sorted))
    n = len(y_sorted)
    pop = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    loss = np.concatenate([[0.0], cum])
    return pop, loss
