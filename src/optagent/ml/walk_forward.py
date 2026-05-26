"""Walk-forward (expanding-window) validation for the direction model.

In-sample accuracy is approximately useless — it's an upper bound limited
only by the model's ability to memorise. The audit ledger keeps it for
diagnostic purposes but the LLM and the user should see an OOS metric.

`walk_forward_eval()` does K folds where each fold uses
`[start..train_end] -> train` and `[train_end+gap..train_end+gap+horizon]`
-> validation. The gap (default = `horizon`) prevents the validation
window from peeking at training labels.

Output keys:
    n_folds, train_rows_per_fold, val_rows_per_fold,
    oos_accuracy_mean, oos_accuracy_std,
    oos_log_loss_mean,
    direction_hit_rate,  # accuracy of the "up vs down" prediction
    sample_predictions   # last fold's predictions for inspection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, build_features, build_target


MIN_TRAIN_FOR_FOLD = 200
DEFAULT_K_FOLDS = 5
DEFAULT_GAP = 5  # match the model's 5-day horizon


@dataclass(frozen=True)
class WalkForwardResult:
    n_folds: int
    oos_accuracy_mean: float
    oos_accuracy_std: float
    oos_log_loss_mean: float
    direction_hit_rate: float
    train_rows_per_fold: list[int]
    val_rows_per_fold: list[int]
    n_oos_samples: int
    wilson_ci_lower: float  # 95% Wilson CI lower bound on pooled accuracy
    wilson_ci_upper: float
    class_baseline_accuracy: float  # majority-class predictor's accuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "n_oos_samples": self.n_oos_samples,
            "oos_accuracy_mean": round(self.oos_accuracy_mean, 4),
            "oos_accuracy_std": round(self.oos_accuracy_std, 4),
            "oos_log_loss_mean": round(self.oos_log_loss_mean, 4),
            "direction_hit_rate": round(self.direction_hit_rate, 4),
            "wilson_ci_lower": round(self.wilson_ci_lower, 4),
            "wilson_ci_upper": round(self.wilson_ci_upper, 4),
            "class_baseline_accuracy": round(self.class_baseline_accuracy, 4),
            "train_rows_per_fold": self.train_rows_per_fold,
            "val_rows_per_fold": self.val_rows_per_fold,
        }


def _wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — robust for binomial proportions even at small n.

    Returns (lower, upper) clamped to [0, 1]. n=0 returns (0, 1) (no info).
    """

    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _fit_predict(X_train, y_train, X_val) -> tuple[np.ndarray, np.ndarray]:
    """Train a fresh classifier and return (predict, predict_proba_up).

    Isolated here so callers can swap the estimator easily.
    """

    from sklearn.ensemble import GradientBoostingClassifier  # noqa: WPS433

    model = GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)
    classes = list(getattr(model, "classes_", [0, 1]))
    idx_up = classes.index(1) if 1 in classes else 1
    return preds.astype(int), proba[:, idx_up]


def _safe_log_loss(y_true: np.ndarray, p_up: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p_up, eps, 1.0 - eps)
    y = y_true.astype(float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def walk_forward_eval(
    ohlcv: pd.DataFrame,
    *,
    k_folds: int = DEFAULT_K_FOLDS,
    gap: int = DEFAULT_GAP,
    horizon: int = 5,
) -> WalkForwardResult | None:
    """Run a K-fold expanding-window evaluation.

    Returns `None` when there isn't enough data for any meaningful fold
    (callers should fall back to in-sample only, with a low-credibility
    annotation in the audit ledger).

    `gap >= horizon` is enforced fail-closed (Codex R4 finding): the
    validation block must start at or after `train_end + horizon` so that
    the last training label cannot leak forward into the validation set.
    """

    if gap < horizon:
        raise ValueError(
            f"walk_forward_eval: gap ({gap}) must be >= horizon ({horizon}) to "
            "prevent label leakage between train and validation"
        )

    features = build_features(ohlcv)
    target = build_target(ohlcv["Close"].astype(float), horizon=horizon)
    df = pd.concat([features, target.rename("y")], axis=1).dropna()
    n = len(df)
    if n < MIN_TRAIN_FOR_FOLD + DEFAULT_GAP + 20:
        return None

    # Place fold split points uniformly between MIN_TRAIN_FOR_FOLD and n - gap - 20.
    last_train_end = n - gap - 20
    if last_train_end <= MIN_TRAIN_FOR_FOLD:
        return None
    split_points = np.linspace(MIN_TRAIN_FOR_FOLD, last_train_end, num=k_folds, dtype=int)
    accuracies: list[float] = []
    losses: list[float] = []
    direction_correct = 0
    direction_total = 0
    train_rows: list[int] = []
    val_rows: list[int] = []

    X = df[list(FEATURE_NAMES)].to_numpy()
    y = df["y"].astype(int).to_numpy()

    for split in split_points:
        train_end = int(split)
        val_start = train_end + gap
        val_end = min(val_start + 20, n)
        if val_start >= val_end:
            continue
        if y[:train_end].sum() in (0, train_end):
            # All-one-class folds skip — sklearn raises on degenerate targets.
            continue
        preds, p_up = _fit_predict(X[:train_end], y[:train_end], X[val_start:val_end])
        true_y = y[val_start:val_end]
        if len(true_y) == 0:
            continue
        accuracies.append(float((preds == true_y).mean()))
        losses.append(_safe_log_loss(true_y, p_up))
        direction_correct += int((preds == true_y).sum())
        direction_total += int(len(true_y))
        train_rows.append(train_end)
        val_rows.append(int(val_end - val_start))

    if not accuracies:
        return None
    pooled_acc = (
        float(direction_correct / direction_total) if direction_total else 0.0
    )
    lo, hi = _wilson_ci(pooled_acc, direction_total)
    # Majority-class baseline across all OOS samples — the accuracy a naive
    # "always predict the dominant class" model would achieve.
    n_pos = int(y[: split_points[-1] + 20].sum()) if len(y) else 0
    # Better: compute on all validation labels actually seen.
    # Re-derive baseline accuracy honestly by counting class frequencies in
    # the validation windows used above.
    val_y_collected: list[int] = []
    for split in split_points:
        train_end = int(split)
        val_start = train_end + gap
        val_end = min(val_start + 20, n)
        if val_start >= val_end:
            continue
        val_y_collected.extend(y[val_start:val_end].tolist())
    if val_y_collected:
        n_ones = sum(val_y_collected)
        n_total = len(val_y_collected)
        class_baseline = max(n_ones / n_total, 1.0 - n_ones / n_total)
    else:
        class_baseline = 0.5
    return WalkForwardResult(
        n_folds=len(accuracies),
        oos_accuracy_mean=float(np.mean(accuracies)),
        oos_accuracy_std=float(np.std(accuracies, ddof=0)),
        oos_log_loss_mean=float(np.mean(losses)),
        direction_hit_rate=pooled_acc,
        train_rows_per_fold=train_rows,
        val_rows_per_fold=val_rows,
        n_oos_samples=direction_total,
        wilson_ci_lower=lo,
        wilson_ci_upper=hi,
        class_baseline_accuracy=float(class_baseline),
    )
