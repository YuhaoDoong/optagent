"""Per-ticker direction model — lazy-train + cache.

`MLDirectionAdapter.signal(ticker)` returns an `MLDirectionSignal` (or
None) by either loading a fresh cached artifact or training a new model
from yfinance's daily history. v0 ships a small sklearn
`GradientBoostingClassifier` with no hyperparameter sweep — the goal is
to lock the API and the caching/refresh policy, not to beat any baseline.

The trained pickle includes: model, FEATURE_NAMES tuple, trained_at,
ticker, n_train_rows, accuracy_self (in-sample, low-credibility), and
classifier metadata so an audit consumer can confirm the model's lineage.

Privacy / safety:
  - `data/ml_cache/<TICKER>.pkl` is gitignored.
  - Ticker is normalised to upper-case and validated against
    `^[A-Z][A-Z0-9.\-]{0,9}$` BEFORE building the file path (no path
    injection).
  - The model output is informational only; it does NOT bypass the
    fail-closed validator.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, build_features, build_target
from .walk_forward import walk_forward_eval


RETRAIN_DAYS = 7
MIN_TRAIN_ROWS = 250
DEFAULT_HORIZON_DAYS = 5
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
DEFAULT_CACHE_DIR = Path("data/ml_cache")
MODEL_VERSION = "ml-direction-v0"


class MLCacheError(RuntimeError):
    """Raised when the cache directory cannot be read or a ticker is unsafe."""


@dataclass(frozen=True)
class MLDirectionSignal:
    """Light per-ticker direction signal.

    `prob_up` is in [0, 1]; `class_label` is "up" / "down" / "neutral"
    using ±0.05 around 0.5 for the neutral band. `feature_snapshot` is the
    final feature row that fed the prediction (audit ledger material).

    `oos_accuracy` is the walk-forward (expanding window) accuracy mean,
    `accuracy_self` is the much-less-credible in-sample upper bound; both
    are kept so the audit ledger can show both numbers.
    """

    ticker: str
    prob_up: float
    class_label: str
    trained_at: datetime
    n_train_rows: int
    accuracy_self: float
    model_version: str
    feature_snapshot: dict[str, float] = field(default_factory=dict)
    oos_accuracy: float | None = None
    oos_log_loss: float | None = None
    n_oos_folds: int | None = None
    credibility: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "prob_up": round(self.prob_up, 4),
            "class_label": self.class_label,
            "trained_at": self.trained_at.astimezone(timezone.utc).isoformat(),
            "n_train_rows": self.n_train_rows,
            "accuracy_self_in_sample": round(self.accuracy_self, 4),
            "oos_accuracy": (round(self.oos_accuracy, 4) if self.oos_accuracy is not None else None),
            "oos_log_loss": (round(self.oos_log_loss, 4) if self.oos_log_loss is not None else None),
            "n_oos_folds": self.n_oos_folds,
            "credibility": self.credibility,
            "model_version": self.model_version,
            "feature_snapshot": {k: round(v, 6) for k, v in self.feature_snapshot.items()},
        }


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not TICKER_RE.match(t):
        raise MLCacheError(f"unsafe_ticker_for_cache_path: {ticker!r}")
    return t


def _classify_prob(p: float) -> str:
    if p > 0.55:
        return "up"
    if p < 0.45:
        return "down"
    return "neutral"


def _train_model(features: pd.DataFrame, target: pd.Series) -> tuple[Any, float, int]:
    """Fit a small gradient-boosted classifier; return (model, in-sample acc, n_rows)."""

    from sklearn.ensemble import GradientBoostingClassifier  # noqa: WPS433

    df = pd.concat([features, target.rename("y")], axis=1).dropna()
    if len(df) < MIN_TRAIN_ROWS:
        raise MLCacheError(f"not_enough_training_rows: {len(df)} < {MIN_TRAIN_ROWS}")
    X = df[list(FEATURE_NAMES)].to_numpy()
    y = df["y"].astype(int).to_numpy()
    model = GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X, y)
    in_sample_acc = float((model.predict(X) == y).mean())
    return model, in_sample_acc, len(df)


def _predict(model: Any, latest_features: pd.Series) -> float:
    x = latest_features[list(FEATURE_NAMES)].to_numpy(dtype=float).reshape(1, -1)
    if not np.isfinite(x).all():
        raise MLCacheError("non_finite_features")
    proba = model.predict_proba(x)[0]
    # sklearn orders class probabilities by sorted class label; class=1 is "up".
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        idx_up = classes.index(1)
    else:
        idx_up = 1
    return float(proba[idx_up])


class MLDirectionAdapter:
    """Lazy-train + cache; produces an `MLDirectionSignal` per ticker."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        retrain_days: int = RETRAIN_DAYS,
        yf_module: Any | None = None,
    ) -> None:
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._retrain_days = retrain_days
        if yf_module is not None:
            self._yf = yf_module
        else:
            try:
                import yfinance as yf  # noqa: WPS433

                self._yf = yf
            except ImportError:
                self._yf = None

    def _cache_path(self, ticker: str) -> Path:
        t = _validate_ticker(ticker)
        return self._cache_dir / f"{t}.pkl"

    def _load(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                blob = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            return None
        if not isinstance(blob, dict) or "model" not in blob or "trained_at" not in blob:
            return None
        # Schema invalidation (Codex R4 finding): cache from an older model
        # version or different feature schema must be retrained.
        if blob.get("model_version") != MODEL_VERSION:
            return None
        cached_features = blob.get("feature_names")
        if cached_features is not None and list(cached_features) != list(FEATURE_NAMES):
            return None
        return blob

    def _save(self, path: Path, blob: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(blob, f)
        os.replace(tmp, path)

    def _is_stale(self, blob: dict[str, Any]) -> bool:
        trained_at = blob.get("trained_at")
        if not isinstance(trained_at, datetime):
            return True
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - trained_at
        return age >= timedelta(days=self._retrain_days)

    def _fetch_history(self, ticker: str) -> pd.DataFrame | None:
        if self._yf is None:
            return None
        try:
            tk = self._yf.Ticker(ticker)
            df = tk.history(period="2y", interval="1d", auto_adjust=False)
        except Exception:  # noqa: BLE001 - degrade silently
            return None
        if df is None or df.empty or len(df) < MIN_TRAIN_ROWS:
            return None
        return df

    def _build_signal(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        blob: dict[str, Any] | None,
    ) -> MLDirectionSignal | None:
        features = build_features(ohlcv)
        latest_row = features.iloc[-1]
        if latest_row.isna().any():
            return None

        if blob is None or self._is_stale(blob):
            target = build_target(ohlcv["Close"].astype(float), horizon=DEFAULT_HORIZON_DAYS)
            try:
                model, in_sample_acc, n_rows = _train_model(features, target)
            except MLCacheError:
                return None
            # Walk-forward eval on the SAME OHLCV — cheap (~1s) and gives the
            # ledger an honest OOS metric.
            wf = walk_forward_eval(ohlcv)
            blob = {
                "model": model,
                "trained_at": datetime.now(timezone.utc),
                "ticker": ticker,
                "n_train_rows": n_rows,
                "accuracy_self": in_sample_acc,
                "feature_names": list(FEATURE_NAMES),
                "model_version": MODEL_VERSION,
                "oos_accuracy": (wf.oos_accuracy_mean if wf is not None else None),
                "oos_log_loss": (wf.oos_log_loss_mean if wf is not None else None),
                "n_oos_folds": (wf.n_folds if wf is not None else None),
            }
            self._save(self._cache_path(ticker), blob)

        prob_up = _predict(blob["model"], latest_row)
        feature_snapshot = {
            k: float(latest_row[k]) for k in FEATURE_NAMES if math.isfinite(float(latest_row[k]))
        }
        oos_accuracy = blob.get("oos_accuracy")
        # Credibility annotation — keeps the ledger / LLM from over-trusting
        # the model. "high" requires >=3 OOS folds AND accuracy > 0.55.
        if oos_accuracy is None:
            credibility = "low"
        elif oos_accuracy > 0.55 and (blob.get("n_oos_folds") or 0) >= 3:
            credibility = "medium"  # still not strong evidence; just better than nothing
        else:
            credibility = "low"
        return MLDirectionSignal(
            ticker=ticker,
            prob_up=prob_up,
            class_label=_classify_prob(prob_up),
            trained_at=blob["trained_at"] if isinstance(blob["trained_at"], datetime) else datetime.now(timezone.utc),
            n_train_rows=int(blob["n_train_rows"]),
            accuracy_self=float(blob["accuracy_self"]),
            model_version=str(blob.get("model_version", MODEL_VERSION)),
            feature_snapshot=feature_snapshot,
            oos_accuracy=oos_accuracy,
            oos_log_loss=blob.get("oos_log_loss"),
            n_oos_folds=blob.get("n_oos_folds"),
            credibility=credibility,
        )

    def signal(self, ticker: str, *, ohlcv: pd.DataFrame | None = None) -> MLDirectionSignal | None:
        """Return the ML direction signal or None if it cannot be computed.

        Callers (orchestrator) typically pass the already-fetched OHLCV to
        avoid a duplicate yfinance call. When `ohlcv` is None the adapter
        fetches its own (longer) history for training.
        """

        try:
            ticker = _validate_ticker(ticker)
        except MLCacheError:
            return None

        blob = self._load(self._cache_path(ticker))
        if blob is not None and not self._is_stale(blob):
            # Fast path — use the cached model on whatever OHLCV we got.
            if ohlcv is None or ohlcv.empty:
                ohlcv = self._fetch_history(ticker)
            if ohlcv is None or ohlcv.empty:
                return None
            try:
                return self._build_signal(ticker, ohlcv, blob)
            except Exception:  # noqa: BLE001
                return None

        # Slow path — need to (re)train. We always pull 2-year history for
        # training regardless of what the caller provided.
        train_df = self._fetch_history(ticker)
        if train_df is None:
            return None
        try:
            return self._build_signal(ticker, train_df, None)
        except Exception:  # noqa: BLE001
            return None
