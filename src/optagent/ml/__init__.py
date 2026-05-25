"""Per-ticker ML direction model (Alt-3 v0).

Lazily trains a small sklearn classifier on 2-year OHLCV history when first
queried for a ticker; caches the trained artifact to
`data/ml_cache/<TICKER>.pkl`. Subsequent queries within `RETRAIN_DAYS` hit
the cache. v0 surfaces the probability + class prediction as an
INFORMATIONAL signal in the screener summary; it does NOT gate the verdict
(the architecture lets a future round promote it to a gate once we have
real OOS validation).
"""

from .direction_model import (
    MLCacheError,
    MLDirectionAdapter,
    MLDirectionSignal,
    RETRAIN_DAYS,
)

__all__ = [
    "MLCacheError",
    "MLDirectionAdapter",
    "MLDirectionSignal",
    "RETRAIN_DAYS",
]
