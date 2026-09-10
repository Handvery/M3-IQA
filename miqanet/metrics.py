"""IQA correlation metrics."""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def correlation_metrics(predictions, targets) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.size != targets.size or predictions.size < 2:
        raise ValueError("PLCC/SRCC require equally sized arrays with at least 2 values")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("PLCC/SRCC require finite predictions and targets")
    if np.ptp(predictions) == 0 or np.ptp(targets) == 0:
        raise ValueError("PLCC/SRCC are undefined for constant predictions or targets")
    return {
        "plcc": float(pearsonr(predictions, targets).statistic),
        "srcc": float(spearmanr(predictions, targets).statistic),
    }


__all__ = ["correlation_metrics"]
