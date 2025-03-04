# utils/metrics.py
from typing import Dict

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    targets: np.ndarray, outputs: np.ndarray, metrics: List[str]
) -> Dict[str, float]:
    results = {}
    for metric in metrics:
        if metric == "r2":
            results["r2"] = r2_score(targets, outputs)
        elif metric == "rmse":
            results["rmse"] = np.sqrt(mean_squared_error(targets, outputs))
        elif metric == "mae":
            results["mae"] = mean_absolute_error(targets, outputs)
        elif metric == "pearson":
            results["pearson"] = pearsonr(targets.flatten(), outputs.flatten())[0]
    return results
