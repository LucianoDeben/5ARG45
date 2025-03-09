# utils/metrics.py
from typing import Dict, List, Optional, Any
import logging
import numpy as np
from scipy.stats import pearsonr
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

def compute_metrics(
    targets: np.ndarray, outputs: np.ndarray, metrics: List[str]
) -> Dict[str, float]:
    """
    Compute specified metrics between targets and outputs.
    
    Args:
        targets: Ground truth values
        outputs: Predicted values
        metrics: List of metric names to compute
        
    Returns:
        Dictionary of metric names and values
    """
    results = {}
    try:
        # Ensure inputs are valid
        if targets is None or outputs is None:
            logger.warning("Cannot compute metrics: targets or outputs is None")
            return results
            
        if len(targets) == 0 or len(outputs) == 0:
            logger.warning(f"Cannot compute metrics: empty data (targets: {len(targets)}, outputs: {len(outputs)})")
            return results
            
        if len(targets) != len(outputs):
            logger.error(f"Dimension mismatch: targets={len(targets)}, outputs={len(outputs)}")
            return results
            
        # Flatten arrays to ensure 1D
        targets_flat = targets.flatten()
        outputs_flat = outputs.flatten()
            
        # Check for NaN values
        valid_mask = ~np.isnan(targets_flat) & ~np.isnan(outputs_flat)
        if np.sum(valid_mask) == 0:
            logger.warning("All data points contain NaN values, cannot compute metrics")
            return results
            
        targets_clean = targets_flat[valid_mask]
        outputs_clean = outputs_flat[valid_mask]
        
        # Compute requested metrics
        for metric in metrics:
            try:
                if metric.lower() == "r2":
                    results["r2"] = float(r2_score(targets_clean, outputs_clean))
                elif metric.lower() == "rmse":
                    results["rmse"] = float(np.sqrt(mean_squared_error(targets_clean, outputs_clean)))
                elif metric.lower() == "mae":
                    results["mae"] = float(mean_absolute_error(targets_clean, outputs_clean))
                elif metric.lower() == "pearson":
                    results["pearson"] = float(pearsonr(targets_clean, outputs_clean)[0])
                elif metric.lower() == "mse":
                    results["mse"] = float(mean_squared_error(targets_clean, outputs_clean))
                elif metric.lower() == "explained_variance":
                    if np.var(targets_clean) == 0:
                        results["explained_variance"] = 0.0
                    else:
                        results["explained_variance"] = 1.0 - np.var(outputs_clean - targets_clean) / np.var(targets_clean)
                elif metric.lower() == "max_error":
                    results["max_error"] = float(np.max(np.abs(outputs_clean - targets_clean)))
            except Exception as metric_error:
                logger.warning(f"Error calculating metric {metric}: {metric_error}")
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
    return results

def aggregate_metrics(
    metrics_dict: Dict[str, Any], 
    reduce_op: str = 'mean'
) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple GPUs in distributed training.
    
    Args:
        metrics_dict: Dictionary of metrics to aggregate
        reduce_op: Reduction operation ('mean', 'sum', 'max', etc.)
        
    Returns:
        Aggregated metrics dictionary
    """
    if not torch.distributed.is_initialized():
        return metrics_dict
        
    aggregated = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float, torch.Tensor)):
            tensor = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            tensor = tensor.to(torch.cuda.current_device())
            torch.distributed.all_reduce(tensor)
            if reduce_op == 'mean':
                tensor = tensor / torch.distributed.get_world_size()
            aggregated[k] = tensor.item()
        elif isinstance(v, dict):
            # Recursively aggregate nested dictionaries
            aggregated[k] = aggregate_metrics(v, reduce_op)
        else:
            # Non-numeric values are kept as is
            aggregated[k] = v
    return aggregated

class MetricTracker:
    """
    Tracks metrics over time and computes statistics.
    
    Enhanced version of the former MetricAggregator with additional functionality.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize the metric tracker.
        
        Args:
            window_size: Number of recent values to keep for each metric
        """
        self.metrics = {}
        self.window_size = window_size

    def update(self, name: str, value: float) -> Dict[str, float]:
        """
        Add a new value for a metric and return computed statistics.
        
        Args:
            name: Name of the metric
            value: New value to add
            
        Returns:
            Dictionary of computed statistics
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name].pop(0)
        return self.compute_stats(name)

    def compute_stats(self, name: str) -> Dict[str, float]:
        """
        Compute statistics for a specific metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Dictionary with mean, std, min, max and last values
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
            
        values = np.array(self.metrics[name])
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "last": float(values[-1]),
        }
        
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get all tracked metrics.
        
        Returns:
            Dictionary mapping metric names to their value histories
        """
        return self.metrics

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for all tracked metrics.
        
        Returns:
            Dictionary mapping metric names to their statistics
        """
        return {
            name: self.compute_stats(name) 
            for name in self.metrics
        }

    def reset(self, name: Optional[str] = None):
        """
        Reset one or all metrics.
        
        Args:
            name: Name of the metric to reset, or None to reset all
        """
        if name and name in self.metrics:
            self.metrics[name] = []
        elif not name:
            self.metrics = {}