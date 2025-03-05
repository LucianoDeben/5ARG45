# utils/visualization.py
import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def plot_boxplot(
    data: Dict[str, List[float]], title: str, ylabel: str, filepath: str
) -> Optional[plt.Figure]:
    """
    Create a boxplot with data points overlaid.
    
    Args:
        data: Dictionary mapping metrics to lists of values
        title: Plot title
        ylabel: Y-axis label
        filepath: Where to save the figure
        
    Returns:
        The matplotlib figure or None if an error occurred
    """
    try:
        # Check if we have valid data to plot
        if not data:
            logger.warning("No data provided for boxplot")
            return None
            
        # Check that all values are non-empty lists
        for metric, values in list(data.items()):
            if not values or not isinstance(values, list):
                logger.warning(f"No values for metric {metric}, removing from plot")
                del data[metric]
                
        if not data:
            logger.warning("No valid metrics with data to plot")
            return None
            
        # Create figure and plot
        fig = plt.figure(figsize=(10, 6))
        metrics = list(data.keys())
        values = [data[m] for m in metrics]
        
        plt.boxplot(values, labels=metrics)
        
        # Overlay actual data points with jitter
        for i, m in enumerate(metrics):
            # Check that we have values to plot
            if not data[m]:
                continue
                
            # Create jittered x-coordinates
            x = np.random.normal(i + 1, 0.04, size=len(data[m]))
            plt.plot(x, data[m], "r.", alpha=0.5)
            
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(filepath, dpi=300)
        logger.info(f"Saved boxplot to {filepath}")
        
        return fig
    except Exception as e:
        logger.error(f"Error creating boxplot: {e}")
        return None
    finally:
        plt.close()


def plot_scatter(
    x: List[float], 
    y: List[float], 
    title: str, 
    xlabel: str, 
    ylabel: str, 
    filepath: str,
    add_regression: bool = True
) -> Optional[plt.Figure]:
    """
    Create a scatter plot with optional regression line.
    
    Args:
        x: X-axis values
        y: Y-axis values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filepath: Where to save the figure
        add_regression: Whether to add a regression line
        
    Returns:
        The matplotlib figure or None if an error occurred
    """
    try:
        # Check if we have valid data to plot
        if not x or not y:
            logger.warning("Empty data provided for scatter plot")
            return None
            
        # Ensure x and y have the same length
        if len(x) != len(y):
            logger.error(f"Dimension mismatch: x={len(x)}, y={len(y)}")
            return None
            
        # Convert to numpy arrays
        x_array = np.array(x)
        y_array = np.array(y)
        
        # Remove any NaN values
        valid_indices = ~np.isnan(x_array) & ~np.isnan(y_array)
        x_clean = x_array[valid_indices]
        y_clean = y_array[valid_indices]
        
        if len(x_clean) < 2:
            logger.warning("Not enough valid data points for scatter plot")
            return None
            
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(x_clean, y_clean, alpha=0.7)
        
        # Add regression line if requested
        if add_regression and len(x_clean) > 1:
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            plt.plot(x_clean, p(x_clean), "r--", alpha=0.8)
            
            # Add R² to the plot
            from scipy.stats import pearsonr
            r, _ = pearsonr(x_clean, y_clean)
            r2 = r**2
            plt.text(
                0.05, 0.95, f"R² = {r2:.4f}", 
                transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top'
            )
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(filepath, dpi=300)
        logger.info(f"Saved scatter plot to {filepath}")
        
        return fig
    except Exception as e:
        logger.error(f"Error creating scatter plot: {e}")
        return None
    finally:
        plt.close()