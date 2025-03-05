# utils/visualization.py
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_boxplot(
    data: Dict[str, List[float]], title: str, ylabel: str, filepath: str
) -> None:
    plt.figure(figsize=(10, 6))
    metrics = list(data.keys())
    values = [data[m] for m in metrics]
    plt.boxplot(values, labels=metrics)
    for i, m in enumerate(metrics):
        x = np.random.normal(i + 1, 0.04, size=len(data[m]))
        plt.plot(x, data[m], "r.", alpha=0.5)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
