import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

class CVResults:
    def __init__(self, results_dict: dict):
        """
        Initializes the CVResults object.

        Args:
            results_dict (dict): A dictionary where keys are model names and values are dictionaries 
                with keys "mean" and "std" (each mapping metric names to values). For example:
                {
                  "LinearRegression": {
                      "mean": {"MSE": 0.0444, "MAE": 0.1406, "R²": 0.3299, "Pearson": 0.6133},
                      "std":  {"MSE": 0.0119, "MAE": 0.0180, "R²": 0.0604, "Pearson": 0.0138}
                  },
                  ...
                }
        """
        self.results_dict = results_dict
        self.results_df = self._create_dataframe(results_dict)
        logging.info("CVResults initialized and DataFrame created.")

    def _create_dataframe(self, results_dict: dict) -> pd.DataFrame:
        """
        Converts the results dictionary to a DataFrame with a MultiIndex for columns.
        """
        data = {}
        for model_name, metrics in results_dict.items():
            row = {}
            for metric, value in metrics["mean"].items():
                row[(metric, "mean")] = value
            for metric, value in metrics["std"].items():
                row[(metric, "std")] = value
            data[model_name] = row
        df = pd.DataFrame.from_dict(data, orient="index")
        # Sort columns by metric name, then statistic
        df = df.reindex(sorted(df.columns, key=lambda x: (x[0], x[1])), axis=1)
        return df

    def get_results_df(self) -> pd.DataFrame:
        """
        Returns the results DataFrame.
        """
        return self.results_df

    def filter_by_metric(self, metric: str, min_mean: float = None, max_mean: float = None) -> pd.DataFrame:
        """
        Filters the results DataFrame by a specified metric's mean value.

        Args:
            metric (str): The metric to filter by (e.g., "MSE").
            min_mean (float, optional): Minimum mean value.
            max_mean (float, optional): Maximum mean value.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        df = self.results_df
        mask = np.ones(len(df), dtype=bool)
        if min_mean is not None:
            mask &= (df[(metric, "mean")] >= min_mean)
        if max_mean is not None:
            mask &= (df[(metric, "mean")] <= max_mean)
        return df[mask]

    def plot_metric(self, metric: str):
        """
        Plots the mean and standard deviation for a specified metric across models.

        Args:
            metric (str): The metric to plot (e.g., "MSE").
        """
        df = self.results_df
        plt.figure(figsize=(8, 6))
        plt.bar(df.index, df[(metric, "mean")], yerr=df[(metric, "std")], capsize=5)
        plt.ylabel(metric)
        plt.title(f"{metric} by Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

