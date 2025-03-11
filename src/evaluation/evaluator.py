# src/evaluation/evaluator.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from scipy.stats import pearsonr, spearmanr

class DrugResponseEvaluator:
    def __init__(
        self,
        model: pl.LightningModule,
        dataloader: DataLoader,
        output_dir: str = "evaluation_results",
    ):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize containers for predictions and targets
        self.predictions = []
        self.targets = []
        self.metadata = []
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return metrics."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Collect predictions and targets
        with torch.no_grad():
            for batch in self.dataloader:
                # Get predictions
                outputs = self.model(batch)
                
                # Store predictions and targets
                self.predictions.append(outputs.cpu().numpy())
                self.targets.append(batch["viability"].cpu().numpy())
                
                # Store metadata if available
                if hasattr(batch, "metadata"):
                    self.metadata.append(batch.metadata)
        
        # Concatenate results
        self.predictions = np.concatenate(self.predictions)
        self.targets = np.concatenate(self.targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return metrics
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Calculate regression metrics
        mae = np.mean(np.abs(self.predictions - self.targets))
        mse = np.mean(np.square(self.predictions - self.targets))
        rmse = np.sqrt(mse)
        
        # Calculate correlation metrics
        pearson_corr, pearson_p = pearsonr(self.predictions.flatten(), self.targets.flatten())
        spearman_corr, spearman_p = spearmanr(self.predictions.flatten(), self.targets.flatten())
        
        # Calculate R2 score
        y_mean = np.mean(self.targets)
        ss_tot = np.sum(np.square(self.targets - y_mean))
        ss_res = np.sum(np.square(self.targets - self.predictions))
        r2 = 1 - (ss_res / ss_tot)
        
        # Return all metrics
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
            "r2_score": r2,
        }
    
    def _generate_visualizations(self) -> None:
        """Generate evaluation visualizations."""
        # 1. Scatter plot of predicted vs. actual values
        plt.figure(figsize=(10, 8))
        plt.scatter(self.targets, self.predictions, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(self.targets), np.min(self.predictions))
        max_val = max(np.max(self.targets), np.max(self.predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel("Actual Viability")
        plt.ylabel("Predicted Viability")
        plt.title("Predicted vs. Actual Viability")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "predicted_vs_actual.png"), dpi=300)
        plt.close()
        
        # 2. Histogram of errors
        errors = self.predictions - self.targets
        plt.figure(figsize=(10, 8))
        plt.hist(errors, bins=50, alpha=0.75)
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.title("Distribution of Prediction Errors")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "error_distribution.png"), dpi=300)
        plt.close()
        
        # 3. Error bar visualization (optional, for grouped data)
        if len(self.metadata) > 0:
            # Implement custom visualization based on metadata
            pass
        
        # Save predictions and targets for further analysis
        results_df = pd.DataFrame({
            "actual": self.targets.flatten(),
            "predicted": self.predictions.flatten(),
            "error": errors.flatten()
        })
        results_df.to_csv(os.path.join(self.output_dir, "prediction_results.csv"), index=False)