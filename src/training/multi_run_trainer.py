# src/training/multi_run_trainer.py
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import json
from tqdm import tqdm
import logging

# Enable Tensor Core optimization for better performance on A100 GPUs
torch.set_float32_matmul_precision('medium')

from torch.utils.data import DataLoader
from src.data.datasets import DatasetFactory
from src.data.loaders import GCTXDataLoader

logger = logging.getLogger(__name__)

class MultiRunTrainer:
    """
    Trainer class for running multiple training runs with different random seeds.
    Supports different dataset splits for each run and aggregates results.
    """
    
    def __init__(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        gctx_loader: GCTXDataLoader,
        dataset_kwargs: Dict[str, Any],
        dataloader_kwargs: Dict[str, Any] = None,
        num_runs: int = 20,
        max_epochs: int = 50,
        patience: int = 10,
        output_dir: str = "results",
        experiment_name: str = "experiment",
        base_seed: int = 42,
        gpu_if_available: bool = True,
        external_test_loaders: Dict[str, DataLoader] = None,
    ):
        """
        Initialize the MultiRunTrainer.
        
        Args:
            model_class: Lightning module class to instantiate
            model_kwargs: Keyword arguments for model initialization
            gctx_loader: GCTX data loader
            dataset_kwargs: Arguments for dataset creation
            dataloader_kwargs: Arguments for DataLoader creation
            num_runs: Number of runs to perform
            max_epochs: Maximum epochs per run
            patience: Patience for early stopping
            output_dir: Base directory for output
            experiment_name: Name of the experiment
            base_seed: Base seed for random number generation
            gpu_if_available: Whether to use GPU if available
            external_test_loaders: Additional test loaders for evaluation
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.gctx_loader = gctx_loader
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs or {
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 4,
        }
        self.num_runs = num_runs
        self.max_epochs = max_epochs
        self.patience = patience
        self.base_seed = base_seed
        self.gpu_if_available = gpu_if_available
        self.external_test_loaders = external_test_loaders or {}
        
        # Set experiment output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Results storage
        self.run_results = []
        self.best_models = []
        self.test_predictions = []
        self.test_targets = []
        self.external_test_results = {name: [] for name in self.external_test_loaders}
        
    def _create_datasets(self, run_seed: int):
        """Create datasets with random split based on run seed."""
        # Use run-specific seed for dataset creation to ensure different splits
        dataset_kwargs = self.dataset_kwargs.copy()
        dataset_kwargs["random_state"] = run_seed
        
        # Create and split datasets
        train_ds, val_ds, test_ds = DatasetFactory.create_and_split_transcriptomics(
            gctx_loader=self.gctx_loader,
            **dataset_kwargs
        )
        
        return train_ds, val_ds, test_ds
    
    def _create_dataloaders(self, train_ds, val_ds, test_ds):
        """Create dataloaders from datasets."""
        # Training dataloader (with shuffle)
        train_kwargs = self.dataloader_kwargs.copy()
        if "shuffle" in train_kwargs:
            train_kwargs["shuffle"] = True  # Ensure training data is shuffled
        
        train_loader = DataLoader(train_ds, **train_kwargs)
        
        # Val/Test dataloaders (no shuffle)
        eval_kwargs = self.dataloader_kwargs.copy()
        if "shuffle" in eval_kwargs:
            eval_kwargs["shuffle"] = False  # Don't shuffle eval data
        
        val_loader = DataLoader(val_ds, **eval_kwargs)
        test_loader = DataLoader(test_ds, **eval_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def _setup_trainer(self, run_id: int):
        """Set up Lightning Trainer for a specific run."""
        # Set up directories for this run
        run_dir = os.path.join(self.experiment_dir, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Set up loggers
        tb_logger = TensorBoardLogger(
            save_dir=run_dir,
            name="tensorboard_logs",
            default_hp_metric=False
        )
        
        csv_logger = CSVLogger(
            save_dir=run_dir,
            name="csv_logs"
        )
        
        # Set up callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(run_dir, "checkpoints"),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=self.patience,
            verbose=True
        )
        
        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stopping],
            logger=[tb_logger, csv_logger],
            accelerator="gpu" if torch.cuda.is_available() and self.gpu_if_available else "cpu",
            devices=1,
            log_every_n_steps=10,
            deterministic=True,
            enable_progress_bar=True,
        )
        
        return trainer, checkpoint_callback
    
    def train(self):
        """
        Run the training process for multiple runs.
        Each run uses a different random seed for dataset splitting and initialization.
        """
        logger.info(f"Starting training for {self.num_runs} runs with {self.max_epochs} epochs each")
        
        # Create dictionary to store aggregated results
        aggregated_results = {
            "runs": [],
            "mean": {},
            "std": {},
            "min": {},
            "max": {},
        }
        
        # Run training for each run
        for run_id in range(self.num_runs):
            run_seed = self.base_seed + run_id
            logger.info(f"Starting run {run_id+1}/{self.num_runs} with seed {run_seed}")
            
            # Set seed for reproducibility
            pl.seed_everything(run_seed)
            
            # Create datasets with run-specific seed
            train_ds, val_ds, test_ds = self._create_datasets(run_seed)
            logger.info(f"Created datasets - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
            
            # Create dataloaders
            train_loader, val_loader, test_loader = self._create_dataloaders(train_ds, val_ds, test_ds)
            
            # Create new model instance
            model = self.model_class(**self.model_kwargs)
            
            # Setup trainer
            trainer, checkpoint_callback = self._setup_trainer(run_id)
            
            # Train model
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            # Load best model
            best_model_path = checkpoint_callback.best_model_path
            if best_model_path and os.path.exists(best_model_path):
                logger.info(f"Loading best model from {best_model_path}")
                best_model = self.model_class.load_from_checkpoint(
                    best_model_path, 
                    model=model.model  # Pass the actual model
                )
                self.best_models.append(best_model)
            else:
                logger.warning("No best model checkpoint found, using current model state")
                self.best_models.append(model)
            
            # Test model on internal test set
            logger.info(f"Testing model on internal test set")
            test_results = trainer.test(self.best_models[-1], dataloaders=test_loader)[0]
            self.run_results.append(test_results)
            
            # Store predictions and targets for visualization
            self.test_predictions.append(self.best_models[-1].test_predictions)
            self.test_targets.append(self.best_models[-1].test_targets)
            
            # Test on external test sets if provided
            for name, ext_test_loader in self.external_test_loaders.items():
                logger.info(f"Testing model on external test set: {name}")
                ext_results = trainer.test(self.best_models[-1], dataloaders=ext_test_loader)[0]
                self.external_test_results[name].append(ext_results)
            
            # Generate run-specific visualizations
            self._generate_run_visualizations(run_id)
            
            # Add results to aggregated results
            aggregated_results["runs"].append(test_results)
        
        # Calculate aggregate statistics
        self._calculate_aggregate_statistics(aggregated_results)
        
        # Generate final visualizations
        self._generate_final_visualizations()
        
        # Save aggregated results
        self._save_aggregated_results(aggregated_results)
        
        logger.info(f"Training completed. Results saved to {self.experiment_dir}")
        
        return aggregated_results
    
    def _calculate_aggregate_statistics(self, aggregated_results):
        """Calculate aggregate statistics from multiple runs."""
        # Extract metrics from all runs
        metrics_dict = {}
        for run_result in self.run_results:
            for metric_name, value in run_result.items():
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(value)
        
        # Calculate statistics
        for metric_name, values in metrics_dict.items():
            values_array = np.array(values)
            aggregated_results["mean"][metric_name] = float(values_array.mean())
            aggregated_results["std"][metric_name] = float(values_array.std())
            aggregated_results["min"][metric_name] = float(values_array.min())
            aggregated_results["max"][metric_name] = float(values_array.max())
        
        # Also do this for external test sets
        for ext_name, ext_results in self.external_test_results.items():
            if not ext_results:
                continue
                
            ext_metrics_dict = {}
            for run_result in ext_results:
                for metric_name, value in run_result.items():
                    key = f"{ext_name}_{metric_name}"
                    if key not in ext_metrics_dict:
                        ext_metrics_dict[key] = []
                    ext_metrics_dict[key].append(value)
            
            # Calculate statistics
            for metric_name, values in ext_metrics_dict.items():
                values_array = np.array(values)
                aggregated_results["mean"][metric_name] = float(values_array.mean())
                aggregated_results["std"][metric_name] = float(values_array.std())
                aggregated_results["min"][metric_name] = float(values_array.min())
                aggregated_results["max"][metric_name] = float(values_array.max())
    
    def _generate_run_visualizations(self, run_id):
        """Generate visualizations for a specific run."""
        run_dir = os.path.join(self.experiment_dir, f"run_{run_id}")
        
        # Plot predictions vs targets
        self._plot_predictions(
            self.test_targets[-1], 
            self.test_predictions[-1], 
            f"Run {run_id} Test Predictions", 
            os.path.join(run_dir, "test_predictions.png")
        )
        
        # If we have external test sets, also plot those
        for name, results in self.external_test_results.items():
            if results and len(results) > run_id:
                if hasattr(self.best_models[-1], f"{name}_predictions") and hasattr(self.best_models[-1], f"{name}_targets"):
                    self._plot_predictions(
                        getattr(self.best_models[-1], f"{name}_targets"),
                        getattr(self.best_models[-1], f"{name}_predictions"),
                        f"Run {run_id} {name} Predictions",
                        os.path.join(run_dir, f"{name}_predictions.png")
                    )
    
    def _generate_final_visualizations(self):
        """Generate final visualizations aggregating all runs."""
        # Boxplot of key metrics across runs
        metrics_to_plot = ["test_rmse", "test_pearson", "test_r2", "test_mae"]
        
        # Prepare data for boxplot
        boxplot_data = []
        labels = []
        
        for metric in metrics_to_plot:
            values = [result[metric] for result in self.run_results if metric in result]
            if values:
                boxplot_data.append(values)
                labels.append(metric.replace('test_', ''))
        
        if boxplot_data:
            plt.figure(figsize=(12, 6))
            plt.boxplot(boxplot_data, labels=labels)
            plt.title("Distribution of Test Metrics Across Runs")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, "test_metrics_boxplot.png"))
            plt.close()
        
        # Create a summary plot of all runs' predictions
        # This will use alpha blending to show the density of predictions
        plt.figure(figsize=(10, 8))
        
        # Create unified arrays of all predictions and targets
        all_targets = np.concatenate(self.test_targets)
        all_preds = np.concatenate(self.test_predictions)
        
        # Plot a 2D histogram (heatmap)
        plt.hist2d(all_targets, all_preds, bins=50, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Count')
        
        # Add perfect prediction line
        min_val = min(np.min(all_targets), np.min(all_preds))
        max_val = max(np.max(all_targets), np.max(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Aggregated Test Predictions Across All Runs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "aggregated_predictions.png"))
        plt.close()
    
    def _plot_predictions(self, y_true, y_pred, title, output_path):
        """Plot true vs predicted values and save the figure."""
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        
        # Add metrics to plot
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
        
        metrics_text = (
            f"MSE: {mse:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"MAE: {mae:.4f}\n"
            f"R²: {r2:.4f}\n"
            f"Pearson: {pearson:.4f}\n"
            f"Spearman: {spearman:.4f}"
        )
        
        plt.annotate(
            metrics_text, 
            xy=(0.05, 0.95), 
            xycoords='axes fraction', 
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        )
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _save_aggregated_results(self, aggregated_results):
        """Save aggregated results to JSON file."""
        # Convert numpy values to Python native types for JSON serialization
        results_for_json = {
            "runs": [],
            "mean": {k: float(v) for k, v in aggregated_results["mean"].items()},
            "std": {k: float(v) for k, v in aggregated_results["std"].items()},
            "min": {k: float(v) for k, v in aggregated_results["min"].items()},
            "max": {k: float(v) for k, v in aggregated_results["max"].items()},
        }
        
        # Process run results
        for run in aggregated_results["runs"]:
            processed_run = {k: float(v) for k, v in run.items()}
            results_for_json["runs"].append(processed_run)
        
        # Save to JSON file
        with open(os.path.join(self.experiment_dir, "aggregated_results.json"), 'w') as f:
            json.dump(results_for_json, f, indent=2)