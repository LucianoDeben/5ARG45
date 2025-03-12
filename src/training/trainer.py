# src/training/multi_run_trainer.py
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import json
from tqdm import tqdm
import logging
import seaborn as sns

# Enable Tensor Core optimization for better performance on A100 GPUs
torch.set_float32_matmul_precision('medium')

from torch.utils.data import DataLoader
from src.data.datasets import DatasetFactory
from src.data.loaders import GCTXDataLoader

logger = logging.getLogger(__name__)

class MultiRunTrainer:
    """
    Trainer class for running multiple training runs with different random seeds.
    Supports either:
    1. Creating datasets from a GCTX loader, or
    2. Using pre-created dataloaders
    """
    
    def __init__(
        self,
        module_class: pl.LightningModule,
        module_kwargs: Dict[str, Any],
        # Dataset creation approach
        gctx_loader: Optional[GCTXDataLoader] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        # Pre-created dataloader approach
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        # Common parameters
        num_runs: int = 5,
        max_epochs: int = 100,
        patience: int = 10,
        output_dir: str = "results",
        experiment_name: str = "drug_response",
        base_seed: int = 42,
        gpu_if_available: bool = True,
        gradient_accumulation_steps: int = 1,
        custom_callbacks: Optional[List[pl.Callback]] = None,
        external_test_loaders: Optional[Dict[str, DataLoader]] = None,
        visualizations_to_generate: Optional[List[str]] = None,
    ):
        """
        Initialize the MultiRunTrainer.
        
        Supports two approaches:
        1. Dataset creation from GCTX (provide gctx_loader, dataset_kwargs, dataloader_kwargs)
        2. Pre-created dataloaders (provide train_dataloader, val_dataloader, test_dataloader)
        
        Args:
            module_class: Lightning module class to instantiate
            module_kwargs: Keyword arguments for model initialization
            
            # Dataset creation approach
            gctx_loader: GCTX data loader
            dataset_kwargs: Arguments for dataset creation
            dataloader_kwargs: Arguments for DataLoader creation
            
            # Pre-created dataloader approach
            train_dataloader: Pre-created training dataloader
            val_dataloader: Pre-created validation dataloader  
            test_dataloader: Pre-created test dataloader
            
            # Common parameters
            num_runs: Number of runs to perform
            max_epochs: Maximum epochs per run
            patience: Patience for early stopping
            output_dir: Base directory for output
            experiment_name: Name of the experiment
            base_seed: Base seed for random number generation
            gpu_if_available: Whether to use GPU if available
            custom_callbacks: Additional Lightning callbacks
            external_test_loaders: Additional named test loaders for evaluation
            visualizations_to_generate: List of visualizations to generate
        """
        self.module_class = module_class
        self.module_kwargs = module_kwargs
        
        # Store parameters for dataset creation approach
        self.gctx_loader = gctx_loader
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs or {
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 4,
        }
        
        # Store pre-created dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Validate that one approach is specified correctly
        if gctx_loader is not None and dataset_kwargs is not None:
            self.use_dataset_creation = True
        elif train_dataloader is not None and val_dataloader is not None:
            self.use_dataset_creation = False
        else:
            raise ValueError("Must provide either (gctx_loader, dataset_kwargs) or (train_dataloader, val_dataloader)")
        
        # Common parameters
        self.num_runs = num_runs
        self.max_epochs = max_epochs
        self.patience = patience
        self.base_seed = base_seed
        self.gpu_if_available = gpu_if_available
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.custom_callbacks = custom_callbacks or []
        self.external_test_loaders = external_test_loaders or {}
        
        # Visualization parameters
        self.visualizations_to_generate = visualizations_to_generate or [
            "predictions", "learning_curves", "residual", "error_distribution", "feature_importance",
            "boxplot", "violinplot", "calibration", "prediction_intervals"
        ]
        
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
        if not self.use_dataset_creation:
            raise ValueError("Dataset creation not enabled; using pre-created dataloaders")
            
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
        if not self.use_dataset_creation:
            raise ValueError("Dataset creation not enabled; using pre-created dataloaders")
            
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
        
        callbacks = [checkpoint_callback, early_stopping] + self.custom_callbacks
        
        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            logger=[tb_logger, csv_logger],
            accelerator="gpu" if torch.cuda.is_available() and self.gpu_if_available else "cpu",
            devices="auto" if torch.cuda.is_available() and self.gpu_if_available else 1,  # Use all available GPUs
            strategy="ddp" if torch.cuda.device_count() > 1 else "auto",  # Add DDP strategy for multi-GPU
            log_every_n_steps=10,
            deterministic=True,
            enable_progress_bar=True,
            accumulate_grad_batches=self.gradient_accumulation_steps,
        )
        
        return trainer, checkpoint_callback
    
    # TODO: Implement hyperparameter optimization methods with Optuna or other libary
    def suggest_hyperparameters(self, trial):
        """
        Suggest hyperparameters for optimization with Optuna.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameter suggestions
        """
        # Example parameters to tune
        suggested_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
        }
        
        return suggested_params
    
    def create_ensemble(self):
        """
        Create an ensemble model from all trained models.
        
        Returns:
            Function that takes input data and returns ensemble prediction
        """
        def ensemble_predict(inputs):
            # Make predictions with each model
            all_preds = []
            for model in self.best_models:
                model.eval()
                with torch.no_grad():
                    preds = model(inputs)
                all_preds.append(preds)
            
            # Stack and average predictions
            stacked_preds = torch.stack(all_preds)
            mean_preds = torch.mean(stacked_preds, dim=0)
            std_preds = torch.std(stacked_preds, dim=0)
            
            return mean_preds, std_preds
        
        return ensemble_predict
    
    def train(self):
        """
        Run the training process for multiple runs.
        Each run uses a different random seed for dataset splitting and initialization.
        """
        logger.info(f"Starting training for {self.num_runs} runs with {self.max_epochs} epochs each")
        
        # Data distribution plot (run once)
        if self.test_dataloader:
            all_targets = []
            for batch in self.test_dataloader:
                _, y = batch["transcriptomics"], batch["viability"]
                all_targets.append(y.cpu().numpy())
            all_targets = np.concatenate(all_targets)
            plt.figure(figsize=(10, 8))
            plt.hist(all_targets, bins=50, alpha=0.7, density=True)
            plt.xlabel("True Cell Viability")
            plt.ylabel("Density")
            plt.title("Test Set Data Distribution")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, "data_distribution.png"))
            plt.close()
        
        # Create dictionary to store aggregated results
        aggregated_results = {
            "runs": [],
            "mean": {},
            "std": {},
            "min": {},
            "max": {},
        }
        
        # Run training for each run
        for run_id in tqdm(range(self.num_runs), desc="Training runs"):
            run_seed = self.base_seed + run_id
            logger.info(f"Starting run {run_id+1}/{self.num_runs} with seed {run_seed}")
            
            # Set seed for reproducibility
            pl.seed_everything(run_seed)
            
            # Get dataloaders for this run
            if self.use_dataset_creation:
                # Create new datasets and dataloaders for each run
                train_ds, val_ds, test_ds = self._create_datasets(run_seed)
                logger.info(f"Created datasets - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
                train_loader, val_loader, test_loader = self._create_dataloaders(train_ds, val_ds, test_ds)
            else:
                # Use pre-created dataloaders
                train_loader = self.train_dataloader
                val_loader = self.val_dataloader
                test_loader = self.test_dataloader
                logger.info("Using pre-created dataloaders")
            
            # Create new model instance
            model = self.module_class(**self.module_kwargs)
            
            # Setup trainer
            trainer, checkpoint_callback = self._setup_trainer(run_id)
            
            # Train model
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            # Load best model
            best_model_path = checkpoint_callback.best_model_path
            if best_model_path and os.path.exists(best_model_path):
                logger.info(f"Loading best model from {best_model_path}")
                best_model = self.module_class.load_from_checkpoint(
                    best_model_path, 
                    model=model.model  # Pass the actual model
                )
                self.best_models.append(best_model)
            else:
                logger.warning("No best model checkpoint found, using current model state")
                self.best_models.append(model)
            
            # Test model on internal test set
            if test_loader:
                logger.info(f"Testing model on internal test set")
                test_results = trainer.test(self.best_models[-1], dataloaders=test_loader)[0]
                self.run_results.append(test_results)
                
                # Store predictions and targets for visualization if they exist
                if hasattr(self.best_models[-1], 'test_predictions') and hasattr(self.best_models[-1], 'test_targets'):
                    self.test_predictions.append(self.best_models[-1].test_predictions)
                    self.test_targets.append(self.best_models[-1].test_targets)
            
            # Test on external test sets if provided
            for name, ext_test_loader in self.external_test_loaders.items():
                logger.info(f"Testing model on external test set: {name}")
                ext_results = trainer.test(self.best_models[-1], dataloaders=ext_test_loader)[0]
                self.external_test_results[name].append(ext_results)
            
            # Generate run-specific visualizations if we have predictions
            if hasattr(self.best_models[-1], 'test_predictions') and hasattr(self.best_models[-1], 'test_targets'):
                self._generate_run_visualizations(run_id)
            
            # Add results to aggregated results
            aggregated_results["runs"].append(test_results if test_loader else {})
        
        # Calculate aggregate statistics
        self._calculate_aggregate_statistics(aggregated_results)
        
        # Generate final visualizations if we have predictions
        if self.test_predictions and self.test_targets:
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
    
# Helper methods for individual visualizations (per-run)
    def _plot_run_predictions(self, run_id: int, y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: str) -> None:
        """Plot predictions vs targets for a specific run."""
        self._plot_predictions(y_true, y_pred, title, output_path)

    def _plot_learning_curves(self, run_id: int, run_dir: str) -> None:
        """Plot learning curves from CSV logs for a specific run."""
        csv_log_path = os.path.join(run_dir, "csv_logs", "version_0", "metrics.csv")
        if os.path.exists(csv_log_path):
            metrics_df = pd.read_csv(csv_log_path)
            plt.figure(figsize=(10, 6))
            if "train_loss" in metrics_df.columns:
                train_loss = metrics_df[metrics_df["train_loss"].notna()][["epoch", "train_loss"]]
                plt.plot(train_loss["epoch"], train_loss["train_loss"], label="Train Loss")
            if "val_loss" in metrics_df.columns:
                val_loss = metrics_df[metrics_df["val_loss"].notna()][["epoch", "val_loss"]]
                plt.plot(val_loss["epoch"], val_loss["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Run {run_id} Learning Curves")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "learning_curves.png"))
            plt.close()

    def _plot_residual(self, run_id: int, run_dir: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot residuals (predicted - true) for a specific run."""
        residuals = y_pred - y_true
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("True Values")
        plt.ylabel("Residuals (Predicted - True)")
        plt.title(f"Run {run_id} Residual Plot")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "residual_plot.png"))
        plt.close()

    def _plot_error_distribution(self, run_id: int, run_dir: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot error distribution histogram for a specific run."""
        errors = y_pred - y_true
        plt.figure(figsize=(10, 8))
        plt.hist(errors, bins=50, alpha=0.7, density=True)
        plt.xlabel("Error (Predicted - True)")
        plt.ylabel("Density")
        plt.title(f"Run {run_id} Error Distribution")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "error_distribution.png"))
        plt.close()

    def _plot_feature_importance(self, run_id: int, run_dir: str) -> None:
        """Plot feature importance for a specific run."""
        if hasattr(self.best_models[-1], 'compute_feature_importance'):
            importance_dict = self.best_models[-1].compute_feature_importance(dataloader=self.test_dataloader, subset_size=100)
            if importance_dict:
                plt.figure(figsize=(12, 6))
                for modality, importances in importance_dict.items():
                    if importances:
                        top_n = 20
                        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n])
                        plt.bar(importances.keys(), importances.values(), alpha=0.7, label=modality.capitalize())
                plt.xlabel("Feature")
                plt.ylabel("Normalized Importance")
                plt.title(f"Run {run_id} Feature Importance (Top 20)")
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "feature_importance.png"))
                plt.close()

    # Helper methods for individual visualizations (aggregated)
    def _plot_boxplot(self) -> None:
        """Plot boxplot of key metrics across runs."""
        metrics_to_plot = ["test_rmse", "test_pearson", "test_r2", "test_mae"]
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

    def _plot_violinplot(self) -> None:
        """Plot violin plot of key metrics across runs."""
        metrics_to_plot = ["test_rmse", "test_pearson", "test_r2", "test_mae"]
        metric_data = []
        metric_names = []
        for metric in metrics_to_plot:
            values = [result[metric] for result in self.run_results if metric in result]
            if values:
                metric_data.extend(values)
                metric_names.extend([metric.replace('test_', '')] * len(values))
        if metric_data:
            plt.figure(figsize=(12, 6))
            sns.violinplot(x=metric_names, y=metric_data, palette="muted")
            plt.title("Distribution of Test Metrics Across Runs (Violin Plot)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, "test_metrics_violinplot.png"))
            plt.close()

    def _plot_aggregated_predictions(self) -> None:
        """Plot aggregated predictions as a 2D histogram across all runs."""
        all_targets = np.concatenate(self.test_targets)
        all_preds = np.concatenate(self.test_predictions)
        plt.figure(figsize=(10, 8))
        plt.hist2d(all_targets, all_preds, bins=50, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Count')
        min_val = min(np.min(all_targets), np.min(all_preds))
        max_val = max(np.max(all_targets), np.max(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title('Aggregated Test Predictions Across All Runs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "aggregated_predictions.png"))
        plt.close()

    def _plot_calibration(self) -> None:
        """Plot calibration curve across all runs."""
        all_targets = np.concatenate(self.test_targets)
        all_preds = np.concatenate(self.test_predictions)
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(all_preds, bins) - 1
        bin_means_pred = []
        bin_means_true = []
        for i in range(len(bins) - 1):
            in_bin = (bin_indices == i)
            if in_bin.sum() > 0:
                bin_means_pred.append(all_preds[in_bin].mean())
                bin_means_true.append(all_targets[in_bin].mean())
        plt.figure(figsize=(10, 8))
        plt.plot(bin_means_pred, bin_means_true, 'o-', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel("Mean Predicted Value")
        plt.ylabel("Mean True Value")
        plt.title("Calibration Plot Across All Runs")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "calibration_plot.png"))
        plt.close()

    def _plot_prediction_intervals(self) -> None:
        """Plot prediction intervals across all runs if uncertainty is available."""
        if hasattr(self.best_models[0], 'test_uncertainties'):
            all_targets = np.concatenate(self.test_targets)
            all_preds = np.concatenate(self.test_predictions)
            all_uncertainties = np.concatenate([model.test_uncertainties for model in self.best_models])
            plt.figure(figsize=(10, 8))
            idx = np.argsort(all_targets)
            plt.plot(all_targets[idx], label="True Values")
            plt.plot(all_preds[idx], label="Predictions")
            plt.fill_between(
                range(len(idx)),
                all_preds[idx] - 1.96 * all_uncertainties[idx],
                all_preds[idx] + 1.96 * all_uncertainties[idx],
                alpha=0.3,
                label="95% Prediction Interval"
            )
            plt.xlabel("Sample (Sorted by True Value)")
            plt.ylabel("Value")
            plt.title("Prediction Intervals Across All Runs")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, "prediction_intervals.png"))
            plt.close()

    # Refactored main visualization methods
    def _generate_run_visualizations(self, run_id: int) -> None:
        """Generate visualizations for a specific run based on user selection."""
        run_dir = os.path.join(self.experiment_dir, f"run_{run_id}")
        has_preds = hasattr(self.best_models[-1], 'test_predictions') and hasattr(self.best_models[-1], 'test_targets')

        if "predictions" in self.visualizations_to_generate and has_preds:
            self._plot_run_predictions(
                run_id, 
                self.test_targets[-1], 
                self.test_predictions[-1], 
                f"Run {run_id} Test Predictions", 
                os.path.join(run_dir, "test_predictions.png")
            )
            # External test sets
            for name, results in self.external_test_results.items():
                if results and len(results) > run_id:
                    if hasattr(self.best_models[-1], f"{name}_predictions") and hasattr(self.best_models[-1], f"{name}_targets"):
                        self._plot_run_predictions(
                            run_id,
                            getattr(self.best_models[-1], f"{name}_targets"),
                            getattr(self.best_models[-1], f"{name}_predictions"),
                            f"Run {run_id} {name} Predictions",
                            os.path.join(run_dir, f"{name}_predictions.png")
                        )

        if "learning_curves" in self.visualizations_to_generate:
            self._plot_learning_curves(run_id, run_dir)

        if "residual" in self.visualizations_to_generate and has_preds:
            self._plot_residual(run_id, run_dir, self.test_targets[-1], self.test_predictions[-1])

        if "error_distribution" in self.visualizations_to_generate and has_preds:
            self._plot_error_distribution(run_id, run_dir, self.test_targets[-1], self.test_predictions[-1])

        if "feature_importance" in self.visualizations_to_generate:
            self._plot_feature_importance(run_id, run_dir)

    def _generate_final_visualizations(self) -> None:
        """Generate final visualizations aggregating all runs based on user selection."""
        has_preds = self.test_predictions and self.test_targets

        if "boxplot" in self.visualizations_to_generate:
            self._plot_boxplot()

        if "violinplot" in self.visualizations_to_generate:
            self._plot_violinplot()

        if "predictions" in self.visualizations_to_generate and has_preds:
            self._plot_aggregated_predictions()

        if "calibration" in self.visualizations_to_generate and has_preds:
            self._plot_calibration()

        if "prediction_intervals" in self.visualizations_to_generate and has_preds:
            self._plot_prediction_intervals()
            
    
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
        plt.xlim([0, 1])
        plt.ylim([0, 1])
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