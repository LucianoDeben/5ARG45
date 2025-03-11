# src/training/trainer.py
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import torch
from torch.utils.data import DataLoader

class MultiRunTrainer:
    def __init__(
        self,
        module_class: pl.LightningModule,
        module_kwargs: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        num_runs: int = 5,
        max_epochs: int = 100,
        patience: int = 10,
        log_dir: str = "logs",
        experiment_name: str = "drug_response",
        seed: int = 42,
        custom_callbacks: Optional[List[pl.Callback]] = None,
    ):
        self.module_class = module_class
        self.module_kwargs = module_kwargs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_runs = num_runs
        self.max_epochs = max_epochs
        self.patience = patience
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.seed = seed
        self.custom_callbacks = custom_callbacks or []
        
        # Results storage
        self.run_results = []
        self.best_models = []
        
    def _setup_trainer(self, run_id: int) -> pl.Trainer:
        # Set seed for reproducibility
        pl.seed_everything(self.seed + run_id)
        
        # Setup logger
        logger = TensorBoardLogger(
            save_dir=self.log_dir,
            name=self.experiment_name,
            version=f"run_{run_id}"
        )
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:02d}-{val_loss:.4f}"
        )
        
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            mode="min"
        )
        
        callbacks = [checkpoint_callback, early_stopping] + self.custom_callbacks
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            logger=logger,
            deterministic=True,
            log_every_n_steps=10,
        )
        
        return trainer, checkpoint_callback
        
    def train(self) -> Dict[str, Any]:
        """Perform multiple training runs and collect results."""
        for run_id in range(self.num_runs):
            print(f"Starting run {run_id+1}/{self.num_runs}")
            
            # Create new model instance
            model = self.module_class(**self.module_kwargs)
            
            # Setup trainer
            trainer, checkpoint_callback = self._setup_trainer(run_id)
            
            # Train model
            trainer.fit(
                model,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.val_dataloader
            )
            
            # Load best model
            best_model_path = checkpoint_callback.best_model_path
            best_model = self.module_class.load_from_checkpoint(best_model_path)
            self.best_models.append(best_model)
            
            # Test model if test data provided
            if self.test_dataloader:
                test_results = trainer.test(best_model, dataloaders=self.test_dataloader)[0]
                self.run_results.append(test_results)
                
        # Aggregate results
        results = self._aggregate_results()
        return results
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from multiple runs."""
        if not self.run_results:
            return {}
            
        # Convert list of dicts to dict of lists
        metrics_dict = {}
        for run_result in self.run_results:
            for metric_name, value in run_result.items():
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(value)
        
        # Calculate statistics
        aggregated_results = {}
        for metric_name, values in metrics_dict.items():
            values_array = np.array(values)
            aggregated_results[f"{metric_name}_mean"] = values_array.mean()
            aggregated_results[f"{metric_name}_std"] = values_array.std()
            aggregated_results[f"{metric_name}_min"] = values_array.min()
            aggregated_results[f"{metric_name}_max"] = values_array.max()
        
        return aggregated_results