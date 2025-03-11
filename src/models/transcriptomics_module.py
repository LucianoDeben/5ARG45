# src/models/transcriptomics_module.py
import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

class TranscriptomicsModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.criterion = torch.nn.MSELoss()
        
        # Save hyperparameters (except model which can't be pickled easily)
        self.save_hyperparameters(ignore=["model"])
        
        # For storing predictions and targets for calculation at end of epoch
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch, batch_type):
        # Extract inputs and targets
        inputs = batch["transcriptomics"]
        targets = batch["viability"].float().view(-1, 1)
        
        # Forward pass
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Log step loss
        self.log(f"{batch_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store outputs and targets for epoch end metric calculation
        return {"loss": loss, "preds": outputs, "targets": targets}
    
    def training_step(self, batch, batch_idx):
        output = self._step(batch, "train")
        
        # Gradient clipping if enabled
        if self.grad_clip is not None:
            self.clip_gradients(optimizer=self.optimizers(), 
                            gradient_clip_val=self.grad_clip)
            
        self.training_step_outputs.append(output)
        return output
    
    def validation_step(self, batch, batch_idx):
        output = self._step(batch, "val")
        self.validation_step_outputs.append(output)
        return output
    
    def test_step(self, batch, batch_idx):
        output = self._step(batch, "test")
        self.test_step_outputs.append(output)
        return output
    
    def _calculate_metrics(self, outputs, prefix):
        # Add .detach() before .cpu().numpy()
        all_preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy().flatten()
        all_targets = torch.cat([x["targets"] for x in outputs]).detach().cpu().numpy().flatten()
        
        # Calculate regression metrics
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        pearson = pearsonr(all_targets, all_preds)[0]
        
        # Log metrics
        self.log(f"{prefix}_mse", mse, prog_bar=True)
        self.log(f"{prefix}_rmse", rmse, prog_bar=True)
        self.log(f"{prefix}_mae", mae, prog_bar=True)
        self.log(f"{prefix}_r2", r2, prog_bar=True)
        self.log(f"{prefix}_pearson", pearson, prog_bar=True)
        
        # Return metrics dict and arrays for potential visualization
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson': pearson
        }, all_targets, all_preds
    
    def on_train_epoch_end(self):
        metrics, _, _ = self._calculate_metrics(self.training_step_outputs, "train")
        self.training_step_outputs.clear()  # Clear to free memory
        return metrics
    
    def on_validation_epoch_end(self):
        metrics, _, _ = self._calculate_metrics(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()  # Clear to free memory
        return metrics
    
    def on_test_epoch_end(self):
        metrics, targets, preds = self._calculate_metrics(self.test_step_outputs, "test")
        self.test_step_outputs.clear()  # Clear to free memory
        
        # Store predictions and targets as attributes for later use
        self.test_predictions = preds
        self.test_targets = targets
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Optional learning rate scheduler
        scheduler = None
        if hasattr(self, 'lr_scheduler_type') and self.lr_scheduler_type:
            if self.lr_scheduler_type == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=self.lr_scheduler_step_size, 
                    gamma=self.lr_scheduler_gamma
                )
            elif self.lr_scheduler_type == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=5
                )
        
        # Return the optimizer with or without scheduler
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }
        return optimizer