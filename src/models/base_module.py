# src/models/base_module.py
import torch
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class DrugResponseModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: Optional[float] = None,
        transcriptomics_feature_names: Optional[List[str]] = None,
        chemical_feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.criterion = torch.nn.MSELoss()
        
        # Define metrics using MetricCollection
        metrics = torchmetrics.MetricCollection({
            "mse": torchmetrics.MeanSquaredError(),
            "mae": torchmetrics.MeanAbsoluteError(),
            "rmse": torchmetrics.MeanSquaredError(squared=False),
            "r2": torchmetrics.R2Score(),
            "pearson": torchmetrics.PearsonCorrCoef(),
            "spearman": torchmetrics.SpearmanCorrCoef(),
        })
        
        # Create separate instances for each stage - these will be automatically
        # moved to the correct device by PyTorch Lightning
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
        # For storing predictions and targets for visualization
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Explicitly initialize attributes that will be set later
        self.test_predictions = None
        self.test_targets = None
        
        # Store feature names for future use
        self.transcriptomics_feature_names = transcriptomics_feature_names or []
        self.chemical_feature_names = chemical_feature_names or []
        
        self.save_hyperparameters(ignore=["model"])
    
    def forward(self, x):
        # Flexible forward to handle both dict and tensor inputs
        return self.model(x)
    
    def _process_batch(self, batch):
        """Extract inputs and targets from batch based on model type"""
        # Handle different input formats
        if isinstance(batch, dict) and "transcriptomics" in batch:
            # For transcriptomics models
            inputs = batch
            targets = batch["viability"].float().view(-1, 1)
        elif isinstance(batch, dict) and "molecular" in batch:
            # For molecular-only models
            inputs = batch
            targets = batch["viability"].float().view(-1, 1)
        elif isinstance(batch, dict) and "viability" in batch:
            # For any batch with viability
            inputs = batch
            targets = batch["viability"].float().view(-1, 1)
        else:
            # Default case for tuple inputs
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
                targets = targets.float().view(-1, 1)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")
        
        return inputs, targets
    
    def _step(self, batch, batch_type):
        # Process batch to get inputs and targets
        inputs, targets = self._process_batch(batch)
        
        # Forward pass
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Log loss
        self.log(f"{batch_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update metrics based on batch type
        if batch_type == "train":
            metric_outputs = self.train_metrics(outputs, targets)
        elif batch_type == "val":
            metric_outputs = self.val_metrics(outputs, targets)
        else:  # test
            metric_outputs = self.test_metrics(outputs, targets)
        
        # Log all metrics with a single call
        self.log_dict(metric_outputs, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs and targets for later analysis
        return {"loss": loss, "preds": outputs.detach(), "targets": targets.detach()}
    
    def training_step(self, batch, batch_idx):
        output = self._step(batch, "train")
        
        # Apply gradient clipping if specified
        if self.grad_clip is not None:
            self.clip_gradients(
                optimizer=self.optimizers(), 
                gradient_clip_val=self.grad_clip
            )
        
        self.training_step_outputs.append(output)
        return output
    
    def validation_step(self, batch, batch_idx):
        output = self._step(batch, "val")
        self.validation_step_outputs.append(output)
        return output
    
    def test_step(self, batch, batch_idx):
        # Process batch to get inputs and targets
        inputs, targets = self._process_batch(batch)
        
        # Forward pass
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Log loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update metrics
        metric_outputs = self.test_metrics(outputs, targets)
        self.log_dict(metric_outputs, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs and targets for later analysis
        self.test_step_outputs.append({
            "loss": loss,
            "preds": outputs.detach(),
            "targets": targets.detach()
        })
        
        return {
            "loss": loss,
            "preds": outputs.detach(),
            "targets": targets.detach()
        }
    
    def on_train_epoch_end(self):
        # Store predictions for visualization if needed
        if self.training_step_outputs:
            try:
                preds = torch.cat([x["preds"] for x in self.training_step_outputs])
                targets = torch.cat([x["targets"] for x in self.training_step_outputs])
                self.train_predictions = preds.cpu().numpy().flatten()
                self.train_targets = targets.cpu().numpy().flatten()
            except Exception as e:
                logger.warning(f"Error processing training outputs: {e}")
            finally:
                self.training_step_outputs.clear()  # Free memory
    
    def on_validation_epoch_end(self):
        # Store predictions for visualization if needed
        if self.validation_step_outputs:
            try:
                preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
                targets = torch.cat([x["targets"] for x in self.validation_step_outputs])
                self.val_predictions = preds.cpu().numpy().flatten()
                self.val_targets = targets.cpu().numpy().flatten()
            except Exception as e:
                logger.warning(f"Error processing validation outputs: {e}")
            finally:
                self.validation_step_outputs.clear()  # Free memory
    
    def on_test_epoch_end(self):
        # Store predictions and targets for visualization
        if self.test_step_outputs:
            try:
                preds = torch.cat([x["preds"] for x in self.test_step_outputs])
                targets = torch.cat([x["targets"] for x in self.test_step_outputs])
                
                # Store as numpy arrays on the module itself
                self.test_predictions = preds.cpu().numpy().flatten()
                self.test_targets = targets.cpu().numpy().flatten()
                
                logger.info(f"Test results stored: {len(self.test_predictions)} predictions")
            except Exception as e:
                logger.error(f"Error processing test outputs: {e}")
            finally:
                self.test_step_outputs.clear()  # Free memory
    
    def configure_optimizers(self):
        # Check if model has custom parameter groups
        if hasattr(self.model, 'get_param_groups'):
            param_groups = self.model.get_param_groups()
            optimizer = torch.optim.Adam(param_groups, lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # Learning rate scheduler
        scheduler = None
        if hasattr(self, 'lr_scheduler_type') and self.lr_scheduler_type:
            if self.lr_scheduler_type == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=getattr(self, 'lr_scheduler_step_size', 10),
                    gamma=getattr(self, 'lr_scheduler_gamma', 0.1)
                )
            elif self.lr_scheduler_type == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=5
                )
            elif self.lr_scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=getattr(self, 'lr_scheduler_t_max', 10)
                )
        else:
            # Default scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }