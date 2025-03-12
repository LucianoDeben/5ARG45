# src/models/base_module.py
import torch
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from captum.attr import IntegratedGradients

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
        
        # Store feature names for interpretability
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
    
    def predict_with_uncertainty(self, inputs, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Monte Carlo Dropout to estimate prediction uncertainty.
        
        Args:
            inputs: Input data (dict or tensor).
            n_samples: Number of Monte Carlo samples.
        
        Returns:
            Tuple of (mean predictions, standard deviation of predictions).
        """
        self.train()  # Enable dropout during inference
        preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self(inputs)
            preds.append(pred)
        preds = torch.stack(preds)
        mean_preds = preds.mean(dim=0)
        std_preds = preds.std(dim=0)
        self.eval()  # Return to evaluation mode
        return mean_preds, std_preds
    
    def test_step(self, batch, batch_idx):
            # Process batch to get inputs and targets
            inputs, targets = self._process_batch(batch)
            
            # Perform prediction with uncertainty
            mean_preds, std_preds = self.predict_with_uncertainty(inputs)
            loss = self.criterion(mean_preds, targets)
            
            # Log loss
            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            
            # Update metrics
            metric_outputs = self.test_metrics(mean_preds, targets)
            self.log_dict(metric_outputs, on_step=False, on_epoch=True, prog_bar=True)
            
            # Store outputs, targets, and uncertainties for later analysis
            return {
                "loss": loss,
                "preds": mean_preds.detach(),
                "targets": targets.detach(),
                "uncertainties": std_preds.detach()
            }
    
    def on_train_epoch_end(self):
        # Store predictions for visualization if needed
        if self.training_step_outputs:
            preds = torch.cat([x["preds"] for x in self.training_step_outputs])
            targets = torch.cat([x["targets"] for x in self.training_step_outputs])
            self.train_predictions = preds.cpu().numpy().flatten()
            self.train_targets = targets.cpu().numpy().flatten()
            self.training_step_outputs.clear()  # Free memory
    
    def on_validation_epoch_end(self):
        # Store predictions for visualization if needed
        if self.validation_step_outputs:
            preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
            targets = torch.cat([x["targets"] for x in self.validation_step_outputs])
            self.val_predictions = preds.cpu().numpy().flatten()
            self.val_targets = targets.cpu().numpy().flatten()
            self.validation_step_outputs.clear()  # Free memory
    
    def on_test_epoch_end(self):
            # Store predictions, targets, and uncertainties for visualization
            if self.test_step_outputs:
                preds = torch.cat([x["preds"] for x in self.test_step_outputs])
                targets = torch.cat([x["targets"] for x in self.test_step_outputs])
                uncertainties = torch.cat([x["uncertainties"] for x in self.test_step_outputs])
                self.test_predictions = preds.cpu().numpy().flatten()
                self.test_targets = targets.cpu().numpy().flatten()
                self.test_uncertainties = uncertainties.cpu().numpy().flatten()
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
    
    def compute_feature_importance(self, dataloader: torch.utils.data.DataLoader, n_steps: int = 50) -> Dict[str, Dict[str, float]]:
        """
        Compute feature importance using Captum's Integrated Gradients.
        
        Args:
            dataloader: DataLoader containing test data.
            n_steps: Number of steps for integrated gradients approximation.
        
        Returns:
            Dict mapping modality ('transcriptomics', 'chemical') to feature importance dicts.
        """
        self.eval()  # Set model to evaluation mode
        ig = IntegratedGradients(self)

        feature_importance = {
            "transcriptomics": {},
            "chemical": {}
        }

        # Process one batch for simplicity (can extend to multiple batches)
        batch = next(iter(dataloader))
        inputs, targets = self._process_batch(batch)

        # Ensure inputs are on the correct device and require gradients
        if isinstance(inputs, dict):
            # Handle multimodal inputs
            baselines = {}
            for modality in inputs:
                if modality in ["transcriptomics", "chemical"]:
                    inputs[modality] = inputs[modality].to(self.device).requires_grad_(True)
                    # Use zero baseline (common for biological and chemical data)
                    baselines[modality] = torch.zeros_like(inputs[modality]).to(self.device)
        else:
            raise ValueError("Feature importance computation requires dict inputs with 'transcriptomics' and/or 'chemical' keys")

        # Compute attributions using Integrated Gradients
        if "transcriptomics" in inputs:
            attributions = ig.attribute(
                inputs["transcriptomics"],
                baselines=baselines["transcriptomics"],
                target=0,  # For regression, attribute to the output scalar
                n_steps=n_steps,
                additional_forward_args=(inputs["chemical"] if "chemical" in inputs else None)
            )
            # Average attributions across samples and normalize
            attr_mean = attributions.abs().mean(dim=0).cpu().numpy()
            attr_sum = attr_mean.sum()
            if attr_sum > 0:
                attr_mean /= attr_sum  # Normalize to sum to 1
            # Map to feature names if provided, otherwise use indices
            feature_names = self.transcriptomics_feature_names if self.transcriptomics_feature_names else [f"feat_{i}" for i in range(len(attr_mean))]
            feature_importance["transcriptomics"] = dict(zip(feature_names, attr_mean))

        if "chemical" in inputs:
            attributions = ig.attribute(
                inputs["chemical"],
                baselines=baselines["chemical"],
                target=0,  # For regression, attribute to the output scalar
                n_steps=n_steps,
                additional_forward_args=(inputs["transcriptomics"] if "transcriptomics" in inputs else None)
            )
            # Average attributions across samples and normalize
            attr_mean = attributions.abs().mean(dim=0).cpu().numpy()
            attr_sum = attr_mean.sum()
            if attr_sum > 0:
                attr_mean /= attr_sum  # Normalize to sum to 1
            # Map to feature names if provided, otherwise use indices
            feature_names = self.chemical_feature_names if self.chemical_feature_names else [f"feat_{i}" for i in range(len(attr_mean))]
            feature_importance["chemical"] = dict(zip(feature_names, attr_mean))

        return feature_importance