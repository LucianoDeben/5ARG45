# src/models/base_module.py
import torch
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Any, Optional, List, Tuple

class DrugResponseModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Default metrics if none provided
        if metrics is None:
            self.metrics = {
                "train": self._get_default_metrics(),
                "val": self._get_default_metrics(),
                "test": self._get_default_metrics(),
            }
        else:
            self.metrics = metrics
        
        self.save_hyperparameters(ignore=["model"])

    def _get_default_metrics(self) -> Dict[str, torchmetrics.Metric]:
        return {
            "mse": torchmetrics.MeanSquaredError(),
            "mae": torchmetrics.MeanAbsoluteError(),
            "r2": torchmetrics.R2Score(),
            "pearson": torchmetrics.PearsonCorrCoef(),
        }
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    def _step(self, batch, batch_idx, stage):
        output = self(batch)
        loss = torch.nn.functional.mse_loss(output, batch["viability"])
        
        # Update metrics
        for name, metric in self.metrics[stage].items():
            metric(output, batch["viability"])
            self.log(f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
    
    def on_test_epoch_end(self):
        # You can implement custom visualizations here
        pass