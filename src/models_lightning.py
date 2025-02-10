from models import FlexibleFCNN
import torch.nn as nn
import torch
from torch.optim import AdamW
import lightning as L  

class LitFCNN(L.LightningModule):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], output_dim=1,
                 activation_fn="relu", dropout_prob=0.2, residual=False,
                 norm_type="batchnorm", weight_init="kaiming", lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = FlexibleFCNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation_fn=activation_fn,
            dropout_prob=dropout_prob,
            residual=residual,
            norm_type=norm_type,
            weight_init=weight_init
        )
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
