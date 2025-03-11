from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from src.config.config_utils import setup_logging
from src.data.datasets import DatasetFactory
from src.data.loaders import GCTXDataLoader
from src.utils.data_validation import validate_batch

# Setup logging and device
logger = setup_logging()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Enhanced timestamp for output directory
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = f"results/run_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"Output directory: {output_dir}")

# Define activation and normalization options
ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
}

NORM_LAYERS = {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm, "none": None}


class FlexibleFCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 256, 128, 64],
        output_dim=1,
        activation_fn="relu",
        dropout_prob=0.2,
        residual=False,
        norm_type="batchnorm",
        weight_init="kaiming",
    ):
        super(FlexibleFCNN, self).__init__()
        self.residual = residual

        # Ensure hidden_dims are consistent for residual
        if residual:
            hidden_dims = [hidden_dims[0]] * len(hidden_dims)

        self.activation = ACTIVATIONS.get(activation_fn.lower(), nn.ReLU)()
        self.norm_type = norm_type.lower()

        # Build layers
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            norm = (
                NORM_LAYERS.get(self.norm_type, nn.Identity)(dims[i + 1])
                if norm_type != "none"
                else nn.Identity()
            )
            self.norms.append(norm)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.output = nn.Linear(hidden_dims[-1], output_dim)

        self._initialize_weights(weight_init)

    def _initialize_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = x
        for layer, norm in zip(self.layers, self.norms):
            identity = out  # Save for residual

            # Main path
            out = layer(out)
            out = norm(out)

            # Optional residual
            if self.residual and (out.shape == identity.shape):
                out = out + identity

            out = self.activation(out)
            out = self.dropout(out)

        return self.output(out)


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson
    }


def plot_predictions(y_true, y_pred, title, output_path):
    """Plot true vs predicted values and save the figure"""
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
    metrics = calculate_metrics(y_true, y_pred)
    metrics_text = '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                 facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        # Validate batch has required keys
        validate_batch(batch, required_keys=["transcriptomics", "viability"])
        
        # Extract inputs and targets
        inputs = batch["transcriptomics"].to(device)
        targets = batch["viability"].to(device).float().view(-1, 1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = epoch_loss
    
    return metrics


def evaluate(model, data_loader, criterion, device, phase="val"):
    """Evaluate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating ({phase})", leave=False):
            # Validate batch has required keys
            validate_batch(batch, required_keys=["transcriptomics", "viability"])
            
            # Extract inputs and targets
            inputs = batch["transcriptomics"].to(device)
            targets = batch["viability"].to(device).float().view(-1, 1)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader.dataset)
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = epoch_loss
    
    return metrics, all_targets, all_preds


def main():
    # Load datasets
    logger.info("Loading datasets...")
    lincs = GCTXDataLoader("./data/processed/LINCS_small_updated.gctx")
    mixseq = GCTXDataLoader("./data/processed/MixSeq.gctx")

    # Create datasets with cell_id grouping
    logger.info("Creating LINCS datasets with cell_id grouping...")
    train_ds_lincs, val_ds_lincs, test_ds_lincs = DatasetFactory.create_and_split_transcriptomics(
        gctx_loader=lincs,
        feature_space="landmark",
        nrows=None,  # Using subset for faster iteration
        test_size=0.3,
        val_size=0.1,
        random_state=42,
        chunk_size=10000,
        group_by="cell_mfc_name",  # Group by cell ID to prevent data leakage
        stratify_by="viability_clipped",
    )

    # Create MixSeq datasets
    logger.info("Creating MixSeq datasets...")
    train_ds_mixseq, val_ds_mixseq, test_ds_mixseq = DatasetFactory.create_and_split_transcriptomics(
        gctx_loader=mixseq,
        feature_space="landmark",
        nrows=None,  # Use all data
        test_size=0.1,
        val_size=0.1,
        random_state=42,
        chunk_size=10000,
        group_by="cell_mfc_name",  # Consistent with LINCS grouping
        stratify_by="viability_clipped",
    )

    # Display dataset information
    logger.info(f"LINCS Training set size: {len(train_ds_lincs)}")
    logger.info(f"LINCS Validation set size: {len(val_ds_lincs)}")
    logger.info(f"LINCS Test set size: {len(test_ds_lincs)}")
    logger.info(f"MixSeq Test set size: {len(test_ds_mixseq)}")

    # Extract input dimensions from first sample
    first_sample = train_ds_lincs[0]
    input_dim = first_sample["transcriptomics"].shape[0]
    logger.info(f"Input dimension: {input_dim}")

    # Create DataLoaders
    batch_size = 256 # Smaller batch size for better convergence
    train_dl_lincs = DataLoader(train_ds_lincs, batch_size=batch_size, shuffle=True)
    val_dl_lincs = DataLoader(val_ds_lincs, batch_size=batch_size, shuffle=False)
    test_dl_lincs = DataLoader(test_ds_lincs, batch_size=batch_size, shuffle=False)
    train_dl_mixseq = DataLoader(train_ds_mixseq, batch_size=batch_size, shuffle=False)

    # Training parameters
    num_epochs = 50
    lr = 1e-3
    grad_clip = 1.0

    # Model and optimizer
    model = FlexibleFCNN(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],
        output_dim=1,
        activation_fn="relu",
        dropout_prob=0.3,
        residual=True,
        norm_type="batchnorm",
        weight_init="kaiming",
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    logger.info("Starting training...")
    train_metrics = []
    val_metrics = []

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        epoch_train_metrics = train_epoch(model, train_dl_lincs, criterion, optimizer, device, grad_clip)
        train_metrics.append(epoch_train_metrics)
        
        # Validation
        epoch_val_metrics, _, _ = evaluate(model, val_dl_lincs, criterion, device, phase="val")
        val_metrics.append(epoch_val_metrics)
        
        logger.info(f"Train Loss: {epoch_train_metrics['loss']:.4f}, RMSE: {epoch_train_metrics['rmse']:.4f}, Pearson: {epoch_train_metrics['pearson']:.4f}")
        logger.info(f"Val Loss: {epoch_val_metrics['loss']:.4f}, RMSE: {epoch_val_metrics['rmse']:.4f}, Pearson: {epoch_val_metrics['pearson']:.4f}")

    # Save model
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate on test sets
    logger.info("Evaluating on LINCS test set...")
    test_metrics_lincs, y_true_lincs, y_pred_lincs = evaluate(model, test_dl_lincs, criterion, device, phase="test")
    
    logger.info("Evaluating on MixSeq test set...")
    test_metrics_mixseq, y_true_mixseq, y_pred_mixseq = evaluate(model, train_dl_mixseq, criterion, device, phase="test")

    # Log test metrics
    logger.info("LINCS Test Metrics:")
    for key, value in test_metrics_lincs.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("MixSeq Test Metrics:")
    for key, value in test_metrics_mixseq.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Plot predictions
    logger.info("Plotting predictions...")
    plot_predictions(
        y_true_lincs, y_pred_lincs, 
        "LINCS Test Set Predictions", 
        os.path.join(output_dir, "lincs_predictions.png")
    )
    
    plot_predictions(
        y_true_mixseq, y_pred_mixseq, 
        "MixSeq Test Set Predictions", 
        os.path.join(output_dir, "mixseq_predictions.png")
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot([m['loss'] for m in train_metrics], label='Train Loss')
    plt.plot([m['loss'] for m in val_metrics], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Pearson correlation plot
    plt.subplot(1, 2, 2)
    plt.plot([m['pearson'] for m in train_metrics], label='Train Pearson')
    plt.plot([m['pearson'] for m in val_metrics], label='Val Pearson')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    plt.title('Training and Validation Pearson Correlation')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    
    logger.info(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()