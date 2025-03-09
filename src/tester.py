import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

from src.config.config_utils import setup_logging
from src.data.datasets import DatasetFactory
from src.data.loaders import GCTXDataLoader

# Setup logging and device
logger = setup_logging()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Enhanced timestamp for output directory
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = f"results/run_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"Output directory: {output_dir}")

# Define a simple FCNN model for viability prediction
class ViabilityPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(ViabilityPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))  # Added BatchNorm
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Final output layer (regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Data inspection function
def inspect_data(dataloader, name="Dataset"):
    """Inspect the data distribution in the dataloader."""
    inputs_list = []
    targets_list = []
    
    logger.info(f"Inspecting {name}...")
    
    for inputs, targets in dataloader:
        inputs_list.append(inputs.numpy())
        targets_list.append(targets.numpy())
    
    inputs_all = np.vstack(inputs_list)
    targets_all = np.concatenate(targets_list)
    
    # Log basic statistics
    logger.info(f"{name} inputs shape: {inputs_all.shape}")
    logger.info(f"{name} targets shape: {targets_all.shape}")
    
    # Input statistics
    input_mean = np.mean(inputs_all)
    input_std = np.std(inputs_all)
    input_min = np.min(inputs_all)
    input_max = np.max(inputs_all)
    logger.info(f"{name} inputs: mean={input_mean:.4f}, std={input_std:.4f}, min={input_min:.4f}, max={input_max:.4f}")
    
    # Check for NaN/Inf values
    nan_count = np.isnan(inputs_all).sum()
    inf_count = np.isinf(inputs_all).sum()
    logger.info(f"{name} inputs: NaN count={nan_count}, Inf count={inf_count}")
    
    # Target statistics
    target_mean = np.mean(targets_all)
    target_std = np.std(targets_all)
    target_min = np.min(targets_all)
    target_max = np.max(targets_all)
    logger.info(f"{name} targets: mean={target_mean:.4f}, std={target_std:.4f}, min={target_min:.4f}, max={target_max:.4f}")
    
    # Plot histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(inputs_all.flatten(), bins=50, alpha=0.7)
    plt.title(f"{name} Input Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(targets_all, bins=50, alpha=0.7)
    plt.title(f"{name} Target Distribution")
    plt.xlabel("Viability")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name.replace(' ', '_').lower()}_distribution.png")
    plt.close()
    
    return {
        'input_mean': input_mean,
        'input_std': input_std,
        'target_mean': target_mean,
        'target_std': target_std
    }

# Load datasets
lincs = GCTXDataLoader("./data/processed/LINCS_CTRP_matched.gctx")
mixseq = GCTXDataLoader("./data/processed/MixSeq.gctx")

logger.info(f"MixSeq column metadata keys: {mixseq.get_column_metadata_keys()}")
logger.info(f"MixSeq row metadata keys: {mixseq.get_row_metadata_keys()}")

# Load the MixSeq datasets
train_ds_mixseq, val_ds_mixseq, test_ds_mixseq = DatasetFactory.create_and_split_datasets(
    gctx_loader=mixseq,
    dataset_type="transcriptomics",
    feature_space="landmark",
    nrows=None,
    test_size=0.1,
    val_size=0.1,
    random_state=42,
    chunk_size=10000,
    group_by=None,
    stratify_by=None,
)

# Load the LINCS datasets
train_ds_lincs, val_ds_lincs, test_ds_lincs = DatasetFactory.create_and_split_datasets(
    gctx_loader=lincs,
    dataset_type="transcriptomics",
    feature_space="landmark",
    nrows=None,
    test_size=0.1,
    val_size=0.1,
    random_state=50,
    chunk_size=10000,
    group_by=None,
    stratify_by=None,
)

# Create all the DataLoaders
train_dl_mixseq = DataLoader(train_ds_mixseq, batch_size=32, shuffle=True)
val_dl_mixseq = DataLoader(val_ds_mixseq, batch_size=32, shuffle=False)
test_dl_mixseq = DataLoader(test_ds_mixseq, batch_size=32, shuffle=False)

# Create all the DataLoaders
train_dl_lincs = DataLoader(train_ds_lincs, batch_size=32, shuffle=True)
val_dl_lincs = DataLoader(val_ds_lincs, batch_size=32, shuffle=False)
test_dl_lincs = DataLoader(test_ds_lincs, batch_size=32, shuffle=False)

# Log dataset sizes
logger.info(f"LINCS dataset sizes: Train={len(train_ds_lincs)}, Val={len(val_ds_lincs)}, Test={len(test_ds_lincs)}")
logger.info(f"MixSeq dataset sizes: Train={len(train_ds_mixseq)}, Val={len(val_ds_mixseq)}, Test={len(test_ds_mixseq)}")

# Inspect the data distributions
lincs_train_stats = inspect_data(train_dl_lincs, "LINCS Train")
mixseq_train_stats = inspect_data(train_dl_mixseq, "MixSeq Train")



#######################################################
#TRAIN THE MODELS
#######################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import time
import os
from tqdm import tqdm
import pandas as pd
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
            
    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)

def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device).float().view(-1, 1)
        
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
        for inputs, targets in tqdm(data_loader, desc=f"Evaluating ({phase})", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)
            
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

def plot_metrics(train_metrics, val_metrics, output_dir):
    """Plot training metrics over time"""
    metrics = ['loss', 'mse', 'rmse', 'mae', 'r2', 'pearson']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3*len(metrics)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(train_metrics[metric], label=f'Train {metric}')
        ax.plot(val_metrics[metric], label=f'Val {metric}')
        ax.set_title(f'{metric.upper()} over epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_metrics.png")
    plt.close()

def plot_predictions(y_true, y_pred, title, output_path):
    """Plot predicted vs actual values"""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add diagonal perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    plt.title(f"{title}\nCorrelation: {corr:.4f}")
    plt.xlabel("Actual Viability")
    plt.ylabel("Predicted Viability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_learning_curve(train_losses, val_losses, lr_history, output_dir):
    """Plot learning curve with learning rate"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_losses, label='Train Loss', color=color)
    ax1.plot(val_losses, label='Val Loss', color='tab:green')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(lr_history, label='LR', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    
    plt.title('Learning Curve and Learning Rate Schedule')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curve.png")
    plt.close()

def train_and_evaluate(
    model, 
    train_loader,
    val_loader, 
    test_loader,
    external_loader,
    device,
    logger,
    output_dir,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=15,
    factor=0.5,
    min_lr=1e-6,
    grad_clip=1.0
):
    """Full training and evaluation pipeline"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience//3, verbose=True, min_lr=min_lr)
    early_stopping = EarlyStopping(patience=patience)
    
    # Metric tracking
    train_metrics_history = {
        'loss': [], 'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'pearson': []
    }
    val_metrics_history = {
        'loss': [], 'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'pearson': []
    }
    lr_history = []
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        
        # Validate
        val_metrics, val_targets, val_preds = evaluate(model, val_loader, criterion, device, "val")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Train R²: {train_metrics['r2']:.4f}, "
                    f"Val R²: {val_metrics['r2']:.4f}, "
                    f"Train Pearson: {train_metrics['pearson']:.4f}, "
                    f"Val Pearson: {val_metrics['pearson']:.4f}, "
                    f"LR: {current_lr:.6f}, "
                    f"Time: {time.time() - epoch_start:.2f}s")
        
        # Save metrics history
        for metric in train_metrics_history.keys():
            train_metrics_history[metric].append(train_metrics[metric])
            val_metrics_history[metric].append(val_metrics[metric])
        
        # Check for early stopping
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best weights
    early_stopping.restore_weights(model)
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    
    # Plot training metrics
    plot_metrics(train_metrics_history, val_metrics_history, output_dir)
    plot_learning_curve(
        train_metrics_history['loss'], 
        val_metrics_history['loss'], 
        lr_history, 
        output_dir
    )
    
    # Evaluate on internal test set
    logger.info("Evaluating on internal test set...")
    test_metrics, test_targets, test_preds = evaluate(model, test_loader, criterion, device, "test")
    logger.info(f"Internal Test - "
                f"Loss: {test_metrics['loss']:.4f}, "
                f"MSE: {test_metrics['mse']:.4f}, "
                f"RMSE: {test_metrics['rmse']:.4f}, "
                f"MAE: {test_metrics['mae']:.4f}, "
                f"R²: {test_metrics['r2']:.4f},"
                f"Pearson: {test_metrics['pearson']:.4f}")
    
    # Plot internal test predictions
    plot_predictions(
        test_targets, 
        test_preds, 
        "Internal Test: Predicted vs Actual Viability", 
        f"{output_dir}/internal_test_predictions.png"
    )
    
    # Evaluate on external dataset
    logger.info("Evaluating on external MixSeq dataset...")
    external_metrics, external_targets, external_preds = evaluate(model, external_loader, criterion, device, "external")
    logger.info(f"External Test - "
                f"Loss: {external_metrics['loss']:.4f}, "
                f"MSE: {external_metrics['mse']:.4f}, "
                f"RMSE: {external_metrics['rmse']:.4f}, "
                f"MAE: {external_metrics['mae']:.4f}, "
                f"R²: {external_metrics['r2']:.4f}, "
                f"Pearson: {external_metrics['pearson']:.4f}")
    
    # Plot external test predictions
    plot_predictions(
        external_targets, 
        external_preds, 
        "External Test (MixSeq): Predicted vs Actual Viability", 
        f"{output_dir}/external_test_predictions.png"
    )
    
    # Save detailed metrics to CSV
    results = {
        'Dataset': ['Train', 'Validation', 'Internal Test', 'External Test'],
        'Loss': [train_metrics_history['loss'][-1], val_metrics_history['loss'][-1], test_metrics['loss'], external_metrics['loss']],
        'MSE': [train_metrics_history['mse'][-1], val_metrics_history['mse'][-1], test_metrics['mse'], external_metrics['mse']],
        'RMSE': [train_metrics_history['rmse'][-1], val_metrics_history['rmse'][-1], test_metrics['rmse'], external_metrics['rmse']],
        'MAE': [train_metrics_history['mae'][-1], val_metrics_history['mae'][-1], test_metrics['mae'], external_metrics['mae']],
        'R²': [train_metrics_history['r2'][-1], val_metrics_history['r2'][-1], test_metrics['r2'], external_metrics['r2']],
        'Pearson': [train_metrics_history['pearson'][-1], val_metrics_history['pearson'][-1], test_metrics['pearson'], external_metrics['pearson']]
    }
    pd.DataFrame(results).to_csv(f"{output_dir}/performance_metrics.csv", index=False)
    
    # Save predictions
    test_df = pd.DataFrame({
        'Actual': test_targets,
        'Predicted': test_preds,
        'Error': test_targets - test_preds,
        'Set': 'Internal Test'
    })
    
    external_df = pd.DataFrame({
        'Actual': external_targets,
        'Predicted': external_preds,
        'Error': external_targets - external_preds,
        'Set': 'External Test'
    })
    
    combined_df = pd.concat([test_df, external_df])
    combined_df.to_csv(f"{output_dir}/predictions.csv", index=False)
    
    # Save model
    torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
    
    # Additional visualization: error distribution
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data=combined_df, x='Error', hue='Set', kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=combined_df, x='Actual', y='Predicted', hue='Set', fill=True, alpha=0.5)
    ax = plt.gca()
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.title('Density Plot: Predicted vs Actual')
    plt.xlabel('Actual Viability')
    plt.ylabel('Predicted Viability')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_analysis.png")
    plt.close()
    
    return {
        'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history,
        'test_metrics': test_metrics,
        'external_metrics': external_metrics,
        'test_predictions': (test_targets, test_preds),
        'external_predictions': (external_targets, external_preds)
    }

# Hyperparameter search function
def hyperparameter_search(
    train_loader, 
    val_loader, 
    test_loader, 
    external_loader, 
    input_dim,
    device,
    logger,
    output_dir,
    search_iterations=3
):
    """Perform a simple hyperparameter search"""
    hidden_dims_options = [
        [512, 256, 128],
        [1024, 512, 256, 128],
        [256, 256, 256],
    ]
    
    dropout_options = [0.2, 0.3, 0.4]
    lr_options = [1e-3, 5e-4, 1e-4]
    weight_decay_options = [1e-4, 1e-5, 1e-6]
    
    best_val_r2 = -float('inf')
    best_config = None
    best_results = None
    
    for i in range(search_iterations):
        # Randomly sample hyperparameters
        hidden_dims = np.random.choice(len(hidden_dims_options))
        dropout = np.random.choice(dropout_options)
        lr = np.random.choice(lr_options)
        wd = np.random.choice(weight_decay_options)
        
        # Create model with these hyperparameters
        model = ViabilityPredictor(
            input_dim=input_dim,
            hidden_dims=hidden_dims_options[hidden_dims],
            dropout_rate=dropout
        )
        
        # Create run directory
        run_dir = f"{output_dir}/hparam_search/run_{i+1}"
        os.makedirs(run_dir, exist_ok=True)
        
        # Log hyperparameters
        logger.info(f"Hyperparameter search iteration {i+1}/{search_iterations}")
        logger.info(f"Hidden dims: {hidden_dims_options[hidden_dims]}")
        logger.info(f"Dropout rate: {dropout}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Weight decay: {wd}")
        
        # Train and evaluate
        results = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            external_loader=external_loader,
            device=device,
            logger=logger,
            output_dir=run_dir,
            learning_rate=lr,
            weight_decay=wd,
            num_epochs=50  # Reduced for search
        )
        
        # Check if this is the best configuration
        val_r2 = results['val_metrics']['r2'][-1]
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_config = {
                'hidden_dims': hidden_dims_options[hidden_dims],
                'dropout_rate': dropout,
                'learning_rate': lr,
                'weight_decay': wd
            }
            best_results = results
            
        logger.info(f"Run {i+1} - Val R²: {val_r2:.4f}, Best so far: {best_val_r2:.4f}")
    
    # Log best configuration
    logger.info(f"Best hyperparameters: {best_config}")
    logger.info(f"Best validation R²: {best_val_r2:.4f}")
    logger.info(f"Internal test R²: {best_results['test_metrics']['r2']:.4f}")
    logger.info(f"External test R²: {best_results['external_metrics']['r2']:.4f}")
    
    # Save best configuration
    with open(f"{output_dir}/best_hyperparameters.txt", "w") as f:
        f.write(f"Best validation R²: {best_val_r2:.4f}\n")
        f.write(f"Internal test R²: {best_results['test_metrics']['r2']:.4f}\n")
        f.write(f"External test R²: {best_results['external_metrics']['r2']:.4f}\n")
        f.write("\nBest hyperparameters:\n")
        for key, value in best_config.items():
            f.write(f"{key}: {value}\n")
    
    return best_config, best_results

# Main execution
def main():
    # Set up hyperparameter search directory
    hp_search_dir = f"{output_dir}/hyperparameter_search"
    os.makedirs(hp_search_dir, exist_ok=True)
    
    # Perform hyperparameter search
    logger.info("Starting hyperparameter search...")
    input_dim = next(iter(train_dl_lincs))[0].shape[1]  # Get input dimension from data
    best_config, _ = hyperparameter_search(
        train_loader=train_dl_lincs,
        val_loader=val_dl_lincs,
        test_loader=test_dl_lincs,
        external_loader=test_dl_mixseq,
        input_dim=input_dim,
        device=device,
        logger=logger,
        output_dir=hp_search_dir,
        search_iterations=5  # Adjust based on your time constraints
    )
    
    # Create final model with best hyperparameters
    logger.info("Training final model with best hyperparameters...")
    final_model = ViabilityPredictor(
        input_dim=input_dim,
        hidden_dims=best_config['hidden_dims'],
        dropout_rate=best_config['dropout_rate']
    )
    
    # Create final model directory
    final_model_dir = f"{output_dir}/final_model"
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Train final model
    final_results = train_and_evaluate(
        model=final_model,
        train_loader=train_dl_lincs,
        val_loader=val_dl_lincs,
        test_loader=test_dl_lincs,
        external_loader=test_dl_mixseq,
        device=device,
        logger=logger,
        output_dir=final_model_dir,
        learning_rate=best_config['learning_rate'],
        weight_decay=best_config['weight_decay'],
        num_epochs=150  # Full training run
    )
    
    logger.info("Training and evaluation completed successfully!")
    
    return final_results

if __name__ == "__main__":
    main()