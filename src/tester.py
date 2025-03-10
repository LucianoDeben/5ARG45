from typing import List, Optional
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
    def __init__(self, input_dim):
        super(ViabilityPredictor, self).__init__()
        
        # Simple model with just two hidden layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

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

# Load datasets
logger.info("Loading datasets...")
lincs = GCTXDataLoader("./data/processed/LINCS_CTRP_matched.gctx")
mixseq = GCTXDataLoader("./data/processed/MixSeq.gctx")

# THE KEY FIX: Enable proper grouping by cell_id to prevent data leakage
logger.info("Creating LINCS datasets with cell_id grouping...")
train_ds_lincs, val_ds_lincs, test_ds_lincs = DatasetFactory.create_and_split_datasets(
    gctx_loader=lincs,
    dataset_type="transcriptomics",
    feature_space="landmark",
    nrows=10000,
    test_size=0.1,
    val_size=0.1,
    random_state=42,
    chunk_size=10000,
    group_by="cell_mfc_name",  # THIS IS THE KEY CHANGE - group by cell ID to prevent data leakage
    stratify_by=None,
)

# Group MixSeq by cell_id too for consistency
logger.info("Creating MixSeq datasets...")
train_ds_mixseq, val_ds_mixseq, test_ds_mixseq = DatasetFactory.create_and_split_datasets(
    gctx_loader=mixseq,
    dataset_type="transcriptomics",
    feature_space="landmark",
    nrows=None,
    test_size=0.1,
    val_size=0.1,
    random_state=42,
    chunk_size=10000,
    group_by="cell_mfc_name",  # Consistent with LINCS grouping
    stratify_by=None,
)

# Smaller batch size for better convergence
batch_size = 128 

# Create DataLoaders
train_dl_lincs = DataLoader(train_ds_lincs, batch_size=batch_size, shuffle=True)
val_dl_lincs = DataLoader(val_ds_lincs, batch_size=batch_size, shuffle=False)
test_dl_lincs = DataLoader(test_ds_lincs, batch_size=batch_size, shuffle=False)
test_dl_mixseq = DataLoader(test_ds_mixseq, batch_size=batch_size, shuffle=False)


        
