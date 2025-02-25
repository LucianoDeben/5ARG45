import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Lasso, LinearRegression, Ridge
import scipy.stats as stats

from data_sets import LINCSDataset
from metrics import get_regression_metrics
from results import CVResults
from utils import run_multiple_iterations_models, train_val_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def train_deep_model_simple(X_train, y_train, X_val, y_val, X_test, 
                            model_factory, training_params, device="cpu"):
    """
    Trains a PyTorch deep learning model using a simple training loop.
    Converts input arrays into DataLoaders, trains for a fixed number of epochs,
    and returns predictions on X_test.
    
    Args:
        X_train, y_train, X_val, y_val, X_test: NumPy arrays.
        model_factory: Callable that returns a new PyTorch model instance.
        training_params: Dict with keys: epochs, batch_size, learning_rate.
        device: 'cpu' or 'cuda'.
    
    Returns:
        y_pred: Numpy array of predictions on X_test.
    """
    # Create TensorDatasets and DataLoaders for training and validation
    batch_size = training_params.get("batch_size", 32)
    epochs = training_params.get("epochs", 20)
    learning_rate = training_params.get("learning_rate", 1e-3)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                  torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(),
                                torch.from_numpy(y_val).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the model
    model = model_factory().to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Simple training loop (no early stopping implemented for simplicity)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Optionally evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # After training, evaluate on test set
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
    y_pred = np.concatenate(predictions, axis=0).flatten()
    return y_pred

# -----------------------------
# Iterative Deep Model Evaluation Function
# -----------------------------
def run_multiple_iterations_deep(X, y, model_factory, n_iterations=20,
                                 train_size=0.6, val_size=0.2, test_size=0.2,
                                 random_state=42, metric_fn=get_regression_metrics,
                                 training_params=None, device="cpu"):
    """
    Runs multiple random splits and trains a PyTorch deep learning model.
    
    Parameters:
        X, y : array-like (features and targets)
        model_factory : Callable that returns a new PyTorch model instance.
        n_iterations : number of iterations.
        train_size, val_size, test_size : fractions summing to 1.0.
        random_state : seed for reproducibility.
        metric_fn : function to compute regression metrics.
        training_params : dict with training parameters (epochs, batch_size, learning_rate).
        device : 'cpu' or 'cuda'.
    
    Returns:
        A CVResults object aggregating the metrics.
    """
    if training_params is None:
        training_params = {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3}
    
    metrics_collection = {"DeepModel": {}}
    
    for i in range(n_iterations):
        seed = random_state + i if random_state is not None else None
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
            X, y, train_size, val_size, test_size, random_state=seed
        )
        # For deep models, we use train and validation separately.
        y_pred = train_deep_model_simple(X_train, y_train, X_val, y_val, X_test,
                                         model_factory, training_params, device=device)
        metrics = metric_fn(y_test, y_pred)
        for metric_name, value in metrics.items():
            if metric_name not in metrics_collection["DeepModel"]:
                metrics_collection["DeepModel"][metric_name] = []
            metrics_collection["DeepModel"][metric_name].append(value)
    
    # Compute mean and std for each metric.
    results_summary = {}
    for model_name, metric_lists in metrics_collection.items():
        mean_metrics = {m: np.mean(vals) for m, vals in metric_lists.items()}
        std_metrics = {m: np.std(vals) for m, vals in metric_lists.items()}
        results_summary[model_name] = {"mean": mean_metrics, "std": std_metrics}
    
    return CVResults(results_summary)

# -----------------------------
# Example: Simple Deep Model Definition and Usage
# -----------------------------
class SimpleFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(SimpleFeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Dummy data: 1000 samples, 10 features.
    dataset = LINCSDataset(gctx_path="../data/processed/LINCS.gctx")
    X, y = dataset.get_pandas()
    
    # Define a model factory for our deep model.
    def deep_model_factory():
        # Input dimension is the number of features.
        return SimpleFeedForward(input_dim=X.shape[1], hidden_dim=32)
    
    training_params = {"epochs": 10, "batch_size": 32, "learning_rate": 1e-3}
    
    # Run deep model training over 5 iterations.
    cv_results_deep = run_multiple_iterations_deep(
        X, y, model_factory=deep_model_factory,
        n_iterations=5, train_size=0.6, val_size=0.2, test_size=0.2,
        random_state=42, metric_fn=get_regression_metrics,
        training_params=training_params, device="cpu"
    )
    
    print("Aggregated Deep Model CV Results:")
    print(cv_results_deep.get_results_df())
    
    # Plot one metric, e.g., Pearson correlation.
    cv_results_deep.plot_metric("Pearson")
