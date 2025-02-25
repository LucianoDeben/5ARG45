from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Any, Dict, Tuple
from sklearn.base import BaseEstimator
from metrics import get_regression_metrics
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

class Trainer(ABC):
    """Abstract base class for training and evaluating models."""
    
    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def model(self) -> Any:
        """Get the underlying model."""
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> Dict:
        """Train the model on training data, with validation for monitoring."""
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict:
        """Evaluate the model on test data."""
        pass
    
class SklearnTrainer(Trainer):
    """Trainer for Scikit-Learn models."""
    
    def __init__(self, model: BaseEstimator, name: str):
        super().__init__(name)
        self.model = model

    @property
    def model(self) -> BaseEstimator:
        return self._model

    @model.setter
    def model(self, value: BaseEstimator):
        if not isinstance(value, BaseEstimator):
            raise ValueError("Model must be a Scikit-Learn estimator")
        self._model = value

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict:
        """Train the Scikit-Learn model."""
        self.model.fit(X_train, y_train)
        val_pred = self.model.predict(X_val)
        val_metrics = get_regression_metrics(y_val, val_pred)
        return {f"val_{key}": value for key, value in val_metrics.items()}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> Dict:
        """Evaluate the Scikit-Learn model on test data."""
        test_pred = self.model.predict(X_test)
        test_metrics = get_regression_metrics(y_test, test_pred)
        return {f"test_{key}": value for key, value in test_metrics.items()}

class PyTorchTrainer(Trainer):
    """Trainer for PyTorch models, supporting multimodal inputs."""
    
    def __init__(self, model: nn.Module, loss_function: nn.Module, 
                 optimizer: optim.Optimizer, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 32, num_epochs: int = 10, name: str = None):
        super().__init__(name)
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, value: nn.Module):
        if not isinstance(value, nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module")
        self._model = value

    def _create_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader for the given dataset."""
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, 
                         num_workers=4, pin_memory=True)

    def train(self, train_dataset, val_dataset, **kwargs) -> Dict:
        """Train the PyTorch model with early stopping."""
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False)

        best_val_loss = float('inf')
        patience = kwargs.get('patience', 3)
        epochs_no_improve = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)  # Handle multimodal input in model
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_true = []
            val_pred = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    val_true.append(labels.cpu())
                    val_pred.append(outputs.cpu())

            val_true = torch.cat(val_true).numpy()
            val_pred = torch.cat(val_pred).numpy()
            val_metrics = get_regression_metrics(val_true, val_pred)

            train_loss = train_loss / len(train_loader)
            val_loss = val_metrics["MSE"]  # Use MSE from get_regression_metrics for consistency
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if kwargs.get('save_model', False):
                    torch.save(self.model.state_dict(), kwargs.get('checkpoint_path', 'model.pth'))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break

        return {**{f"val_{key}": value for key, value in val_metrics.items()}, "history": history}

    def evaluate(self, test_dataset, **kwargs) -> Dict:
        """Evaluate the PyTorch model on test data."""
        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        self.model.eval()
        test_true = []
        test_pred = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                test_true.append(labels.cpu())
                test_pred.append(outputs.cpu())

        test_true = torch.cat(test_true).numpy()
        test_pred = torch.cat(test_pred).numpy()
        test_metrics = get_regression_metrics(test_true, test_pred)
        return {f"test_{key}": value for key, value in test_metrics.items()}