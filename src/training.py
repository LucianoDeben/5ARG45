import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from metrics import get_regression_metrics
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _forward_pass(model, X_batch, device):
    """
    Perform a forward pass through the model, ensuring all inputs are on the correct device.
    
    Args:
        model (nn.Module): The PyTorch model.
        X_batch (list of torch.Tensor): List of input tensors.
        device (torch.device or str): The device to run the computation on.
    
    Returns:
        torch.Tensor: The model outputs.
    """
    # Move each input tensor to the target device if not already there.
    X_batch = [x.to(device) if x.device != device else x for x in X_batch]
    
    # Confirm that every tensor is on the proper device.
    for i, x in enumerate(X_batch):
        assert x.device.type == torch.device(device).type, (
            f"Input tensor {i} is on {x.device.type}, expected {torch.device(device).type}."
        )
    
    # Call the model using unpacked inputs.
    return model(*X_batch)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=50,
    device="cpu",
    gradient_clipping=5.0,
    early_stopping_patience=10,
    model_name="Model",
    use_mixed_precision=True,
):
    """
    A unified training loop for any PyTorch model. This loop handles:
      - Forward and backward passes with optional mixed precision.
      - Multiple input modalities (unimodal/multimodal) where the last element of a batch is assumed to be the target.
      - Gradient clipping and learning rate scheduling.
      - Early stopping based on validation loss.
    
    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (optional): Learning rate scheduler (expects scheduler.step(val_loss)).
        epochs (int): Maximum number of training epochs.
        device (str): Device to use, e.g., "cpu" or "cuda".
        gradient_clipping (float): Maximum norm for gradient clipping.
        early_stopping_patience (int): Number of epochs with no improvement to wait before stopping.
        model_name (str): Name of the model for logging purposes.
        use_mixed_precision (bool): If True, uses automatic mixed precision for faster GPU training.
    
    Returns:
        (train_losses, val_losses): Tuple of lists containing the training and validation loss per epoch.
    """
    model.to(device)
    scaler = GradScaler(enabled=use_mixed_precision)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            # Unpack batch assuming the last element is the target.
            *X_batch, y_batch = batch
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_mixed_precision):
                outputs = _forward_pass(model, X_batch, device)
                # If the output is of shape (..., 1), squeeze the final dimension.
                if outputs.dim() > 1 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(dim=-1)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()

            if gradient_clipping is not None:
                # Unscale before clipping to get the true gradients.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            scaler.step(optimizer)
            scaler.update()

            # Multiply the batch loss by the batch size for correct averaging later.
            running_train_loss += loss.item() * y_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad(), autocast(enabled=use_mixed_precision):
            for batch in val_loader:
                *X_batch, y_batch = batch
                y_batch = y_batch.to(device)
                outputs = _forward_pass(model, X_batch, device)
                if outputs.dim() > 1 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(dim=-1)
                val_loss = criterion(outputs, y_batch)
                running_val_loss += val_loss.item() * y_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Step the scheduler if provided (e.g., ReduceLROnPlateau expects a metric).
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}"
        )

        # Early stopping check: reset patience if validation loss improves.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save the best model state (moved to CPU for portability).
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info(
                    f"Early stopping triggered for {model_name} at epoch {epoch+1}."
                )
                break

    # Restore the best model weights.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


def train_multimodal_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=10,
    device="cpu",
    gradient_clipping=1.0,
    early_stopping_patience=5,
    model_name="Perturbinator",
    use_mixed_precision=True,
):
    model.to(device)
    scaler = GradScaler(enabled=use_mixed_precision)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        running_train_loss = 0.0
        num_train_samples = 0

        for batch in train_loader:
            features = batch["features"].to(device).float()
            gene_labels = batch["labels"].to(device).float()
            smiles_tokens = batch["smiles_tokens"].to(device).long()
            dosages = batch["dosage"].to(device).float()
            viability_labels = batch.get("viability", None)
            if viability_labels is not None:
                viability_labels = viability_labels.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_mixed_precision):
                outputs = model(features, smiles_tokens, dosages)

                # Calculate loss based on task type
                if model.task_type == "multi-task":
                    loss = criterion(outputs, gene_labels, viability_labels)
                elif model.task_type == "gene-expression":
                    loss = criterion(outputs, gene_labels, None)
                elif model.task_type == "viability":
                    loss = criterion(outputs, None, viability_labels)
                else:
                    raise ValueError(f"Unsupported task type: {model.task_type}")

            scaler.scale(loss).backward()

            if gradient_clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            scaler.step(optimizer)
            scaler.update()

            batch_size = features.size(0)
            running_train_loss += loss.item() * batch_size
            num_train_samples += batch_size

        epoch_train_loss = running_train_loss / num_train_samples
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad(), autocast(enabled=use_mixed_precision):
            for batch in val_loader:
                features = batch["features"].to(device).float()
                gene_labels = batch["labels"].to(device).float()
                smiles_tokens = batch["smiles_tokens"].to(device).long()
                dosages = batch["dosage"].to(device).float()
                viability_labels = batch.get("viability", None)
                if viability_labels is not None:
                    viability_labels = viability_labels.to(device).float()

                outputs = model(features, smiles_tokens, dosages)

                # Calculate validation loss
                if model.task_type == "multi-task":
                    val_loss = criterion(outputs, gene_labels, viability_labels)
                elif model.task_type == "gene-expression":
                    val_loss = criterion(outputs, gene_labels, None)
                elif model.task_type == "viability":
                    val_loss = criterion(outputs, None, viability_labels)
                else:
                    raise ValueError(f"Unsupported task type: {model.task_type}")

                batch_size = features.size(0)
                running_val_loss += val_loss.item() * batch_size
                num_val_samples += batch_size

        epoch_val_loss = running_val_loss / num_val_samples
        val_losses.append(epoch_val_loss)

        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs} - {model_name}, "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the model
    torch.save(model.state_dict(), f"./{model_name}.pt")

    return train_losses, val_losses



class NestedCVTrainer:
    """
    A unified nested cross-validation trainer that supports both scikit-learn and PyTorch models.
    
    For each outer fold:
      - The outer test set is held out for final evaluation.
      - The outer training set is further split using inner folds (for hyperparameter tuning
        or performance estimation).
    
    The trainer is framework-agnostic: 
      - For scikit-learn models, it uses the model's .fit() and .predict() methods.
      - For PyTorch models, the user must supply training and prediction callbacks.
    
    Attributes:
        dataset: The dataset (e.g., an instance of LINCSDataset) with a to_numpy() method.
        nested_splits: Nested splits as returned by nested_stratified_group_split.
        model_builder: A callable that returns a new model instance.
        model_type: Either "sklearn" or "pytorch".
        train_fn: (Optional) For PyTorch models—a callable that trains a model on given data.
                  Signature: train_fn(model, train_data, val_data, train_params) -> trained_model.
        predict_fn: (Optional) For PyTorch models—a callable that returns predictions.
                    Signature: predict_fn(model, test_data) -> y_pred.
        evaluation_fn: A callable to evaluate metrics. Defaults to evaluate_regression_metrics.
    """
    def __init__(
        self,
        dataset,
        nested_splits: List[Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]],
        model_builder: Callable[[], object],
        model_type: str = "sklearn",  # or "pytorch"
        train_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None,
        evaluation_fn: Callable = get_regression_metrics,
        train_params: Optional[dict] = None,
    ):
        self.dataset = dataset
        self.nested_splits = nested_splits
        self.model_builder = model_builder
        self.model_type = model_type.lower()
        self.train_fn = train_fn
        self.predict_fn = predict_fn
        self.evaluation_fn = evaluation_fn
        self.train_params = train_params if train_params is not None else {}

        if self.model_type not in ["sklearn", "pytorch"]:
            raise ValueError("model_type must be either 'sklearn' or 'pytorch'")
        
        # For PyTorch, train_fn and predict_fn must be provided.
        if self.model_type == "pytorch" and (self.train_fn is None or self.predict_fn is None):
            raise ValueError("For PyTorch models, please provide train_fn and predict_fn.")

    def run(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
        """
        Runs nested cross-validation training.
        
        Returns:
            outer_metrics: Array of shape (n_outer_splits, 4) with evaluation metrics per outer fold.
            overall_mean: Mean of metrics across outer folds.
            outer_split_details: List containing inner fold metrics for each outer fold.
        """
        # Load entire dataset in-memory (assumes dataset.to_numpy() returns (X, y))
        X_all, y_all = self.dataset.to_numpy()
        
        outer_metrics = []
        outer_split_details = []  # to store inner CV details for each outer fold

        for fold, (outer_train_idx, outer_test_idx, inner_splits) in enumerate(self.nested_splits):
            logging.info(f"--- Outer Fold {fold+1} ---")
            
            # Inner CV on outer training set:
            inner_metrics = []
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
                X_inner_train = X_all[inner_train_idx]
                y_inner_train = y_all[inner_train_idx]
                X_inner_val = X_all[inner_val_idx]
                y_inner_val = y_all[inner_val_idx]
                
                model_inner = self.model_builder()
                if self.model_type == "sklearn":
                    model_inner.fit(X_inner_train, y_inner_train)
                    y_pred_inner = model_inner.predict(X_inner_val)
                else:  # PyTorch
                    model_inner = self.train_fn(model_inner, (X_inner_train, y_inner_train),
                                                (X_inner_val, y_inner_val), self.train_params)
                    y_pred_inner = self.predict_fn(model_inner, X_inner_val)
                
                metrics_inner = self.evaluation_fn(y_inner_val, y_pred_inner)
                logging.info(f"Outer Fold {fold+1}, Inner Fold {inner_fold+1}: "
                             f"MSE={metrics_inner[0]:.4f}, MAE={metrics_inner[1]:.4f}, "
                             f"R²={metrics_inner[2]:.4f}, Pearson={metrics_inner[3]:.4f}")
                inner_metrics.append(metrics_inner)
            inner_metrics = np.array(inner_metrics)
            inner_mean = np.mean(inner_metrics, axis=0)
            inner_std = np.std(inner_metrics, axis=0)
            logging.info(f"Outer Fold {fold+1} Inner CV Mean: {inner_mean}, Std: {inner_std}")
            outer_split_details.append((inner_mean, inner_std))
            
            # Train final model on full outer training set:
            X_train_outer = X_all[outer_train_idx]
            y_train_outer = y_all[outer_train_idx]
            X_test_outer = X_all[outer_test_idx]
            y_test_outer = y_all[outer_test_idx]
            
            model_outer = self.model_builder()
            if self.model_type == "sklearn":
                model_outer.fit(X_train_outer, y_train_outer)
                y_pred_outer = model_outer.predict(X_test_outer)
            else:  # PyTorch
                model_outer = self.train_fn(model_outer, (X_train_outer, y_train_outer), None, self.train_params)
                y_pred_outer = self.predict_fn(model_outer, X_test_outer)
            
            metrics_outer = self.evaluation_fn(y_test_outer, y_pred_outer)
            logging.info(f"Outer Fold {fold+1} Test: MSE={metrics_outer[0]:.4f}, "
                         f"MAE={metrics_outer[1]:.4f}, R²={metrics_outer[2]:.4f}, "
                         f"Pearson={metrics_outer[3]:.4f}")
            outer_metrics.append(metrics_outer)
        
        outer_metrics = np.array(outer_metrics)
        overall_mean = np.mean(outer_metrics, axis=0)
        overall_std = np.std(outer_metrics, axis=0)
        logging.info(f"Overall Metrics: Mean={overall_mean}, Std={overall_std}")
        
        return outer_metrics, (overall_mean, overall_std), outer_split_details

