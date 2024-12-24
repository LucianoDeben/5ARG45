import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training import _forward_pass


def weighted_score_func(y_true, y_pred):
    r2_val = r2_score(y_true, y_pred)
    pearson_val, _ = pearsonr(y_true, y_pred)
    return 0.5 * r2_val + 0.5 * pearson_val


def evaluate_shallow_model(model, X_test, y_test, calculate_metrics=True):
    """
    Evaluate a shallow model on a test set.

    Args:
        model: Trained shallow model (e.g., LinearRegression, Ridge).
        X_test: Test features (numpy array or pandas DataFrame).
        y_test: True target values (numpy array or pandas Series).
        calculate_metrics (bool): If True, calculate additional regression metrics.

    Returns:
        Tuple[float, np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
            - test_loss (float): Mean squared error on the test set.
            - y_true (np.ndarray): True target values.
            - y_pred (np.ndarray): Predicted target values.
            - metrics (dict, optional): Additional metrics if `calculate_metrics` is True.
    """
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Compute test loss (MSE)
    test_loss = mean_squared_error(y_test, y_pred)

    # Convert y_test to numpy array if not already
    y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test

    if calculate_metrics:
        mse, mae, r2, pearson_coef = evaluate_regression_metrics(y_true, y_pred)

        # Compute weighted score (average of R² and Pearson)
        weighted_score = 0.5 * r2 + 0.5 * pearson_coef

        metrics = {
            "MSE": mse,
            "MAE": mae,
            "R²": r2,
            "Pearson Correlation": pearson_coef,
            "Weighted Score": weighted_score,
        }
        return test_loss, metrics
    return test_loss


def evaluate_model(
    model, test_loader, criterion=None, device="cpu", calculate_metrics=True
):
    """
    Evaluate a trained (unimodal or multimodal) model on a test set.

    Args:
        model (nn.Module): Trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (callable, optional): Loss function.
        device (str): "cpu" or "cuda".
        calculate_metrics (bool): If True, calculate additional regression metrics.

    Returns:
        dict: Dictionary with keys like "MSE", "MAE", "R²", "Pearson Correlation".
    """
    model.to(device)
    model.eval()

    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Extract inputs
            if isinstance(batch, dict):  # Multimodal case
                features = batch["features"].to(device)  # Unperturbed expression
                labels = batch["labels"].to(device)  # Perturbed expression
                additional_inputs = batch.get(
                    "smiles", None
                )  # Additional inputs like SMILES
                # Forward pass
                outputs = model(features, additional_inputs)
            else:  # Unimodal case
                *inputs, labels = batch
                labels = labels.to(device)
                outputs = _forward_pass(model, inputs, device)

            outputs = (
                outputs.squeeze(dim=-1)
                if outputs.dim() > 1 and outputs.size(-1) == 1
                else outputs
            )

            # Compute loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)

            # Store predictions and labels
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute metrics
    metrics = {}
    if criterion is not None:
        metrics["MSE"] = test_loss / len(test_loader.dataset)
    if calculate_metrics:
        mse, mae, r2, pearson_coef = evaluate_regression_metrics(all_labels, all_preds)
        metrics.update({"MAE": mae, "R2": r2, "PCC": pearson_coef})

    return metrics


def evaluate_regression_metrics(y_true, y_pred):
    """
    Evaluate regression metrics: MSE, MAE, R², Pearson correlation.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        tuple: (mse, mae, r2, pearson_coef)
    """
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    pearson_coef, _ = pearsonr(y_true.flatten(), y_pred.flatten())

    return mse, mae, r2, pearson_coef
