import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_metrics(y_true, y_pred):
    """
    Evaluate regression metrics.

    Args:
        y_true: True target values (NumPy array or PyTorch tensor).
        y_pred: Predicted target values (NumPy array or PyTorch tensor).

    Returns:
        Tuple of mean squared error, mean absolute error, R^2, and Pearson correlation coefficient.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_coef, _ = pearsonr(y_true.flatten(), y_pred.flatten())

    return mse, mae, r2, pearson_coef


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
    model,
    test_loader,
    criterion,
    device="cpu",
    calculate_metrics=True,
):
    """
    Evaluate a PyTorch model.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): Loss function (e.g., MSELoss).
        device (str): Device to run evaluation on.
        calculate_metrics (bool): Whether to compute additional metrics.

    Returns:
        dict: Metrics including MSE, MAE, R², Pearson Correlation.
    """
    model.to(device)
    model.eval()

    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            y_true.append(y_batch)
            y_pred.append(outputs)

    # Finalize loss computation
    test_loss /= len(test_loader.dataset)
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    metrics = {"MSE": test_loss}
    if calculate_metrics:
        mse, mae, r2, pearson_coef = evaluate_regression_metrics(y_true, y_pred)
        metrics.update({"MAE": mae, "R²": r2, "Pearson Correlation": pearson_coef})

    return metrics
