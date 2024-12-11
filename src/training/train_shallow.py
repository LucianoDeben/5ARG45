from typing import Dict

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from utils.evaluation import evaluate_regression_metrics


def train_shallow_model(model, X_train, y_train, X_val, y_val, epochs=50):
    """
    Train shallow models with loss tracking for training and validation.

    Args:
        model: Shallow model (e.g., LinearRegression, Ridge).
        X_train: Training features.
        y_train: Training target values.
        X_val: Validation features.
        y_val: Validation target values.
        epochs (int): Number of epochs to train.

    Returns:
        Tuple of (train_losses, val_losses): Lists of losses for each epoch.
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Train the model
        model.fit(X_train, y_train)

        # Calculate training loss
        y_train_pred = model.predict(X_train)
        train_loss = mean_squared_error(y_train, y_train_pred)
        train_losses.append(train_loss)

        # Calculate validation loss
        y_val_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val, y_val_pred)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

    return train_losses, val_losses


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
