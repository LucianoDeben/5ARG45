import numpy as np
import torch

from evaluation import evaluate_regression_metrics


# Updated Training Function with Best Practices
def train_unimodal_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    device,
    gradient_clipping=1.0,
    early_stopping_patience=5,
):
    model.to(device)
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience
            best_model_state = model.state_dict()  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    return train_losses, val_losses


def evaluate_unimodal_model(
    model, test_loader, criterion, device, calculate_metrics=True
):
    """
    Evaluate a unimodal model on a test set.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        criterion (torch.nn.Module): Loss function (e.g., MSELoss).
        device (torch.device): Device to run the evaluation on.
        calculate_metrics (bool): If True, return additional regression metrics.

    Returns:
        Tuple[float, np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
            - test_loss (float): Average test loss.
            - y_true (np.ndarray): True target values.
            - y_pred (np.ndarray): Predicted target values.
            - metrics (dict, optional): Dictionary of additional metrics if `calculate_metrics` is True.
    """
    model.to(device)
    model.eval()

    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Skip empty batches
            if X_batch.size(0) == 0:
                continue

            # Move data to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)

            # Compute batch loss
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)  # Accumulate weighted loss

            # Collect predictions and true values
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())

    # Finalize loss computation
    test_loss /= len(test_loader.dataset)

    # Convert collected values to numpy arrays
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    if calculate_metrics:
        mse, mae, r2, pearson_coef = evaluate_regression_metrics(
            y_true.flatten(), y_pred.flatten()
        )
        metrics = {
            "MSE": mse,
            "MAE": mae,
            "R²": r2,
            "Pearson Correlation": pearson_coef,
        }
        return test_loss, y_true, y_pred, metrics

    return test_loss, y_true, y_pred
