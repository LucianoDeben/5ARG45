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


def evaluate_model(model, test_loader, criterion, device="cpu", calculate_metrics=True):
    """
    Evaluate a trained model on a test set.

    Args:
        model (nn.Module): Trained PyTorch model.
        test_loader (DataLoader): DataLoader for test set.
        criterion (callable): Loss function.
        device (str): "cpu" or "cuda".
        calculate_metrics (bool): If True, calculate additional regression metrics.

    Returns:
        metrics (dict): Dictionary with keys like "MSE", "MAE", "R²", "Pearson Correlation".
    """
    model.to(device)
    model.eval()

    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            *X_batch, y_batch = batch
            y_batch = y_batch.to(device)
            outputs = _forward_pass(model, X_batch, device)
            outputs = (
                outputs.squeeze(dim=-1)
                if outputs.dim() > 1 and outputs.size(-1) == 1
                else outputs
            )

            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * y_batch.size(0)

            y_true.append(y_batch.cpu())
            y_pred.append(outputs.cpu())

    test_loss /= len(test_loader.dataset)
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    metrics = {"MSE": test_loss}
    if calculate_metrics:
        # Adjust to your use-case (e.g., if classification, you'd compute accuracy, etc.)
        _, mae, r2, pearson_coef = evaluate_regression_metrics(y_true, y_pred)
        # You might change or extend evaluate_regression_metrics to handle other tasks
        metrics.update({"MAE": mae, "R²": r2, "Pearson Correlation": pearson_coef})

    return metrics


def evaluate_multimodal_model(
    model, test_loader, criterion=None, device="cpu", calculate_metrics=True
):
    model.to(device)
    model.eval()

    test_loss = 0.0
    gene_preds, gene_labels = [], []
    viability_preds, viability_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device).float()
            smiles_tokens = batch["smiles_tokens"].to(device).long()
            dosages = batch["dosage"].to(device).float()

            gene_labels_batch = batch["labels"].to(device).float()
            viability_labels_batch = batch.get("viability", None)
            if viability_labels_batch is not None:
                viability_labels_batch = viability_labels_batch.to(device).float()

            outputs = model(features, smiles_tokens, dosages)

            if criterion is not None:
                if model.task_type == "multi-task":
                    loss = criterion(outputs, gene_labels_batch, viability_labels_batch)
                elif model.task_type == "gene-expression":
                    loss = criterion(outputs, gene_labels_batch, None)
                elif model.task_type == "viability":
                    loss = criterion(outputs, None, viability_labels_batch)
                test_loss += loss.item() * features.size(0)

            if "gene_expression" in outputs:
                gene_preds.append(outputs["gene_expression"].cpu())
                gene_labels.append(gene_labels_batch.cpu())
            if "viability" in outputs:
                viability_preds.append(outputs["viability"].cpu())
                viability_labels.append(viability_labels_batch.cpu())

    metrics = {}

    if gene_preds and gene_labels:
        gene_preds = torch.cat(gene_preds, dim=0).numpy()
        gene_labels = torch.cat(gene_labels, dim=0).numpy()
        gene_metrics = evaluate_regression_metrics(gene_labels, gene_preds)
        metrics["gene_expression_metrics"] = {
            "MSE": gene_metrics[0],
            "MAE": gene_metrics[1],
            "R2": gene_metrics[2],
            "PCC": gene_metrics[3],
        }

    if viability_preds and viability_labels:
        viability_preds = torch.cat(viability_preds, dim=0).squeeze().numpy()
        viability_labels = torch.cat(viability_labels, dim=0).squeeze().numpy()
        viability_metrics = evaluate_regression_metrics(
            viability_labels, viability_preds
        )
        metrics["viability_metrics"] = {
            "MSE": viability_metrics[0],
            "MAE": viability_metrics[1],
            "R2": viability_metrics[2],
            "PCC": viability_metrics[3],
        }

    if criterion is not None:
        metrics["Total Loss"] = test_loss / len(test_loader.dataset)

    return metrics


def evaluate_regression_metrics(y_true, y_pred):
    """
    Evaluate regression metrics: MSE, MAE, R², Pearson correlation.

    Args:
        y_true (array-like or tensor): True values.
        y_pred (array-like or tensor): Predicted values.

    Returns:
        tuple: (mse, mae, r2, pearson_coef)
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure the inputs are flattened
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_coef, _ = pearsonr(y_true, y_pred)

    return mse, mae, r2, pearson_coef
