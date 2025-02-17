import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def get_regression_metrics(y_true, y_pred):
    """
    Evaluate regression metrics: MSE, MAE, R², Pearson correlation.

    Args:
        y_true (array-like or tensor): True values.
        y_pred (array-like or tensor): Predicted values.

    Returns:
        dict: {"MSE": mse, "MAE": mae, "R²": r2, "Pearson": pearson_coef}
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_coef, _ = pearsonr(y_true, y_pred)

    return {"MSE": mse, "MAE": mae, "R²": r2, "Pearson": pearson_coef}
