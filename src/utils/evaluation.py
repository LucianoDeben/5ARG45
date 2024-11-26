from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_metrics(y_true, y_pred):
    """
    Evaluate regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Tuple of mean squared error, mean absolute error, R^2, and Pearson correlation coefficient
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_coef, _ = pearsonr(y_true, y_pred)
    return mse, mae, r2, pearson_coef
