import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import learning_curve


def get_model_predictions(model, X):
    """
    Get predictions from either a shallow or deep learning model.

    Args:
        model: The model (shallow or PyTorch).
        X: Input data (numpy array, pandas DataFrame, or torch Tensor).

    Returns:
        numpy array: Predicted values.
    """
    if hasattr(model, "predict"):  # Shallow model
        return model.predict(X)
    else:  # Deep learning model
        model.eval()
        with torch.no_grad():
            if isinstance(X, pd.DataFrame):
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
            elif isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X
            return model(X_tensor).cpu().numpy()


def plot_residuals_combined(models, X_test, y_test):
    """
    Plot residuals for multiple models in a single figure.

    Args:
        models (dict): Dictionary of model names and models.
        X_test: Test features.
        y_test: True target values.
    """
    plt.figure(figsize=(12, 8))
    for name, model in models.items():
        try:
            y_test_pred = get_model_predictions(model, X_test)
            residuals = y_test.flatten() - y_test_pred.flatten()
            sns.scatterplot(x=y_test_pred.flatten(), y=residuals, label=name)
        except Exception as e:
            print(f"Error in model {name}: {e}")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot for All Models")
    plt.legend()
    plt.show()


def plot_predictions_combined(models, X_test, y_test):
    """
    Plot predictions vs actual values for multiple models.

    Args:
        models (dict): Dictionary of model names and models.
        X_test: Test features.
        y_test: True target values.
    """
    plt.figure(figsize=(12, 8))
    for name, model in models.items():
        try:
            y_test_pred = get_model_predictions(model, X_test)
            sns.scatterplot(x=y_test.flatten(), y=y_test_pred.flatten(), label=name)
        except Exception as e:
            print(f"Error in model {name}: {e}")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction vs. Actual for All Models")
    plt.legend()
    plt.show()


def plot_error_boxplots(models, X_test, y_test):
    """
    Plot a boxplot of errors for multiple models.

    Args:
        models (dict): Dictionary of model names and models.
        X_test: Test features.
        y_test: True target values.
    """
    errors = {}
    for name, model in models.items():
        try:
            y_test_pred = get_model_predictions(model, X_test)
            errors[name] = y_test.flatten() - y_test_pred.flatten()
        except Exception as e:
            print(f"Error in model {name}: {e}")

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=pd.DataFrame(errors))
    plt.xlabel("Model")
    plt.ylabel("Error")
    plt.title("Boxplot of Errors for Different Models")
    plt.show()


def plot_learning_curves_combined(models, X_train, y_train):
    """
    Plot learning curves (train vs validation error) for shallow models.

    Args:
        models (dict): Dictionary of shallow models.
        X_train: Training features.
        y_train: True target values.
    """
    plt.figure(figsize=(12, 8))
    for name, model in models.items():
        if hasattr(model, "predict"):  # Only for shallow models
            try:
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
                )
                train_scores_mean = -train_scores.mean(axis=1)
                val_scores_mean = -val_scores.mean(axis=1)
                plt.plot(train_sizes, train_scores_mean, label=f"{name} Training Error")
                plt.plot(train_sizes, val_scores_mean, label=f"{name} Validation Error")
            except Exception as e:
                print(f"Error in model {name}: {e}")
    plt.xlabel("Training Size")
    plt.ylabel("Validation Error")
    plt.title("Learning Curves for All Models")
    plt.legend()
    plt.show()


def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models.

    Args:
        model: A tree-based model with `feature_importances_` attribute.
        feature_names (list): List of feature names.
        model_name (str): Name of the model.
    """
    if hasattr(model, "feature_importances_"):
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title(f"Feature Importance for {model_name}")
            plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(
                range(len(importances)),
                [feature_names[i] for i in indices],
                rotation=90,
            )
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.show()
        except Exception as e:
            print(f"Error in plotting feature importance for {model_name}: {e}")


def plot_loss_curves(train_losses, val_losses, model_name):
    """
    Plot training and validation loss curves.

    Args:
        train_losses (list): Training losses per epoch.
        val_losses (list): Validation losses per epoch.
        model_name (str): Name of the model.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves ({model_name})")
    plt.legend()
    plt.show()


def create_visualizations(models, X_test, y_test, train_losses=None, val_losses=None):
    """
    Create all visualizations for a given set of models and datasets.

    Args:
        models (dict): Dictionary of model names and models.
        X_test: Test features.
        y_test: True target values.
        train_losses (dict, optional): Training losses for each model.
        val_losses (dict, optional): Validation losses for each model.
    """
    print("Generating residual plots...")
    plot_residuals_combined(models, X_test, y_test)

    print("Generating prediction plots...")
    plot_predictions_combined(models, X_test, y_test)

    print("Generating error boxplots...")
    plot_error_boxplots(models, X_test, y_test)

    if train_losses and val_losses:
        print("Generating loss curves...")
        for name in models.keys():
            plot_loss_curves(train_losses[name], val_losses[name], name)
