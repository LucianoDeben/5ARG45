import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from sklearn.manifold import TSNE
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


def plot_pca_variance(pca, dataset_name="Dataset", save_path=None):
    """
    Create an explained variance (scree) plot.

    Args:
        pca (PCA): Fitted PCA object.
        dataset_name (str): Name of the dataset (default: "Dataset").
    """
    explained_var_cumul = np.cumsum(pca.explained_variance_ratio_) * 100
    fig = px.area(
        x=range(1, len(explained_var_cumul) + 1),
        y=explained_var_cumul,
        labels={"x": "# Components", "y": "Cumulative Explained Variance (%)"},
        title=f"{dataset_name}: PCA Explained Variance",
    )
    if save_path:
        try:
            fig.write_html(save_path)
        except Exception as e:
            logging.error(f"Error saving visualization: {e}")
    fig.show()


def create_pca_biplot(
    pca,
    X,
    Y,
    features,
    dataset_name="Dataset",
    top_n_loadings=10,
    sample_size=1000,
    loading_scale=10,
    dimension="2D",
    save_path=None,
):
    """
    Create a PCA biplot with Plotly for 2D or 3D.

    Args:
        pca (PCA): Fitted PCA object.
        X (pd.DataFrame): Scaled feature data.
        y (pd.Series): Target data.
        features (list): List of feature names.
        dataset_name (str): Name of the dataset.
        top_n_loadings (int): Number of top loadings to display.
        sample_size (int): Number of samples to plot.
        loading_scale (int): Scaling factor for loadings.
        dimension (str): '2D' or '3D' for the PCA dimensionality.

    Returns:
        plotly.graph_objects.Figure: The PCA biplot figure
    """

    # Downsample X for plotting
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    # Calculate loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings *= loading_scale  # Scale the loadings

    # Select top-n features contributing most to PC1, PC2, PC3
    component_scores = np.abs(loadings[:, :3]).sum(axis=1)
    top_features_idx = np.argsort(component_scores)[-top_n_loadings:]
    top_features = [features[i] for i in top_features_idx]
    top_loadings = loadings[top_features_idx]

    # Create 2D traces
    data_points_2d = go.Scatter(
        x=X["PC1"],
        y=X["PC2"],
        mode="markers",
        marker=dict(
            size=5,
            color=Y,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Viability"),
        ),
        name="Data Points (2D)",
        showlegend=False,
        visible=True if dimension == "2D" else False,
    )

    loadings_2d = [
        go.Scatter(
            x=[0, top_loadings[i, 0]],
            y=[0, top_loadings[i, 1]],
            mode="lines+text",
            line=dict(color="red", width=2),
            text=[None, feature],
            showlegend=False,
            visible=True if dimension == "2D" else False,
        )
        for i, feature in enumerate(top_features)
    ]

    # Create 3D traces
    data_points_3d = go.Scatter3d(
        x=X["PC1"],
        y=X["PC2"],
        z=X["PC3"],
        mode="markers",
        marker=dict(
            size=5,
            color=Y,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Viability"),
        ),
        name="Data Points (3D)",
        showlegend=False,
        visible=True if dimension == "3D" else False,
    )

    loadings_3d = [
        go.Scatter3d(
            x=[0, top_loadings[i, 0]],
            y=[0, top_loadings[i, 1]],
            z=[0, top_loadings[i, 2]],
            mode="lines+text",
            line=dict(color="red", width=2),
            text=[None, feature],
            showlegend=False,
            visible=True if dimension == "3D" else False,
        )
        for i, feature in enumerate(top_features)
    ]

    # Combine all traces, but only 2D or 3D will be visible based on 'dimension'
    all_traces = [data_points_2d] + loadings_2d + [data_points_3d] + loadings_3d

    # Initialize figure
    fig = go.Figure(data=all_traces)

    if dimension == "2D":
        fig.update_layout(
            title=f"{dataset_name}: 2D PCA Biplot (Explained Variance: {sum(pca.explained_variance_ratio_[:3]) * 100:.2f}%)",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
        )
    else:  # 3D
        fig.update_layout(
            title=f"{dataset_name}: 3D PCA Biplot (Explained Variance: {sum(pca.explained_variance_ratio_[:3]) * 100:.2f}%)",
            # Hide 2D axes in 3D mode
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            scene=dict(
                xaxis=dict(
                    visible=True,
                    title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)",
                ),
                yaxis=dict(
                    visible=True,
                    title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)",
                ),
                zaxis=dict(
                    visible=True,
                    title=f"PC3 ({pca.explained_variance_ratio_[2] * 100:.2f}%)",
                ),
            ),
        )

    # Update menus for toggling visibility (data, loadings)
    # Note: Dimension is fixed, so we only toggle among relevant traces.
    if dimension == "2D":
        # Indices: [data_points_2d] + loadings_2d
        # data_points_2d at index 0, loadings_2d follow
        # data_points_3d and loadings_3d are invisible
        visibility_show_all = [True] * (1 + len(loadings_2d)) + [False] * (
            1 + len(loadings_3d)
        )
        visibility_data_only = (
            [True] + [False] * len(loadings_2d) + [False] * (1 + len(loadings_3d))
        )
        visibility_loadings_only = (
            [False] + [True] * len(loadings_2d) + [False] * (1 + len(loadings_3d))
        )
    else:
        # dimension == '3D'
        # Indices: [data_points_2d, loadings_2d..., data_points_3d, loadings_3d...]
        # 2D are invisible anyway
        visibility_show_all = [False] * (1 + len(loadings_2d)) + [True] * (
            1 + len(loadings_3d)
        )
        visibility_data_only = (
            [False] * (1 + len(loadings_2d)) + [True] + [False] * len(loadings_3d)
        )
        visibility_loadings_only = (
            [False] * (1 + len(loadings_2d)) + [False] + [True] * len(loadings_3d)
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="Show All",
                        method="update",
                        args=[{"visible": visibility_show_all}, {}],
                    ),
                    dict(
                        label="Show Data Points Only",
                        method="update",
                        args=[{"visible": visibility_data_only}, {}],
                    ),
                    dict(
                        label="Show Loadings Only",
                        method="update",
                        args=[{"visible": visibility_loadings_only}, {}],
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
            ),
        ]
    )
    if save_path:
        try:
            fig.write_html(save_path)
        except Exception as e:
            logging.error(f"Error saving visualization: {e}")
    fig.show()


def create_tsne_plot(
    X,
    Y,
    target_column,
    dimension="2D",
    dataset_name="Dataset",
    sample_size=1000,
    perplexity=50,
    max_iter=1000,
    random_state=42,
    save_path=None,
):
    """
    Create a t-SNE plot in either 2D or 3D with Plotly Express.
    This version is simplified for a regression task and does not include toggling menus.

    Args:
        X (pd.DataFrame or np.ndarray): Feature data.
        y (pd.DataFrame or np.ndarray): Corresponding target data with a column `target_column`.
        target_column (str): Name of the target column in y.
        dimension (str): '2D' or '3D' for the t-SNE dimensionality.
        dataset_name (str): Title prefix for the plot.
        perplexity (float): t-SNE perplexity.
        n_iter (int): Number of t-SNE iterations.
        random_state (int): Random state for reproducibility.

    Returns:
        plotly.graph_objects.Figure: The t-SNE figure.
    """

    # Downsample X for plotting
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    # Convert X to a numpy array if needed
    if hasattr(X, "values"):
        X_values = X.values
    else:
        X_values = X

    # Reset indices and ensure y aligns if it's a DataFrame
    if hasattr(Y, "reset_index"):
        Y = Y.reset_index(drop=True)

    # Determine dimensionality
    n_components = 2 if dimension == "2D" else 3

    # Run t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )
    projections = tsne.fit_transform(X_values)

    # Create a DataFrame for the projections and the target column
    proj_df = pd.DataFrame(projections)
    proj_df[target_column] = Y

    if dimension == "2D":
        fig = px.scatter(
            proj_df,
            x=0,
            y=1,
            color=target_column,
            title=f"{dataset_name}: 2D t-SNE Plot",
            labels={0: "t-SNE 1", 1: "t-SNE 2"},
        )
    else:
        fig = px.scatter_3d(
            proj_df,
            x=0,
            y=1,
            z=2,
            color=target_column,
            title=f"{dataset_name}: 3D t-SNE Plot",
            labels={0: "t-SNE 1", 1: "t-SNE 2", 2: "t-SNE 3"},
        )
        # Increase marker size for 3D for better visibility
        fig.update_traces(marker_size=5)
    if save_path:
        try:
            fig.write_html(save_path)
        except Exception as e:
            logging.error(f"Error saving visualization: {e}")
    fig.show()


def create_coefficients_visualization(
    trained_models, feature_sets, top_n=10, save_path=None
):
    """
    Create an interactive Plotly visualization for the top N coefficients for all models and feature sets.
    Sets the initial view to the 'TF Data' feature set and 'Linear' model combination.
    """

    data_dict = {}
    feature_models_map = {}

    # Build data dictionary
    for (feature_name, model_name), trained_model in trained_models.items():
        X_train, _, _, _ = feature_sets[feature_name]

        # Extract feature names
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns
        else:
            feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

        # Retrieve coefficients or feature importances
        if hasattr(trained_model, "coef_"):
            coefficients = trained_model.coef_.ravel()
        elif hasattr(trained_model, "feature_importances_"):
            coefficients = trained_model.feature_importances_
        else:
            continue

        if len(feature_names) != len(coefficients):
            raise ValueError(
                f"Mismatch between feature names ({len(feature_names)}) and coefficients ({len(coefficients)})."
            )

        # Create DataFrame and select top_n
        coeff_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coefficients,
            }
        )
        coeff_df["AbsCoefficient"] = coeff_df["Coefficient"].abs()
        coeff_df = coeff_df.sort_values("AbsCoefficient", ascending=False).head(top_n)

        # Split into positive and negative
        pos_df = coeff_df[coeff_df["Coefficient"] > 0]
        neg_df = coeff_df[coeff_df["Coefficient"] <= 0]

        pos_trace = go.Bar(
            x=pos_df["Feature"],
            y=pos_df["Coefficient"],
            marker_color="blue",
            name="Positive",
            showlegend=True,
        )
        neg_trace = go.Bar(
            x=neg_df["Feature"],
            y=neg_df["Coefficient"],
            marker_color="red",
            name="Negative",
            showlegend=True,
        )

        data_dict[(feature_name, model_name)] = (pos_trace, neg_trace)
        if feature_name not in feature_models_map:
            feature_models_map[feature_name] = []
        if model_name not in feature_models_map[feature_name]:
            feature_models_map[feature_name].append(model_name)

    # Combine all traces
    all_traces = []
    index_map = {}
    for key, (pos_trace, neg_trace) in data_dict.items():
        start_idx = len(all_traces)
        all_traces.append(pos_trace)
        all_traces.append(neg_trace)
        index_map[key] = (start_idx, start_idx + 1)

    fig = go.Figure(data=all_traces)

    # Set default view to 'TF Data' and 'Linear'
    default_feature = "TF Data"
    default_model = "Linear"
    # Check if this combination exists, if not, pick the first available
    if (default_feature, default_model) not in data_dict:
        if feature_models_map:
            first_feature = sorted(feature_models_map.keys())[0]
            first_model = sorted(feature_models_map[first_feature])[0]
            default_feature, default_model = first_feature, first_model
        else:
            # No data at all
            default_feature = None
            default_model = None

    def generate_visibility_array(selected_feature, selected_model):
        visibility = [False] * len(all_traces)
        if (selected_feature, selected_model) in index_map:
            idx_pos, idx_neg = index_map[(selected_feature, selected_model)]
            visibility[idx_pos] = True
            visibility[idx_neg] = True
        return visibility

    # Set initial visibility
    if default_feature is not None and default_model is not None:
        initial_visibility = generate_visibility_array(default_feature, default_model)
        for i, vis in enumerate(initial_visibility):
            fig.data[i].visible = vis
        fig.update_layout(
            title=f"Top {top_n} Features: {default_feature} - {default_model}"
        )
    else:
        fig.update_layout(title=f"Top {top_n} Coefficients for All Models and Datasets")

    # Create dropdown buttons
    # Feature dropdown
    feature_buttons = []
    for feat in sorted(feature_models_map.keys()):
        # On feature selection, default to first model for that feature
        first_model_for_feat = sorted(feature_models_map[feat])[0]
        visibility = generate_visibility_array(feat, first_model_for_feat)
        feature_buttons.append(
            dict(
                label=feat,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Top {top_n} Features: {feat} - {first_model_for_feat}"},
                ],
            )
        )

    # Model dropdown
    # We'll just base it on the currently displayed feature set (which defaults to default_feature)
    # Without dynamic callbacks, we must stick to a chosen feature set for these model buttons.
    model_buttons = []
    if default_feature in feature_models_map:
        for m in sorted(feature_models_map[default_feature]):
            visibility = generate_visibility_array(default_feature, m)
            model_buttons.append(
                dict(
                    label=m,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"Top {top_n} Features: {default_feature} - {m}"},
                    ],
                )
            )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=feature_buttons,
                direction="down",
                showactive=True,
                x=0.3,
                xanchor="center",
                y=1.2,
                yanchor="top",
                pad={"r": 10, "t": 10},
                name="Feature Set",
            ),
            dict(
                buttons=model_buttons,
                direction="down",
                showactive=True,
                x=0.7,
                xanchor="center",
                y=1.2,
                yanchor="top",
                pad={"r": 10, "t": 10},
                name="Model Type",
            ),
        ],
        xaxis_title="Feature Name",
        yaxis_title="Coefficient Value",
        legend_title="Coefficient Sign",
    )
    if save_path:
        try:
            fig.write_html(save_path)
        except Exception as e:
            logging.error(f"Error saving visualization: {e}")
    fig.show()


def create_feature_importance_visualization(
    trained_models, feature_sets, top_n=10, save_path=None
):
    """
    Create a visualization of feature importances for models that have feature_importances_ attribute.
    For models without feature_importances_, it will skip them.

    Args:
        trained_models (dict): Dictionary of trained models with (feature_name, model_name) as keys.
        feature_sets (dict): Dictionary of feature sets with feature names as keys.
        top_n (int): Number of top features to display.
    """
    # Collect all traces by (feature_name, model_name)
    all_traces = []
    index_map = {}  # (feature_name, model_name) -> trace index
    feature_models_map = {}  # feature_name -> list of model_names

    trace_count = 0
    for (feature_name, model_name), trained_model in trained_models.items():
        if hasattr(trained_model, "feature_importances_"):
            X_train, _, _, _ = feature_sets[feature_name]
            if hasattr(X_train, "columns"):
                feature_names = X_train.columns
            else:
                feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

            importances = trained_model.feature_importances_
            df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            df = df.sort_values("Importance", ascending=False).head(top_n)

            trace = go.Bar(
                x=df["Feature"],
                y=df["Importance"],
                name=f"{feature_name} - {model_name}",
                visible=False,
            )
            all_traces.append(trace)
            index_map[(feature_name, model_name)] = trace_count
            trace_count += 1

            if feature_name not in feature_models_map:
                feature_models_map[feature_name] = []
            if model_name not in feature_models_map[feature_name]:
                feature_models_map[feature_name].append(model_name)
        else:
            logging.warning(
                f"{model_name} does not have feature_importances_. Skipping."
            )

    if not all_traces:
        logging.warning(
            "No models with feature_importances_ found. Cannot visualize feature importances."
        )
        return

    # Create the figure
    fig = go.Figure(data=all_traces)

    # Determine default feature set and model (alphabetically or first found)
    if feature_models_map:
        default_feature = sorted(feature_models_map.keys())[0]
        default_model = sorted(feature_models_map[default_feature])[0]
    else:
        # No data found
        default_feature = None
        default_model = None

    def generate_visibility_array(selected_feature, selected_model):
        visibility = [False] * len(all_traces)
        if (selected_feature, selected_model) in index_map:
            idx = index_map[(selected_feature, selected_model)]
            visibility[idx] = True
        return visibility

    # Set initial visibility to default
    if default_feature is not None and default_model is not None:
        initial_visibility = generate_visibility_array(default_feature, default_model)
        for i, vis in enumerate(initial_visibility):
            fig.data[i].visible = vis
        fig.update_layout(
            title=f"Top {top_n} Feature Importances: {default_feature} - {default_model}"
        )
    else:
        fig.update_layout(title=f"Top {top_n} Feature Importances")

    # Create dropdowns
    # 1) Feature Set Buttons: Select a feature set -> default to showing first model in that feature set
    feature_buttons = []
    for feat in sorted(feature_models_map.keys()):
        first_model_for_feat = sorted(feature_models_map[feat])[0]
        visibility = generate_visibility_array(feat, first_model_for_feat)
        feature_buttons.append(
            dict(
                label=feat,
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": f"Top {top_n} Feature Importances: {feat} - {first_model_for_feat}"
                    },
                ],
            )
        )

    # 2) Model Buttons: Select a model type -> show that model for the currently chosen feature set
    # Without dynamic linking, we assume the currently chosen feature set is the default one.
    # To properly link them, each model button sets visibility for (default_feature, model).
    model_set = set(m for models in feature_models_map.values() for m in models)
    model_buttons = []
    if default_feature is not None:
        for m in sorted(model_set):
            visibility = generate_visibility_array(default_feature, m)
            # If this model doesn't exist for default_feature, it will show blank (all False)
            model_buttons.append(
                dict(
                    label=m,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "title": f"Top {top_n} Feature Importances: {default_feature} - {m}"
                        },
                    ],
                )
            )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=feature_buttons,
                direction="down",
                showactive=True,
                x=0.3,
                xanchor="center",
                y=1.2,
                yanchor="top",
                pad={"r": 10, "t": 10},
                name="Feature Set",
            ),
            dict(
                buttons=model_buttons,
                direction="down",
                showactive=True,
                x=0.7,
                xanchor="center",
                y=1.2,
                yanchor="top",
                pad={"r": 10, "t": 10},
                name="Model Type",
            ),
        ],
        xaxis_title="Feature",
        yaxis_title="Importance",
        legend_title="",
    )
    if save_path:
        try:
            fig.write_html(save_path)
        except Exception as e:
            logging.error(f"Error saving visualization: {e}")
    fig.show()


def create_model_weighted_score_visualization(results_df, save_path=None):
    """
    Create a bar plot comparing the weighted scores of regression models grouped by feature sets.

    Args:
        results_df (pd.DataFrame): The results DataFrame with a MultiIndex (Feature Set, Regression Model Type)
                                   and columns including 'Weighted Score'.
        save_path (str, optional): File path to save the HTML plot. If None, the plot is not saved.
    """
    fig = px.bar(
        results_df.reset_index(),
        x="Regression Model Type",
        y="Weighted Score",
        color="Feature Set",
        barmode="group",
        title="Weighted Score of Regression Models",
        labels={"Weighted Score": "Weighted Score", "Regression Model Type": "Models"},
    )

    fig.update_layout(
        xaxis=dict(title="Regression Models"),
        yaxis=dict(title="Weighted Score"),
        legend=dict(title="Feature Set"),
    )

    if save_path:
        try:
            fig.write_html(save_path)
        except Exception as e:
            logging.error(f"Error saving visualization at {save_path}: {e}")

    fig.show()


def create_feature_set_weighted_score_visualization(results_df, save_path=None):
    """
    Create a bar plot comparing the weighted scores of feature sets grouped by regression models.

    Args:
        results_df (pd.DataFrame): The results DataFrame with a MultiIndex (Feature Set, Regression Model Type)
                                   and columns including 'Weighted Score'.
        save_path (str, optional): File path to save the HTML plot. If None, the plot is not saved.
    """
    fig = px.bar(
        results_df.reset_index(),
        x="Feature Set",
        y="Weighted Score",
        color="Regression Model Type",
        barmode="group",
        title="Weighted Score of Feature Sets by Models",
        labels={"Weighted Score": "Weighted Score", "Feature Set": "Feature Set"},
    )

    fig.update_layout(
        xaxis=dict(title="Feature Sets"),
        yaxis=dict(title="Weighted Score"),
        legend=dict(title="Regression Models"),
    )

    if save_path:
        try:
            fig.write_html(save_path)
        except Exception as e:
            logging.error(f"Error saving visualization at {save_path}: {e}")

    fig.show()


def create_feature_set_distribution_visualization(results_df, save_path=None):
    """
    Create a box plot showing the distribution of weighted scores across different feature sets.

    Args:
        results_df (pd.DataFrame): The results DataFrame with a MultiIndex (Feature Set, Regression Model Type)
                                   and columns including 'Weighted Score'.
        save_path (str, optional): File path to save the HTML plot. If None, the plot is not saved.
    """
    fig = px.box(
        results_df.reset_index(),
        x="Feature Set",
        y="Weighted Score",
        color="Feature Set",
        title="Weighted Score Distribution Across Feature Sets",
        labels={"Weighted Score": "Weighted Score", "Feature Set": "Feature Sets"},
        points="outliers",
    )

    fig.update_layout(
        xaxis=dict(title="Feature Sets"),
        yaxis=dict(title="Weighted Score"),
        legend=dict(title="Feature Sets"),
    )

    if save_path:
        try:
            fig.write_html(save_path)
        except Exception as e:
            logging.error(f"Error saving visualization at {save_path}: {e}")

    fig.show()
