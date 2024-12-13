import logging

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn import clone
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import evaluate_model, evaluate_shallow_model
from training import train_model
from visualizations import (
    create_coefficients_visualization,
    create_feature_importance_visualization,
)


class ModelPipeline:
    def __init__(self, models, feature_sets, scoring="neg_mean_squared_error", cv=5):
        self.models = models
        self.feature_sets = feature_sets
        self.trained_models = {}
        self.results = {}
        self.scoring = scoring
        self.cv = cv

    def train_and_evaluate(self):
        for feature_name, (
            X_train,
            y_train,
            X_test,
            y_test,
        ) in self.feature_sets.items():
            for model_name, base_model in self.models.items():
                logging.debug(f"Training {model_name} on {feature_name}...")

                # Clone the base model to ensure no overwriting
                model = clone(base_model)

                # Perform cross-validation
                scores = cross_val_score(
                    model, X_train, y_train, scoring=self.scoring, cv=self.cv
                )
                mean_score = np.mean(scores)
                std_score = np.std(scores)

                # Train the model on the full training data
                trained_model = model.fit(X_train, y_train)

                # Store the trained model
                self.trained_models[(feature_name, model_name)] = trained_model

                # Evaluate the model on the test set
                test_loss, metrics = evaluate_shallow_model(
                    trained_model, X_test, y_test
                )

                # Add mean and std of cross-validation scores to the results
                metrics["CV Mean Score"] = mean_score
                metrics["CV Std Score"] = std_score

                # Store the evaluation metrics
                self.results[(feature_name, model_name)] = metrics

    def get_results(self):
        results_df = pd.DataFrame.from_dict(self.results, orient="index")
        results_df.index = pd.MultiIndex.from_tuples(
            results_df.index, names=["Feature Set", "Regression Model Type"]
        )
        return results_df.sort_index()

    def visualize_coefficients(self, top_n=10, save_path=None):
        create_coefficients_visualization(
            self.trained_models, self.feature_sets, top_n, save_path=save_path
        )


class NonLinearModelPipeline:
    def __init__(
        self, models, feature_sets, param_distributions, scoring="r2", cv=5, n_iter=20
    ):
        """
        Initialize the NonLinearModelPipeline.

        Args:
            models (dict): Dictionary with model names as keys and model instances as values.
            feature_sets (dict): Dictionary of feature sets with keys (e.g., "TF Data", "Gene Data")
                                 and values as tuples (X_train, y_train, X_test, y_test).
            param_distributions (dict): Parameter distributions for each model to be used in RandomizedSearchCV.
            scoring (str or callable): Scoring metric used for RandomizedSearchCV.
            cv (int): Number of folds for cross-validation.
            n_iter (int): Number of parameter settings sampled for RandomizedSearchCV.
        """
        self.models = models
        self.feature_sets = feature_sets
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.trained_models = {}
        self.results = {}

    def train_and_evaluate(self):
        for feature_name, (
            X_train,
            y_train,
            X_test,
            y_test,
        ) in self.feature_sets.items():
            for model_name, base_model in self.models.items():
                logging.debug(
                    f"Running RandomizedSearchCV for {model_name} on {feature_name}..."
                )

                # Check if param distributions exist for this model
                if model_name not in self.param_distributions:
                    logging.warning(
                        f"No parameter distributions for {model_name}. Using default parameters."
                    )
                    best_model = clone(base_model)
                    best_model.fit(X_train, y_train)
                    test_loss, metrics = evaluate_shallow_model(
                        best_model, X_test, y_test
                    )
                else:
                    # Perform RandomizedSearchCV
                    search = RandomizedSearchCV(
                        estimator=clone(base_model),
                        param_distributions=self.param_distributions[model_name],
                        n_iter=self.n_iter,
                        scoring=self.scoring,
                        cv=self.cv,
                        random_state=42,
                        n_jobs=-1,
                    )

                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    test_loss, metrics = evaluate_shallow_model(
                        best_model, X_test, y_test
                    )

                    # Store CV results as well (mean and std)
                    # search.cv_results_ has detailed info
                    # mean_test_score: mean of CV folds
                    # std_test_score: std of CV folds
                    mean_score = search.cv_results_["mean_test_score"][
                        search.best_index_
                    ]
                    std_score = search.cv_results_["std_test_score"][search.best_index_]

                    metrics["CV Mean Score"] = mean_score
                    metrics["CV Std Score"] = std_score

                # Store the trained model and metrics
                self.trained_models[(feature_name, model_name)] = best_model
                self.results[(feature_name, model_name)] = metrics

    def get_results(self):
        results_df = pd.DataFrame.from_dict(self.results, orient="index")
        results_df.index = pd.MultiIndex.from_tuples(
            results_df.index, names=["Feature Set", "Regression Model Type"]
        )
        return results_df.sort_index()

    def visualize_feature_importances(self, top_n=10, save_path=None):
        """
        Visualize feature importances for models that have feature_importances_ attribute.

        Args:
            top_n (int): Number of top features to display.
        """
        create_feature_importance_visualization(
            self.trained_models, self.feature_sets, top_n, save_path=save_path
        )


class DLModelsPipeline:
    def __init__(self, model_class, feature_sets, model_params, epochs=20):
        self.model_class = model_class
        self.feature_sets = feature_sets
        self.model_params = model_params
        self.trained_models = {}
        self.results = {}
        self.epochs = epochs

    def train_and_evaluate(self):
        for feature_name, (
            train_loader,
            val_loader,
            test_loader,
        ) in self.feature_sets.items():
            logging.debug(f"Training model on {feature_name}...")

            # Dynamically create the model for each feature set
            input_dim = train_loader.dataset[0][0].shape[0]
            model = self.model_class(input_dim=input_dim, **self.model_params)

            # Train the model on the full training data
            device = "cuda" if torch.cuda.is_available() else "cpu"
            criterion = torch.nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", patience=5, verbose=True
            )
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=self.epochs,
                device=device,
                gradient_clipping=1.0,
                early_stopping_patience=10,
            )

            # Store the trained model
            self.trained_models[feature_name] = model

            # Evaluate the model on the test set
            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
                calculate_metrics=True,
            )

            # Store the evaluation metrics
            self.results[feature_name] = metrics

    def get_results(self):
        results_df = pd.DataFrame.from_dict(self.results, orient="index")
        results_df.index.name = "Feature Set"
        return results_df.sort_index()
