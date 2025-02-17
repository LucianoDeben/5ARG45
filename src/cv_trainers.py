import logging

import numpy as np
from evaluation import evaluate_model
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Subset
from training import train_model
import torch



class BaseNestedCVTrainer(ABC):
    def __init__(self, dataset, nested_splits, evaluation_fn, use_inner_cv: bool = True):
        """
        Args:
            dataset: The dataset object; must implement to_numpy() returning (X, y)
            nested_splits: List of tuples (outer_train_idx, outer_test_idx, inner_splits)
            evaluation_fn: A callable evaluating (y_true, y_pred) → metrics dict.
            use_inner_cv (bool): If True, iterate over all inner splits; if False, use only the first inner split.
        """
        self.dataset = dataset
        self.nested_splits = nested_splits
        self.evaluation_fn = evaluation_fn
        self.use_inner_cv = use_inner_cv
        # Pre-load data for convenience.
        self.X_all, self.y_all = self.dataset.to_numpy()

    def run(self):
        outer_metrics = []
        outer_split_details = []
        for fold, (outer_train_idx, outer_test_idx, inner_splits) in enumerate(self.nested_splits):
            logging.info(f"--- Outer Fold {fold+1} ---")
            inner_metrics_list = []
            if self.use_inner_cv:
                for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
                    metrics_inner = self.train_inner_model(inner_train_idx, inner_val_idx)
                    logging.info(f"Outer Fold {fold+1}, Inner Fold {inner_fold+1}: {metrics_inner}")
                    inner_metrics_list.append(metrics_inner)
            else:
                # Use only the first inner split.
                inner_train_idx, inner_val_idx = inner_splits[0]
                metrics_inner = self.train_inner_model(inner_train_idx, inner_val_idx)
                inner_metrics_list.append(metrics_inner)
            # Aggregate inner metrics.
            keys = list(inner_metrics_list[0].keys())
            inner_mean = {k: np.mean([m[k] for m in inner_metrics_list]) for k in keys}
            inner_std = {k: np.std([m[k] for m in inner_metrics_list]) for k in keys}
            logging.info(f"Outer Fold {fold+1} Inner CV: Mean={inner_mean}, Std={inner_std}")
            outer_split_details.append((inner_mean, inner_std))
            
            metrics_outer = self.train_outer_model(outer_train_idx, outer_test_idx)
            logging.info(f"Outer Fold {fold+1} Test Metrics: {metrics_outer}")
            outer_metrics.append(metrics_outer)
        keys = list(outer_metrics[0].keys())
        overall_mean = {k: np.mean([m[k] for m in outer_metrics]) for k in keys}
        overall_std = {k: np.std([m[k] for m in outer_metrics]) for k in keys}
        logging.info(f"Overall Metrics: Mean={overall_mean}, Std={overall_std}")
        return outer_metrics, (overall_mean, overall_std), outer_split_details

    @abstractmethod
    def train_inner_model(self, inner_train_idx, inner_val_idx):
        """Train on the inner split and return predictions (or metrics) on the inner validation set."""
        pass

    @abstractmethod
    def train_outer_model(self, outer_train_idx, outer_test_idx):
        """Train on the outer training set and return predictions (or metrics) on the outer test set."""
        pass


class SklearnNestedCVTrainer(BaseNestedCVTrainer):
    def __init__(self, dataset, nested_splits, evaluation_fn, model_builder, use_inner_cv: bool = True):
        """
        Args:
            model_builder: A callable that returns a fresh scikit-learn model instance.
            use_inner_cv: Whether to iterate over all inner splits or use only the first.
        """
        super().__init__(dataset, nested_splits, evaluation_fn, use_inner_cv=use_inner_cv)
        self.model_builder = model_builder

    def train_inner_model(self, inner_train_idx, inner_val_idx):
        X_train = self.X_all[inner_train_idx]
        y_train = self.y_all[inner_train_idx]
        X_val = self.X_all[inner_val_idx]
        model = self.model_builder()
        model.fit(X_train, y_train)
        return model.predict(X_val)

    def train_outer_model(self, outer_train_idx, outer_test_idx):
        X_train = self.X_all[outer_train_idx]
        y_train = self.y_all[outer_train_idx]
        X_test = self.X_all[outer_test_idx]
        model = self.model_builder()
        model.fit(X_train, y_train)
        return model.predict(X_test)
    
class PyTorchNestedCVTrainer(BaseNestedCVTrainer):
    def __init__(self, dataset, nested_splits, evaluation_fn, model_builder, train_params, use_inner_cv: bool = True):
        """
        Args:
            model_builder: A callable that returns a fresh PyTorch model.
            train_params: Dictionary with training parameters (epochs, batch_size, device, criterion, etc.)
            use_inner_cv: Whether to iterate over all inner splits or use only the first.
        """
        super().__init__(dataset, nested_splits, evaluation_fn, use_inner_cv=use_inner_cv)
        self.model_builder = model_builder
        self.train_params = train_params
        self.device = train_params.get("device", "cpu")
        self.batch_size = train_params.get("batch_size", 32)

    def _create_loader(self, indices, shuffle=False):
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=shuffle)

    def train_inner_model(self, inner_train_idx, inner_val_idx):
        train_loader = self._create_loader(inner_train_idx, shuffle=True)
        val_loader = self._create_loader(inner_val_idx, shuffle=False)
        model_inner = self.model_builder()
        model_inner.to(self.device)
        optimizer = self.train_params.get("optimizer_fn")(
            model_inner.parameters(), lr=self.train_params.get("learning_rate", 1e-3)
        )
        train_model(
            model=model_inner,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=self.train_params.get("criterion"),
            optimizer=optimizer,
            scheduler=self.train_params.get("scheduler", None),
            epochs=self.train_params.get("epochs", 50),
            device=self.device,
            gradient_clipping=self.train_params.get("gradient_clipping", 5.0),
            early_stopping_patience=self.train_params.get("early_stopping_patience", 10),
            model_name=f"OuterFoldInner",
            use_mixed_precision=self.train_params.get("use_mixed_precision", False),
        )
        # Use our evaluate_model function to get metrics dictionary.
        metrics_inner = evaluate_model(
            model_inner, val_loader, self.train_params.get("criterion"), device=self.device
        )
        return metrics_inner

    def train_outer_model(self, outer_train_idx, outer_test_idx):
        train_loader = self._create_loader(outer_train_idx, shuffle=True)
        test_loader = self._create_loader(outer_test_idx, shuffle=False)
        model_outer = self.model_builder()
        model_outer.to(self.device)
        optimizer = self.train_params.get("optimizer_fn")(
            model_outer.parameters(), lr=self.train_params.get("learning_rate", 1e-3)
        )
        train_model(
            model=model_outer,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=self.train_params.get("criterion"),
            optimizer=optimizer,
            scheduler=self.train_params.get("scheduler", None),
            epochs=self.train_params.get("epochs", 50),
            device=self.device,
            gradient_clipping=self.train_params.get("gradient_clipping", 5.0),
            early_stopping_patience=self.train_params.get("early_stopping_patience", 10),
            model_name=f"OuterFold",
            use_mixed_precision=self.train_params.get("use_mixed_precision", False),
        )
        metrics_outer = evaluate_model(
            model_outer, test_loader, self.train_params.get("criterion"), device=self.device
        )
        return metrics_outer
