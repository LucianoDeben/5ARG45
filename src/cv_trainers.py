import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from typing import List, Tuple, Optional, Callable, Union

from evaluation import evaluate_model
from training import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Splitting Functions
# ---------------------------------------------------------------------------

def sampled_split(
    dataset, 
    test_size: float = 0.5, 
    random_state: int = 42, 
    stratify: bool = False, 
    stratify_col: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly splits the dataset indices into training and test sets.
    Optionally uses stratification.
    """
    indices = np.arange(len(dataset))
    row_metadata = dataset.get_row_metadata()
    
    if stratify:
        if stratify_col is None or stratify_col not in row_metadata.columns:
            raise ValueError("Stratification enabled but a valid stratify_col was not provided.")
        stratify_values = row_metadata[stratify_col].values
    else:
        stratify_values = None
        
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values
    )
    logging.info(f"Sampled split: {len(train_idx)} train samples, {len(test_idx)} test samples.")
    return train_idx, test_idx

def kfold_split(
    dataset, 
    n_splits: int = 5, 
    random_state: int = 42,
    kfold_type: str = "regular",  # Options: "regular", "stratified", "grouped"
    group_or_stratify_col: Optional[str] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates a proper k-fold split of the dataset.
    - "regular": uses KFold.
    - "stratified": uses StratifiedKFold (requires group_or_stratify_col to be provided).
    - "grouped": uses GroupKFold (requires group_or_stratify_col).
    
    Returns a list of (train_idx, test_idx) pairs.
    """
    indices = np.arange(len(dataset))
    row_metadata = dataset.get_row_metadata()
    
    if kfold_type == "regular":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(indices))
    elif kfold_type == "stratified":
        if group_or_stratify_col is None or group_or_stratify_col not in row_metadata.columns:
            raise ValueError("Stratified k-fold requires a valid group_or_stratify_col.")
        stratify_values = row_metadata[group_or_stratify_col].values
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(indices, stratify_values))
    elif kfold_type == "grouped":
        if group_or_stratify_col is None or group_or_stratify_col not in row_metadata.columns:
            raise ValueError("Grouped k-fold requires a valid group_or_stratify_col.")
        groups = row_metadata[group_or_stratify_col].values
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(indices, groups=groups))
    else:
        raise ValueError("Invalid kfold_type. Must be 'regular', 'stratified', or 'grouped'.")
    
    logging.info(f"K-fold ({kfold_type}) split: Generated {len(splits)} folds.")
    return splits

# ---------------------------------------------------------------------------
# Base CV Trainer
# ---------------------------------------------------------------------------

class BaseCVTrainer(ABC):
    def __init__(
        self, 
        dataset, 
        evaluation_fn: Callable, 
        n_splits: int = 20, 
        split_mode: str = "sampled",  # Options: "sampled" or "kfold"
        test_size: float = 0.5,
        random_state: int = 42,
        stratify: bool = False,         # Used in "sampled" mode or for kfold_type "stratified"
        kfold_type: str = "regular",      # Options: "regular", "stratified", "grouped" (only used if split_mode == "kfold")
        group_or_stratify_col: Optional[str] = None  # Column for stratification or grouping in k-fold mode.
    ):
        """
        Base trainer for cross-validation.
        
        Depending on split_mode, it either generates repeated random splits ("sampled")
        or a proper k-fold split ("kfold").
        """
        self.dataset = dataset
        self.evaluation_fn = evaluation_fn
        self.n_splits = n_splits
        self.split_mode = split_mode
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.kfold_type = kfold_type
        self.group_or_stratify_col = group_or_stratify_col
        
        # Get the full dataset as numpy arrays.
        self.X_all, self.y_all = self.dataset.to_numpy()
        
        # Generate splits.
        self.splits = []
        if self.split_mode == "sampled":
            for i in range(self.n_splits):
                rs = self.random_state + i
                train_idx, test_idx = sampled_split(
                    self.dataset, 
                    test_size=self.test_size, 
                    random_state=rs, 
                    stratify=self.stratify, 
                    stratify_col=self.group_or_stratify_col
                )
                self.splits.append((train_idx, test_idx))
        elif self.split_mode == "kfold":
            kf_splits = kfold_split(
                self.dataset, 
                n_splits=self.n_splits, 
                random_state=self.random_state, 
                kfold_type=self.kfold_type, 
                group_or_stratify_col=self.group_or_stratify_col
            )
            self.splits = kf_splits
        else:
            raise ValueError("Invalid split_mode. Must be 'sampled' or 'kfold'.")
        
        logging.info(f"Generated {len(self.splits)} splits using split_mode='{self.split_mode}'.")

    @abstractmethod
    def train_model(self, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train a model on the training split and return (y_true, y_pred) on the test split.
        """
        pass

    def run(self, result_key: Optional[str] = None):
        metrics_list = []
        for i, (train_idx, test_idx) in enumerate(self.splits):
            logging.info(f"Running split {i+1}/{len(self.splits)}...")
            y_true, y_pred = self.train_model(train_idx, test_idx)
            metrics = self.evaluation_fn(y_true, y_pred)
            logging.info(f"Split {i+1} metrics: {metrics}")
            metrics_list.append(metrics)
        # Aggregate metrics over all splits.
        keys = list(metrics_list[0].keys())
        overall_mean = {k: np.mean([m[k] for m in metrics_list]) for k in keys}
        overall_std = {k: np.std([m[k] for m in metrics_list]) for k in keys}
        logging.info(f"Overall CV Metrics: Mean={overall_mean}, Std={overall_std}")
        
        if result_key is None:
            result_key = "default"
        results = {result_key: {"mean": overall_mean, "std": overall_std}}
        from results import CVResults  # Assuming you have a CVResults class defined elsewhere.
        cv_results = CVResults(results)
        return cv_results

# ---------------------------------------------------------------------------
# Sklearn CV Trainer
# ---------------------------------------------------------------------------

class SklearnCVTrainer(BaseCVTrainer):
    def __init__(self, dataset, evaluation_fn: Callable, model_builder: Callable, **kwargs):
        """
        model_builder: A callable that returns a fresh scikit-learn model instance.
        Additional kwargs are passed to BaseCVTrainer.
        """
        super().__init__(dataset, evaluation_fn, **kwargs)
        self.model_builder = model_builder

    def train_model(self, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_train = self.X_all[train_idx]
        y_train = self.y_all[train_idx]
        X_test = self.X_all[test_idx]
        model = self.model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.y_all[test_idx], y_pred

# ---------------------------------------------------------------------------
# PyTorch CV Trainer
# ---------------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

class PyTorchCVTrainer(BaseCVTrainer):
    def __init__(self, dataset, evaluation_fn: Callable, model_builder: Callable, train_params: dict, **kwargs):
        """
        model_builder: Callable returning a fresh PyTorch model.
        train_params: Dictionary with training parameters (epochs, batch_size, device, criterion, etc.).
        Additional kwargs are passed to BaseCVTrainer.
        
        Optionally, train_params can include:
            "use_validation": bool (if True, split outer training set into train and validation).
            "val_size": float (fraction of outer training set to use as validation).
            "val_stratify": bool (if True, stratify the inner train/val split using stratify_col).
        """
        super().__init__(dataset, evaluation_fn, **kwargs)
        self.model_builder = model_builder
        self.train_params = train_params
        self.device = train_params.get("device", "cpu")
        self.batch_size = train_params.get("batch_size", 32)
        self.use_validation = train_params.get("use_validation", False)
        self.val_size = train_params.get("val_size", 0.2)

    def _create_loader(self, indices: np.ndarray, shuffle: bool = False):
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=shuffle)

    def train_model(self, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # If validation is enabled, further split the outer training set.
        if self.use_validation:
            row_metadata = self.dataset.get_row_metadata()
            stratify_values = None
            if self.train_params.get("val_stratify", False) and self.stratify_col is not None:
                stratify_values = row_metadata.iloc[train_idx][self.stratify_col].values
            inner_train_idx, inner_val_idx = train_test_split(
                train_idx,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=stratify_values
            )
            train_loader = self._create_loader(inner_train_idx, shuffle=True)
            val_loader = self._create_loader(inner_val_idx, shuffle=False)
            test_loader = self._create_loader(test_idx, shuffle=False)
            
            model = self.model_builder()
            model.to(self.device)
            optimizer = self.train_params.get("optimizer_fn")(
                model.parameters(), lr=self.train_params.get("learning_rate", 1e-3)
            )
            # train_model and evaluate_model are assumed implemented elsewhere.
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=self.train_params.get("criterion"),
                optimizer=optimizer,
                scheduler=self.train_params.get("scheduler", None),
                epochs=self.train_params.get("epochs", 50),
                device=self.device,
                gradient_clipping=self.train_params.get("gradient_clipping", 5.0),
                early_stopping_patience=self.train_params.get("early_stopping_patience", 10),
                model_name="PyTorchCVTrainer",
                use_mixed_precision=self.train_params.get("use_mixed_precision", False),
            )
        else:
            train_loader = self._create_loader(train_idx, shuffle=True)
            test_loader = self._create_loader(test_idx, shuffle=False)
            model = self.model_builder()
            model.to(self.device)
            optimizer = self.train_params.get("optimizer_fn")(
                model.parameters(), lr=self.train_params.get("learning_rate", 1e-3)
            )
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,  # No separate validation, using test_loader for monitoring.
                criterion=self.train_params.get("criterion"),
                optimizer=optimizer,
                scheduler=self.train_params.get("scheduler", None),
                epochs=self.train_params.get("epochs", 50),
                device=self.device,
                gradient_clipping=self.train_params.get("gradient_clipping", 5.0),
                early_stopping_patience=self.train_params.get("early_stopping_patience", 10),
                model_name="PyTorchCVTrainer",
                use_mixed_precision=self.train_params.get("use_mixed_precision", False),
            )
        # After training, evaluate on the outer test set.
        y_true, y_pred = evaluate_model(model, test_loader, device=self.device)
        return y_true.numpy(), y_pred.numpy()