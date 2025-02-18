import logging
import numpy as np
import pandas as pd
from data_sets import LINCSDataset
from utils import nested_stratified_group_split_fixed
from metrics import get_regression_metrics
from cv_trainers import SklearnNestedCVTrainer, PyTorchNestedCVTrainer
from results import CVResults
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import torch
import torch.nn as nn
import torch.optim as optim
from models import FlexibleFCNN

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Example for Scikit-Learn models:
    feature_space_options = ["landmark"]
    sklearn_model_builders = {
        "LinearRegression": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=0.5),
    }
    
    base_dataset = LINCSDataset(
        gctx_path="../data/processed/LINCS.gctx",
        in_memory=True,
        normalize="z-score",
        feature_space="landmark"
    )
    nested_splits = nested_stratified_group_split_fixed(base_dataset, n_folds=5, random_state=42)
    
    results = {}
    for feature_space in feature_space_options:
        feat_label = feature_space if isinstance(feature_space, str) else ",".join(feature_space)
        for model_name, builder in sklearn_model_builders.items():
            combined_key = (feat_label, model_name)
            logging.info(f"Evaluating {combined_key}")
            dataset = LINCSDataset(
                gctx_path="../data/processed/LINCS.gctx",
                in_memory=True,
                normalize="z-score",
                feature_space=feature_space
            )
            trainer = SklearnNestedCVTrainer(
                dataset=dataset,
                nested_splits=nested_splits,
                evaluation_fn=get_regression_metrics,
                model_builder=builder,
                use_inner_cv=False
            )
            cv_results = trainer.run(result_key=combined_key)
            results[combined_key] = cv_results.get_results_df().loc[combined_key]
    
    # Example for PyTorch models:
    pytorch_model_builders = {
        "FlexibleFCNN_1": lambda ds: FlexibleFCNN(
            input_dim=ds.to_numpy()[0].shape[1],
            hidden_dims=[512, 256, 128, 64],
            output_dim=1,
            activation_fn="relu",
            dropout_prob=0.2,
            residual=False,
            norm_type="batchnorm",
            weight_init="kaiming"
        ),
        "FlexibleFCNN_2": lambda ds: FlexibleFCNN(
            input_dim=ds.to_numpy()[0].shape[1],
            hidden_dims=[256, 128, 64],
            output_dim=1,
            activation_fn="relu",
            dropout_prob=0.3,
            residual=False,
            norm_type="batchnorm",
            weight_init="kaiming"
        )
    }
    
    train_params = {
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gradient_clipping": 5.0,
        "early_stopping_patience": 10,
        "use_mixed_precision": False,
        "criterion": nn.MSELoss(),
        "optimizer_fn": optim.Adam
    }
    
    for feature_space in feature_space_options:
        feat_label = feature_space if isinstance(feature_space, str) else ",".join(feature_space)
        for model_name, builder in pytorch_model_builders.items():
            combined_key = (feat_label, model_name)
            logging.info(f"Evaluating {combined_key}")
            dataset = LINCSDataset(
                gctx_path="../data/processed/LINCS.gctx",
                in_memory=True,
                normalize="z-score",
                feature_space=feature_space
            )
            trainer = PyTorchNestedCVTrainer(
                dataset=dataset,
                nested_splits=nested_splits,
                evaluation_fn=get_regression_metrics,
                model_builder=lambda: builder(dataset),
                train_params=train_params,
                use_inner_cv=False
            )
            cv_results = trainer.run(result_key=combined_key)
            results[combined_key] = cv_results.get_results_df().loc[combined_key]
    
    final_df = pd.concat(results.values(), axis=0)
    print(final_df)
