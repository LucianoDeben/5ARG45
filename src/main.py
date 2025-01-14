# %% Import required libraries and modules
import logging
import os
import sys

import decoupler as dc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import FlexibleFCNN, SparseKnowledgeNetwork
from pipelines import train_and_evaluate_models
from preprocess import filter_and_prepare_datasets, split_data
from utils import create_dataloader, load_config, load_sampled_data

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load Config
config = load_config("../config.yaml")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


# %% Load and Preprocess Datasets
logging.info("Loading datasets...")

# Load datasets
datasets = {
    "TF Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_tf_file"], sample_size=1000
        ),
        "viability",
    ),
    "Landmark Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_landmark_file"], sample_size=1000
        ),
        "viability",
    ),
    "Best Inferred Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_best_inferred_file"], sample_size=1000
        ),
        "viability",
    ),
    "Gene Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_gene_file"],
            sample_size=1000,
            use_chunks=True,
            chunk_size=1000,
        ),
        "viability",
    ),
}

# %% Split datasets into train/val/test sets
split_datasets = {}
for name, (X, y_col) in datasets.items():
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, target_name=y_col, config=config, stratify_by="cell_mfc_name"
    )
    split_datasets[name] = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

# %% Load Collectri network
collectri_net = dc.get_collectri(organism="human", split_complexes=False)

# Filter datasets for all splits (train, val, test)
filtered_datasets = {
    name: {
        split: filter_and_prepare_datasets(collectri_net, {name: data[split]})[name]
        for split in ["train", "val", "test"]
    }
    for name, data in split_datasets.items()
}

# Create new dataloaders for all filtered datasets
dataloaders = {
    name: {split: create_dataloader(X, y) for split, (X, y) in splits.items()}
    for name, splits in filtered_datasets.items()
}

# %% Define feature sets
feature_sets = {
    name: (splits["train"], splits["val"], splits["test"])
    for name, splits in dataloaders.items()
}

# Define Model Configurations
model_configs = {
    "FlexibleFCNN": {
        "model_class": FlexibleFCNN,
        "model_params": {
            "input_dim": None,
            "hidden_dims": [512, 256, 128, 64],
            "output_dim": 1,
            "activation_fn": "relu",
            "dropout_prob": 0.2,
            "residual": True,
            "norm_type": "batchnorm",
            "weight_init": "xavier",
        },
        "criterion": nn.MSELoss(),
        "optimizer_class": optim.AdamW,
        "optimizer_params": {"lr": 0.001, "weight_decay": 1e-4},
        "scheduler_class": ReduceLROnPlateau,
        "scheduler_params": {"mode": "min", "patience": 5},
        "train_params": {"epochs": 20, "gradient_clipping": 1.0},
    },
    "SparseKnowledgeNetwork": {
        "model_class": SparseKnowledgeNetwork,
        "model_params": {
            "net": collectri_net,
            "hidden_dims": [512, 256, 128, 64],
            "output_dim": 1,
            "first_activation": "tanh",
            "downstream_activation": "relu",
            "dropout_prob": 0.2,
            "weight_init": "xavier",
            "use_batchnorm": True,
        },
        "criterion": nn.MSELoss(),
        "optimizer_class": optim.AdamW,
        "optimizer_params": {"lr": 0.001, "weight_decay": 1e-4},
        "scheduler_class": ReduceLROnPlateau,
        "scheduler_params": {"mode": "min", "patience": 5},
        "train_params": {"epochs": 20, "gradient_clipping": 1.0},
    },
}

# %% Train and Evaluate Models
results = train_and_evaluate_models(
    feature_sets=feature_sets,
    model_configs=model_configs,
    device=device,
    collectri_net=collectri_net,
)

# %% Save results
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.index = pd.MultiIndex.from_tuples(
    results_df.index, names=["Feature Set", "Model Name"]
)
results_df.to_csv("combined_metrics.csv", index=False)
logging.info("Results saved to combined_metrics.csv.")

# %% Display results
print(
    results_df.style.format(precision=3)
    .set_caption("Regression Model Evaluation Metrics")
    .highlight_max(subset=["R²", "Pearson Correlation"], color="lightgreen")
    .highlight_min(subset=["MAE", "MSE"], color="lightgreen")
)
