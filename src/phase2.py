# %% Import required libraries and modules
import logging
import os
import sys
import warnings

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

import decoupler as dc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import evaluate_model
from models import FlexibleFCNN, SparseKnowledgeNetwork
from preprocess import create_gene_tf_matrix, filter_dataset_and_network, split_data
from training import train_model
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

SAMPLE_SIZE = None

# Load datasets
datasets = {
    "Landmark Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_landmark_file"], sample_size=SAMPLE_SIZE
        ),
        "viability",
    ),
    "Best Inferred Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_best_inferred_file"],
            sample_size=SAMPLE_SIZE,
        ),
        "viability",
    ),
    "Gene Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_gene_file"],
            sample_size=SAMPLE_SIZE,
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


# %% Filter datasets and networks for all splits (train, val, test)
filtered_split_datasets = {}
filtered_networks = {}

for name, splits in split_datasets.items():
    filtered_splits = {}
    logging.info(f"Filtering splits for {name} dataset...")
    for split_name, (X, y) in splits.items():
        filtered_X, filtered_net = filter_dataset_and_network(X, collectri_net)
        filtered_splits[split_name] = (filtered_X, y)
        # Store the network once (same across splits as genes remain consistent)
        if name not in filtered_networks:
            filtered_networks[name] = filtered_net
    filtered_split_datasets[name] = filtered_splits

# %% Create dataloaders
dataloaders = {
    name: {
        split: create_dataloader(X, y, batch_size=32)
        for split, (X, y) in splits.items()
    }
    for name, splits in filtered_split_datasets.items()
}

# %% Train FlexibleFCNN and SparseKnowledgeNetwork sequentially
results = {}

for name, loaders in dataloaders.items():
    train_loader, val_loader, test_loader = (
        loaders["train"],
        loaders["val"],
        loaders["test"],
    )

    # Train FlexibleFCNN
    logging.info(f"Training FlexibleFCNN on {name} feature set...")
    fcnn_model = FlexibleFCNN(
        input_dim=next(iter(train_loader))[0].shape[1],
        hidden_dims=[512, 256, 128, 64],
        output_dim=1,
        activation_fn="relu",
        dropout_prob=0.2,
        residual=True,
        norm_type="batchnorm",
        weight_init="xavier",
    ).to(device)
    num_params = sum(p.numel() for p in fcnn_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters FCNN model: {num_params}")

    fcnn_criterion = nn.MSELoss()
    fcnn_optimizer = optim.AdamW(fcnn_model.parameters(), lr=0.001, weight_decay=1e-4)

    fcnn_train_losses, fcnn_val_losses = train_model(
        model=fcnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=fcnn_criterion,
        optimizer=fcnn_optimizer,
        scheduler=ReduceLROnPlateau(
            fcnn_optimizer, mode="min", patience=5, verbose=True
        ),
        epochs=20,
        device=device,
        gradient_clipping=1.0,
        early_stopping_patience=5,
        model_name=f"FlexibleFCNN_{name}",
    )

    fcnn_metrics = evaluate_model(
        fcnn_model, test_loader, fcnn_criterion, device=device, calculate_metrics=True
    )
    results[f"{name}_FlexibleFCNN"] = fcnn_metrics

    # Train SparseKnowledgeNetwork
    logging.info(f"Training SparseKnowledgeNetwork on {name} feature set...")
    sparse_gene_tf_matrix = create_gene_tf_matrix(
        filtered_networks[name],
        filtered_split_datasets[name]["train"][0].columns.tolist(),
    )

    sparse_model = SparseKnowledgeNetwork(
        gene_tf_matrix=sparse_gene_tf_matrix.to(device),
        hidden_dims=[512, 256, 128, 64],
        output_dim=1,
        first_activation="tanh",
        downstream_activation="relu",
        dropout_prob=0.2,
        weight_init="xavier",
        use_batchnorm=True,
    ).to(device)

    num_params = sum(p.numel() for p in sparse_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters Constrained model: {num_params}")

    sparse_criterion = nn.MSELoss()
    sparse_optimizer = optim.AdamW(
        sparse_model.parameters(), lr=0.001, weight_decay=1e-4
    )

    sparse_train_losses, sparse_val_losses = train_model(
        model=sparse_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=sparse_criterion,
        optimizer=sparse_optimizer,
        scheduler=ReduceLROnPlateau(
            sparse_optimizer, mode="min", patience=5, verbose=True
        ),
        epochs=20,
        device=device,
        gradient_clipping=1.0,
        early_stopping_patience=5,
        model_name=f"SparseKnowledgeNetwork_{name}",
    )

    sparse_metrics = evaluate_model(
        sparse_model,
        test_loader,
        sparse_criterion,
        device=device,
        calculate_metrics=True,
    )
    results[f"{name}_SparseKnowledgeNetwork"] = sparse_metrics

# Log and save results
logging.info(f"Training and evaluation completed. Results: {results}")
results_df = pd.DataFrame.from_dict(results, orient="index")
# Split the keys into MultiIndex
results_df.index = pd.MultiIndex.from_tuples(
    [tuple(key.split("_")) for key in results_df.index], names=["Feature Set", "Model"]
)
# Save results to CSV
results_df.to_csv("results.csv", index=True)
logging.info("Results saved to results.csv.")
