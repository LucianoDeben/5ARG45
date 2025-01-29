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
from models import FlexibleFCNN
from preprocess import split_data
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

# %% Load and Preprocess Datasets with TF Activity Inference
logging.info("Loading datasets and running TF activity inference...")


def run_tf_activity_inference(X, net, min_n=1):
    """
    Run TF activity inference on the input data.

    Args:
        X (pd.DataFrame): Gene expression matrix, including metadata columns.
        net (pd.DataFrame): Regulatory network for TF activity inference.
        min_n (int): Minimum number of targets for each TF.

    Returns:
        pd.DataFrame: TF activity matrix with metadata reattached.
    """
    import logging

    import pandas as pd
    import scanpy as sc

    # Separate metadata columns from gene expression
    metadata_cols = [
        "cell_mfc_name",
        "viability",
        "pert_dose",
    ]
    metadata = X[metadata_cols]
    gene_expression = X.drop(columns=metadata_cols)

    # Filter the network for shared genes
    shared_genes = net["target"].unique()
    shared_genes = [gene for gene in shared_genes if gene in gene_expression.columns]
    logging.debug(f"Number of shared genes: {len(shared_genes)}")
    assert shared_genes, "No shared genes between network and gene expression matrix!"

    # Filter network and gene expression
    net_filtered = net[net["target"].isin(shared_genes)]
    logging.debug(f"Filtered network has {len(net_filtered)} interactions.")
    gene_expression = gene_expression[shared_genes]

    # Create AnnData object
    adata = sc.AnnData(
        X=gene_expression.values,
        obs=pd.DataFrame(index=gene_expression.index),
        var=pd.DataFrame(index=gene_expression.columns),
    )
    logging.info(f"AnnData object created with shape: {adata.shape}")

    # Run ULM for TF activity inference
    dc.run_ulm(
        mat=adata,
        net=net_filtered,
        source="source",
        target="target",
        weight="weight",
        min_n=min_n,
        use_raw=False,
    )

    tf_activity = pd.DataFrame(adata.obsm["ulm_estimate"], index=adata.obs.index)

    # Convert the index of tf_activity to integers to match metadata
    tf_activity.index = tf_activity.index.astype(int)

    # Reattach metadata columns
    tf_activity = tf_activity.join(metadata)

    return tf_activity


# Load Collectri network
collectri_net = dc.get_collectri(organism="human", split_complexes=False)

# Iterate through datasets, apply TF activity inference
datasets = {
    name: (
        run_tf_activity_inference(
            load_sampled_data(file_path, sample_size=30000),
            collectri_net,
            min_n=config.get("min_n", 1),
        ),
        "viability",
    )
    for name, file_path in config["data_paths"].items()
}

# %% Split datasets into train/val/test sets
split_datasets = {}
for name, (X, y_col) in datasets.items():
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X,
        target_name=y_col,
        config=config,
        stratify_by="cell_mfc_name",  # Ensure `cell_mfc_name` is present
    )
    split_datasets[name] = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


# %% Create dataloaders
dataloaders = {
    name: {
        split: create_dataloader(X, y, batch_size=32)
        for split, (X, y) in splits.items()
    }
    for name, splits in split_datasets.items()
}

# %% Train FlexibleFCNN sequentially
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

# Log and save results
logging.info(f"Training and evaluation completed. Results: {results}")
results_df = pd.DataFrame.from_dict(results, orient="index")

# Split the keys into MultiIndex
results_df.index = pd.MultiIndex.from_tuples(
    [
        (key.split("_")[0], "FlexibleFCNN") for key in results_df.index
    ],  # Explicitly add model name
    names=["Feature Set", "Model"],  # Names for the MultiIndex levels
)

# Save results to CSV
results_df.to_csv("results.csv", index=True)
logging.info("Results saved to results.csv.")
