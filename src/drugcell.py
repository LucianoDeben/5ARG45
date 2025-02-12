# %% Import required libraries and modules
import logging
import os
import sys
import warnings

import wandb

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import evaluate_model
from models import genecell_nn  # Our gene-only ontology model
from preprocess import split_data
from training import train_model  # Our training function (to be modified)
from utils import (
    create_dataloader,
    filter_dataset_columns,
    load_config,
    load_mapping,
    load_ontology,
    load_sampled_data,
)

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

datasets = {
    "Landmark Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_landmark_file"], sample_size=SAMPLE_SIZE
        ),
        "viability",
    ),
        # "Best Inferred Data": (
        #     load_sampled_data(
        #         config["data_paths"]["preprocessed_best_inferred_file"],
        #         sample_size=SAMPLE_SIZE,
        #     ),
        #     "viability",
        # ),
        # "Gene Data": (
        #     load_sampled_data(
        #         config["data_paths"]["preprocessed_gene_file"],
        #         sample_size=SAMPLE_SIZE,
        #         use_chunks=True,
        #         chunk_size=1000,
        #     ),
        #     "viability",
        # ),
}

landmark_mapping = load_mapping("../data/raw/landmark_mapping.txt")
best_inferred_mapping = load_mapping("../data/raw/best_inferred_mapping.txt")
gene_mapping = load_mapping("../data/raw/gene_mapping.txt")

# Dictionary to map dataset names to their corresponding mapping.
mapping_dict = {
    "Landmark Data": landmark_mapping,
    "Best Inferred Data": best_inferred_mapping,
    "Gene Data": gene_mapping,
}
filtered_datasets = {}
for name, (df, target_col) in datasets.items():
    mapping = mapping_dict[name]
    filtered_df = filter_dataset_columns(df, mapping)
    filtered_datasets[name] = (filtered_df, target_col)

filtered_split_datasets = {}
for name, (df, target_col) in filtered_datasets.items():
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df, target_name=target_col, config=config, stratify_by="cell_mfc_name"
    )
    filtered_split_datasets[name] = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

dataloaders = {
    name: {
        split: create_dataloader(X, y, batch_size=2056)
        for split, (X, y) in splits.items()
    }
    for name, splits in filtered_split_datasets.items()
}

results = {}


class GenecellWrapper(genecell_nn):
    """
    A wrapper around genecell_nn that returns only the final prediction.
    """
    def forward(self, x):
        aux_out_map, term_out_map = super().forward(x)
        return aux_out_map["final"]



results = {}

for name, loaders in dataloaders.items():
    # Initialize Weights & Biases for each experiment
    wandb.init(
        project="5ARG45",
        name=f"genecell_{name}",
        mode="online",  # Change to "online" if you want to sync immediately
    )
    wandb.config = {
        "lr": 0.001,
        "architecture": "GeneCell",
        "dataset": "LINCS/CTRPv2",  # Update this as appropriate
        "epochs": 20,
        "batch_size": 32,
        "num_hiddens_genotype": 6,
        "num_hiddens_final": 6,
    }
    
    logging.info(f"Training on {name} feature set...")
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    logging.info(f"Training GeneOntologyModel on {name} feature set...")
    # Use the mapping corresponding to this feature set to load the ontology.
    mapping = mapping_dict[name]
    dG, root, term_size_map, term_direct_gene_map = load_ontology("../data/raw/drugcell_ont.txt", mapping)

    # Determine gene_dim from training data (assumes X_train is a DataFrame or tensor with features as columns)
    gene_dim = filtered_split_datasets[name]["train"][0].shape[1]
    num_hiddens_genotype = 6  # Example hyperparameter
    num_hiddens_final = 6     # Example hyperparameter

    # Instantiate the wrapped model that returns only the final output.
    gene_ontology_model = GenecellWrapper(
        term_size_map,
        term_direct_gene_map,
        dG,
        gene_dim,
        root,
        num_hiddens_genotype,
        num_hiddens_final,
    ).to(device)

    num_params = sum(
        p.numel() for p in gene_ontology_model.parameters() if p.requires_grad
    )
    print(f"Number of trainable parameters: {num_params}")

    ontology_criterion = nn.MSELoss()
    ontology_optimizer = optim.AdamW(gene_ontology_model.parameters(), lr=0.001, weight_decay=1e-4)
    ontology_scheduler = ReduceLROnPlateau(ontology_optimizer, mode="min", patience=5, verbose=True)

    # Train the model using your train_model function (which remains unchanged).
    ontology_train_losses, ontology_val_losses = train_model(
        model=gene_ontology_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=ontology_criterion,
        optimizer=ontology_optimizer,
        scheduler=ontology_scheduler,
        epochs=20,
        device=device,
        gradient_clipping=1.0,
        early_stopping_patience=5,
        model_name=f"GeneOntologyModel_{name}",
    )

    # Optionally log losses to wandb.
    wandb.log({
        "train_loss": ontology_train_losses[-1],
        "val_loss": ontology_val_losses[-1],
    })

    ontology_metrics = evaluate_model(
        gene_ontology_model,
        test_loader,
        ontology_criterion,
        device=device,
        calculate_metrics=True,
    )
    results[f"{name}_GeneOntologyModel"] = ontology_metrics

    # Finish this wandb run.
    wandb.finish()

logging.info(f"Training and evaluation completed. Results: {results}")
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.index = pd.MultiIndex.from_tuples(
    [tuple(key.split("_")) for key in results_df.index],
    names=["Feature Set", "Model"],
)
results_df.to_csv("results_drugcell.csv", index=True)
logging.info("Results saved to results.csv.")
