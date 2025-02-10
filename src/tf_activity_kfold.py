# %% Import required libraries and modules
import logging
import os
import sys
import warnings

from evaluation import evaluate_model
from models import FlexibleFCNN

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

import decoupler as dc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

use_fixed_validation = False

# Note: now using the new split function that returns either train/val/test or train/test splits,
# along with our k_fold_cross_validation routine.
from preprocess import run_tf_activity_inference, split_data_flexible, k_fold_cross_validation  
from training import train_model  
from utils import create_dataloader, load_config, load_sampled_data, sanity_check

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

# %% Load original datasets (before TF activity inference)
logging.info("Loading original datasets...")

SAMPLE_SIZE = None  # Or set a sample size if desired

# Load Collectri network
collectri_net = dc.get_collectri(organism="human", split_complexes=False)

# Load datasets into a dictionary.
# These are the raw, preprocessed datasets before applying TF activity inference.
original_datasets = {
    "Landmark Data": (
        load_sampled_data(config["data_paths"]["preprocessed_landmark_file"], sample_size=SAMPLE_SIZE),
        "viability",
    ),
    "Best Inferred Data": (
        load_sampled_data(config["data_paths"]["preprocessed_best_inferred_file"], sample_size=SAMPLE_SIZE),
        "viability",
    ),
    "Gene Data": (
        load_sampled_data(
            config["data_paths"]["preprocessed_gene_file"],
            sample_size=SAMPLE_SIZE,
            use_chunks=True,
            chunk_size=2000,
        ),
        "viability",
    ),
}

# Define the list of min_n values to test and algorithms to benchmark.
min_n_values = [1, 5, 10, 20, 50]
algorithms = [
    "ulm",
    # "viper",
    # "aucell",
    # "mlm",
]

# Dictionary to store all results
all_results = {}

# %% Main processing loop
for min_n in min_n_values:
    for algorithm in algorithms:
        logging.info(f"Processing TF activity inference with min_n = {min_n} and algorithm = {algorithm}")

        # Apply TF activity inference for each dataset using the current min_n and algorithm.
        datasets = {
            name: (
                run_tf_activity_inference(data, collectri_net, min_n=min_n, algorithm=algorithm),
                target,
            )
            for name, (data, target) in original_datasets.items()
        }

        # Log the shapes of the data after TF activity inference.
        for name, (X, y_col) in datasets.items():
            logging.info(f"min_n={min_n}, algo={algorithm} - {name} shape: {X.shape}")

        # %% Split each dataset using split_data_flexible
        split_datasets = {}
        for name, (X, y_col) in datasets.items():
            if use_fixed_validation:
                X_train, y_train, X_val, y_val, X_test, y_test = split_data_flexible(
                    df=X,
                    config=config,
                    target_name=y_col,
                    stratify_by="cell_mfc_name",
                    random_state=42,
                    return_val=True
                )
                split_datasets[name] = {
                    "train": (X_train, y_train),
                    "val": (X_val, y_val),
                    "test": (X_test, y_test),
                }
            else:
                # Train/test split for k-fold CV:
                X_train, y_train, X_test, y_test = split_data_flexible(
                    df=X,
                    config=config,
                    target_name=y_col,
                    stratify_by="cell_mfc_name",
                    random_state=42,
                    return_val=False
                )
                split_datasets[name] = {
                    "train": (X_train, y_train),
                    "test": (X_test, y_test),
                }

        # %% Standardize the splits using StandardScaler
        from sklearn.preprocessing import StandardScaler

        for name, splits in split_datasets.items():
            scaler = StandardScaler()
            if use_fixed_validation:
                X_train, y_train = splits["train"]
                X_val, y_val = splits["val"]
                X_test, y_test = splits["test"]

                X_train_scaled = scaler.fit_transform(X_train.values)
                X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
                splits["train"] = (X_train_scaled_df, y_train)

                X_val_scaled = scaler.transform(X_val.values)
                X_val_scaled_df = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
                splits["val"] = (X_val_scaled_df, y_val)

                X_test_scaled = scaler.transform(X_test.values)
                X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
                splits["test"] = (X_test_scaled_df, y_test)
            else:
                X_train, y_train = splits["train"]
                X_test, y_test = splits["test"]

                X_train_scaled = scaler.fit_transform(X_train.values)
                X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
                splits["train"] = (X_train_scaled_df, y_train)

                X_test_scaled = scaler.transform(X_test.values)
                X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
                splits["test"] = (X_test_scaled_df, y_test)

        # %% Create dataloaders for the Test set (and Validation if fixed split)
        dataloaders = {}
        for name, splits in split_datasets.items():
            if use_fixed_validation:
                # Build loaders for both validation and test sets.
                _, (X_val, y_val) = None, splits["val"]
                _, (X_test, y_test) = None, splits["test"]
                dataloaders[name] = {
                    "val": create_dataloader(X_val, y_val, batch_size=1028, shuffle=False),
                    "test": create_dataloader(X_test, y_test, batch_size=1028, shuffle=False)
                }
            else:
                # Only test loader is built (for final evaluation).
                _, (X_test, y_test) = None, splits["test"]
                dataloaders[name] = {"test": create_dataloader(X_test, y_test, batch_size=1028, shuffle=False)}

        # %% K-Fold Cross-Validation on the Training set (if not using fixed validation)
        kfold_results = {}
        if not use_fixed_validation:
            for name, splits in split_datasets.items():
                logging.info(f"Performing 5-fold CV on training data for dataset: {name}")
                X_train, y_train = splits["train"]
                # Convert training data to a PyTorch Dataset.
                train_dataset = create_dataset(X_train, y_train)

                fold_metrics = k_fold_cross_validation(
                    model_class=lambda: FlexibleFCNN(
                        input_dim=X_train.shape[1],
                        hidden_dims=[512, 256, 128, 64],
                        output_dim=1,
                        activation_fn="relu",
                        dropout_prob=0.2,
                        residual=False,
                        norm_type="batchnorm",
                        weight_init="kaiming",
                    ),
                    train_dataset=train_dataset,
                    k_folds=5,
                    batch_size=1028,
                    epochs=10,  # adjust epochs as needed
                    criterion_fn=nn.MSELoss(),
                    optimizer_fn=lambda params: optim.AdamW(params, lr=0.001, weight_decay=1e-5),
                    scheduler_fn=lambda opt: ReduceLROnPlateau(opt, mode="min", patience=5, verbose=True),
                    device=device,
                    use_mixed_precision=True,
                )
                avg_metrics = {key: sum(fold_metrics[f][key] for f in fold_metrics) / len(fold_metrics) 
                               for key in fold_metrics[0]}
                kfold_results[name] = {"fold_metrics": fold_metrics, "avg_metrics": avg_metrics}
                logging.info(f"Dataset: {name} - Average K-Fold CV Metrics: {avg_metrics}")

        # %% Final Training on Full Training Set and Evaluation on Test Set
        final_results = {}
        for name, splits in split_datasets.items():
            logging.info(f"Final training on full training set for dataset: {name}")
            if use_fixed_validation:
                # If using a fixed split, use the train loader and the fixed validation set.
                X_train, y_train = splits["train"]
                _, (X_val, y_val) = None, splits["val"]
                _, (X_test, y_test) = None, splits["test"]
                train_loader = create_dataloader(X_train, y_train, batch_size=1028, shuffle=True)
                # Here you might want to use the validation loader during training.
                val_loader = create_dataloader(X_val, y_val, batch_size=1028, shuffle=False)
                test_loader = dataloaders[name]["test"]
            else:
                X_train, y_train = splits["train"]
                _, (X_test, y_test) = None, splits["test"]
                train_loader = create_dataloader(X_train, y_train, batch_size=1028, shuffle=True)
                test_loader = dataloaders[name]["test"]

            # Create a new model instance.
            fcnn_model = FlexibleFCNN(
                input_dim=X_train.shape[1],
                hidden_dims=[512, 256, 128, 64],
                output_dim=1,
                activation_fn="relu",
                dropout_prob=0.2,
                residual=False,
                norm_type="batchnorm",
                weight_init="kaiming",
            ).to(device)

            fcnn_criterion = nn.MSELoss()
            fcnn_optimizer = optim.AdamW(fcnn_model.parameters(), lr=0.001, weight_decay=1e-5)

            # Quick sanity check on the model.
            sanity_check(fcnn_model, train_loader, device)

            # Train the model.
            fcnn_train_losses, fcnn_val_losses = train_model(
                model=fcnn_model,
                train_loader=train_loader,
                val_loader=test_loader,  # Here using the test set for final evaluation
                criterion=fcnn_criterion,
                optimizer=fcnn_optimizer,
                scheduler=ReduceLROnPlateau(fcnn_optimizer, mode="min", patience=5, verbose=True),
                epochs=20,
                device=device,
                gradient_clipping=5.0,
                early_stopping_patience=10,
                model_name=f"FlexibleFCNN_{name}_min_n_{min_n}_algo_{algorithm}",
                use_mixed_precision=True,
            )

            # Evaluate the final model on the fixed test set.
            fcnn_metrics = evaluate_model(
                fcnn_model, test_loader, fcnn_criterion, device=device, calculate_metrics=True
            )
            final_results[name] = fcnn_metrics
            logging.info(f"Final Evaluation Metrics for dataset {name}: {fcnn_metrics}")

        # Store the results for the current min_n and algorithm.
        all_results[f"min_n_{min_n}_algo_{algorithm}"] = {
            "CV": kfold_results if not use_fixed_validation else None,
            "Final": final_results,
        }

# Optionally, flatten the nested dictionary into a DataFrame for easier analysis.
flat_results = {}
for key_outer, res in all_results.items():
    for dataset_key, metrics in res["Final"].items():
        composite_key = (dataset_key, key_outer)
        flat_results[composite_key] = metrics

results_df = pd.DataFrame.from_dict(flat_results, orient="index")
results_df.index = pd.MultiIndex.from_tuples(
    list(flat_results.keys()),
    names=["Dataset_and_Model", "min_n_and_algo"],
)

results_df.to_csv("results_all_min_n_and_algos.csv", index=True)
logging.info("All results saved to results_all_min_n_and_algos.csv.")
