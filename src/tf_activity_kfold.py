import logging
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

# Import your custom functions.
from evaluation import evaluate_model
from models import FlexibleFCNN
from preprocess import run_tf_activity_inference, split_data_flexible, k_fold_cross_validation  
from training import train_model  
from utils import create_dataloader, create_dataset, load_config, load_sampled_data, sanity_check

# Add the src directory to the Python path (if needed)
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration.
config = load_config("../config.yaml")

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load original gene datasets.
SAMPLE_SIZE = 10000  # Or set a sample size if desired
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

# Load TF networks.
import decoupler as dc
collectri_net = dc.get_collectri(organism="human", split_complexes=False)
dorothea_net = dc.get_dorothea(organism="human")
networks = {"collectri": collectri_net, "dorothea": dorothea_net}

# Define interference algorithms and threshold values.
interference_algorithms = ["ulm"]
min_n_values = [1]

# Dictionary to store all results.
# Each key will be a composite key: e.g. "dorothea_min_n_5_ulm"
all_results = {}

# Helper function: Convert DataFrame/Series pair to a TensorDataset.
def create_dataset(X: pd.DataFrame, y: pd.Series) -> TensorDataset:
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)

# Main experiment loop.
for network_name, network_obj in networks.items():
    for algo in interference_algorithms:
        for min_n in min_n_values:
            composite_key = f"{network_name}_min_n_{min_n}_{algo}"
            logging.info(f"Processing configuration: {composite_key}")
            config_results = {}  # To store final evaluation metrics for each feature set.
            
            for dataset_name, (data, target) in original_datasets.items():
                # Run TF interference inference.
                try:
                    interference_output = run_tf_activity_inference(data, network_obj, min_n=min_n, algorithm=algo)
                except Exception as e:
                    logging.error(f"Error in run_tf_activity_inference for {dataset_name} with {composite_key}: {e}")
                    continue

                logging.info(f"{composite_key} - {dataset_name} interference output shape: {interference_output.shape}")

                # Split data (train/test) using your unified splitting function.
                X_train, y_train, X_test, y_test = split_data_flexible(
                    df=interference_output,
                    config=config,
                    target_name=target,
                    stratify_by="cell_mfc_name",
                    random_state=42,
                    return_val=False
                )

                # Standardize the splits.
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train.values)
                X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
                X_test_scaled = scaler.transform(X_test.values)
                X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

                # Create a test DataLoader.
                test_loader = create_dataloader(X_test_scaled_df, y_test, batch_size=1028, shuffle=False)

                # --- K-Fold Cross-Validation on the Training set ---
                train_dataset = create_dataset(X_train_scaled_df, y_train)
                fold_metrics = k_fold_cross_validation(
                    model_class=lambda: FlexibleFCNN(
                        input_dim=X_train_scaled_df.shape[1],
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
                    epochs=20,  # adjust epochs as needed
                    criterion_fn=nn.MSELoss(),
                    optimizer_fn=lambda params: optim.AdamW(params, lr=0.001, weight_decay=1e-5),
                    scheduler_fn=lambda opt: ReduceLROnPlateau(opt, mode="min", patience=5, verbose=True),
                    device=device,
                    use_mixed_precision=True,
                )
                # Compute average CV metrics.
                avg_metrics = {key: sum(fold_metrics[f][key] for f in fold_metrics) / len(fold_metrics)
                               for key in fold_metrics[0]}
                cv_results = {"fold_metrics": fold_metrics, "avg_metrics": avg_metrics}
                logging.info(f"{composite_key} - {dataset_name} CV Avg Metrics: {avg_metrics}")

                # --- Final Training on Full Training Set and Evaluation on Test Set ---
                train_loader = create_dataloader(X_train_scaled_df, y_train, batch_size=1028, shuffle=True)
                model_instance = FlexibleFCNN(
                    input_dim=X_train_scaled_df.shape[1],
                    hidden_dims=[512, 256, 128, 64],
                    output_dim=1,
                    activation_fn="relu",
                    dropout_prob=0.2,
                    residual=False,
                    norm_type="batchnorm",
                    weight_init="kaiming",
                ).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.AdamW(model_instance.parameters(), lr=0.001, weight_decay=1e-5)
                sanity_check(model_instance, train_loader, device)
                train_losses, val_losses = train_model(
                    model=model_instance,
                    train_loader=train_loader,
                    val_loader=test_loader,  # Using the test set for final evaluation.
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True),
                    epochs=20,
                    device=device,
                    gradient_clipping=5.0,
                    early_stopping_patience=10,
                    model_name=f"FlexibleFCNN_{dataset_name}_{composite_key}",
                    use_mixed_precision=True,
                )
                final_metrics = evaluate_model(model_instance, test_loader, criterion, device=device, calculate_metrics=True)
                logging.info(f"{composite_key} - {dataset_name} Final Metrics: {final_metrics}")
                
                # Store final evaluation metrics for this feature set.
                config_results[dataset_name] = final_metrics
            
            # Store the CV and final results for this configuration.
            all_results[composite_key] = {"CV": cv_results, "Final": config_results}

# --- Save Combined Results ---
logging.info(f"Training and evaluation completed. Combined Results: {all_results}")

# Flatten the nested dictionary into a DataFrame.
flat_results = {}
for key_outer, res in all_results.items():
    for dataset_key, metrics in res["Final"].items():
        composite_key = (dataset_key, key_outer)
        flat_results[composite_key] = metrics

results_df = pd.DataFrame.from_dict(flat_results, orient="index")
results_df.index = pd.MultiIndex.from_tuples(
    list(flat_results.keys()),
    names=["Dataset_and_Model", "Network_min_n_Algorithm"],
)

results_csv_path = "results_all_combinations.csv"
results_df.to_csv(results_csv_path, index=True)
logging.info(f"All results saved to {results_csv_path}")
