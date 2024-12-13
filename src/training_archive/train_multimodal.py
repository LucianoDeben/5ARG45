import sys
from pathlib import Path

import numpy as np
import torch

from evaluation import evaluate_regression_metrics

# It's better to set up your Python environment so you don't need to modify sys.path
# But if necessary, adjust the path accordingly
sys.path.append(str(Path(__file__).resolve().parents[2]))


def train_multimodal_model(
    chem_model,
    trans_model,
    multimodal_model,
    train_chem_loader,
    train_trans_loader,
    val_chem_loader,
    val_trans_loader,
    optimizer,
    criterion,
    device,
    epochs,
):
    """
    Train a multimodal deep learning model with validation loss tracking.

    Args:
        chem_model: Model for processing chemical data.
        trans_model: Model for processing transcriptomics data.
        multimodal_model: Model for combining embeddings and making predictions.
        train_chem_loader: DataLoader for training chemical data.
        train_trans_loader: DataLoader for training transcriptomics data.
        val_chem_loader: DataLoader for validation chemical data.
        val_trans_loader: DataLoader for validation transcriptomics data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Computation device (e.g., "cuda" or "cpu").
        epochs: Number of epochs to train.

    Returns:
        Tuple of (train_losses, val_losses): Losses per epoch for training and validation.
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}")
        total_train_loss = 0.0
        num_train_batches = 0

        # Training phase
        chem_model.train()
        trans_model.train()
        multimodal_model.train()

        for chem_data_batch, (trans_data_batch, target_batch) in zip(
            train_chem_loader, train_trans_loader
        ):
            optimizer.zero_grad()

            # Move data to the appropriate device
            chem_data_batch = chem_data_batch.to(device)
            trans_data_batch = trans_data_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass
            chem_embedding = chem_model(chem_data_batch)
            trans_embedding = trans_model(trans_data_batch)
            output = multimodal_model(chem_embedding, trans_embedding)

            # Compute loss
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        average_train_loss = total_train_loss / num_train_batches
        train_losses.append(average_train_loss)

        # Validation phase
        chem_model.eval()
        trans_model.eval()
        multimodal_model.eval()

        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for chem_data_batch, (trans_data_batch, target_batch) in zip(
                val_chem_loader, val_trans_loader
            ):
                # Move data to the appropriate device
                chem_data_batch = chem_data_batch.to(device)
                trans_data_batch = trans_data_batch.to(device)
                target_batch = target_batch.to(device)

                # Forward pass
                chem_embedding = chem_model(chem_data_batch)
                trans_embedding = trans_model(trans_data_batch)
                output = multimodal_model(chem_embedding, trans_embedding)

                # Compute loss
                loss = criterion(output, target_batch)

                total_val_loss += loss.item()
                num_val_batches += 1

        average_val_loss = total_val_loss / num_val_batches
        val_losses.append(average_val_loss)

        print(
            f"Epoch {epoch+1}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}"
        )

    return train_losses, val_losses


def evaluate_multimodal_model(
    chem_model,
    trans_model,
    multimodal_model,
    chem_data_loader,
    trans_data_loader,
    criterion,
    device,
):
    """
    Evaluate a multimodal model on the test set.

    Args:
        chem_model: Model for processing chemical data.
        trans_model: Model for processing transcriptomics data.
        multimodal_model: Model for combining embeddings and making predictions.
        chem_data_loader: DataLoader for chemical data.
        trans_data_loader: DataLoader for transcriptomics data.
        criterion: Loss function.
        device: Computation device (e.g., "cuda" or "cpu").

    Returns:
        Tuple of (test_loss, y_true, y_pred, metrics): Test loss, true values, predictions, and metrics.
    """
    chem_model.eval()
    trans_model.eval()
    multimodal_model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for chem_data_batch, (trans_data_batch, target_batch) in zip(
            chem_data_loader, trans_data_loader
        ):
            # Move data to the appropriate device
            chem_data_batch = chem_data_batch.to(device)
            trans_data_batch = trans_data_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass
            chem_embedding = chem_model(chem_data_batch)
            trans_embedding = trans_model(trans_data_batch)
            output = multimodal_model(chem_embedding, trans_embedding)

            # Compute loss
            loss = criterion(output, target_batch)
            total_loss += loss.item()

            y_true.extend(target_batch.cpu().numpy())
            y_pred.extend(output.cpu().numpy())

    test_loss = total_loss / len(chem_data_loader)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse, mae, r2, pearson_coef = evaluate_regression_metrics(
        y_true.flatten(), y_pred.flatten()
    )

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "R²": r2,
        "Pearson Correlation": pearson_coef,
    }

    return test_loss, y_true, y_pred, metrics


# if __name__ == "__main__":
#     # Set the device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load data
#     final_df = pd.read_csv("data/processed/final_dataset.csv")
#     chem_df, viability_df, transcriptomics_df = partition_data(final_df)

#     # Prepare chemical data
#     smiles_list = chem_df["canonical_smiles"].tolist()
#     targets = viability_df["viability"].tolist()
#     chem_data_loader = prepare_chemical_data(smiles_list, targets, batch_size=32)

#     # Prepare transcriptomics data
#     transcriptomics_data_loader = prepare_transcriptomics_data(
#         transcriptomics_df, targets, batch_size=32
#     )

#     # Initialize models
#     num_node_features = 15
#     num_edge_features = 10

#     chem_model = GNN(
#         num_node_features=num_node_features,
#         num_edge_features=num_edge_features,
#         hidden_dim=64,
#         output_dim=128,
#         dropout=0.1,  # If applicable
#     ).to(device)

#     trans_model = TranscriptomicsNN(
#         input_dim=transcriptomics_df.shape[1],
#         hidden_dim=512,
#         output_dim=128,
#         dropout=0.1,  # If applicable
#     ).to(device)

#     multimodal_model = MultimodalNN(
#         chem_output_dim=128,
#         trans_output_dim=128,
#         hidden_dim=256,
#         output_dim=1,
#         dropout=0.1,  # If applicable
#     ).to(device)

#     # Initialize optimizer and loss function
#     optimizer = torch.optim.Adam(
#         list(chem_model.parameters())
#         + list(trans_model.parameters())
#         + list(multimodal_model.parameters()),
#         lr=0.001,
#         weight_decay=1e-5,
#     )
#     criterion = torch.nn.MSELoss()

#     print("Start training the models!")

#     # Train the model
#     train_multimodal_model(
#         chem_model,
#         trans_model,
#         multimodal_model,
#         chem_data_loader,
#         transcriptomics_data_loader,
#         optimizer,
#         criterion,
#         device,
#         epochs=10,
#     )
