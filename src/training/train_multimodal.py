import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader

# It's better to set up your Python environment so you don't need to modify sys.path
# But if necessary, adjust the path accordingly
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.gnn import GNN
from src.models.multimodal_nn import MultimodalNN
from src.models.transcriptomics_nn import TranscriptomicsNN
from src.preprocess.data_loader import (
    prepare_chemical_data,
    prepare_transcriptomics_data,
)
from src.preprocess.preprocess import partition_data


def train_multimodal_model(
    chem_model,
    trans_model,
    multimodal_model,
    chem_data_loader,
    trans_data_loader,
    optimizer,
    criterion,
    device,
    epochs,
):
    chem_model.train()
    trans_model.train()
    multimodal_model.train()

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}")
        total_loss = 0.0
        num_batches = 0

        # Ensure that both data loaders provide data in sync
        for chem_data_batch, (trans_data_batch, target_batch) in zip(
            chem_data_loader, trans_data_loader
        ):
            optimizer.zero_grad()

            # Move data to the appropriate device
            chem_data_batch = chem_data_batch.to(device)
            trans_data_batch = trans_data_batch.to(device)
            target_batch = target_batch.to(device)

            chem_embedding = chem_model(chem_data_batch)
            trans_embedding = trans_model(trans_data_batch)

            output = multimodal_model(chem_embedding, trans_embedding)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    final_df = pd.read_csv("data/processed/final_dataset.csv")
    chem_df, viability_df, transcriptomics_df = partition_data(final_df)

    # Prepare chemical data
    smiles_list = chem_df["canonical_smiles"].tolist()
    targets = viability_df["viability"].tolist()
    chem_data_loader = prepare_chemical_data(smiles_list, targets, batch_size=32)

    # Prepare transcriptomics data
    transcriptomics_data_loader = prepare_transcriptomics_data(
        transcriptomics_df, targets, batch_size=32
    )

    # Initialize models
    num_node_features = 15
    num_edge_features = 10

    chem_model = GNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=64,
        output_dim=128,
        dropout=0.1,  # If applicable
    ).to(device)

    trans_model = TranscriptomicsNN(
        input_dim=transcriptomics_df.shape[1],
        hidden_dim=512,
        output_dim=128,
        dropout=0.1,  # If applicable
    ).to(device)

    multimodal_model = MultimodalNN(
        chem_output_dim=128,
        trans_output_dim=128,
        hidden_dim=256,
        output_dim=1,
        dropout=0.1,  # If applicable
    ).to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        list(chem_model.parameters())
        + list(trans_model.parameters())
        + list(multimodal_model.parameters()),
        lr=0.001,
        weight_decay=1e-5,
    )
    criterion = torch.nn.MSELoss()

    print("Start training the models!")

    # Train the model
    train_multimodal_model(
        chem_model,
        trans_model,
        multimodal_model,
        chem_data_loader,
        transcriptomics_data_loader,
        optimizer,
        criterion,
        device,
        epochs=10,
    )
