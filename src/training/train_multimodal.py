import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocess.preprocess import partition_data
from src.models.gnn import GNN
from src.models.multimodal_nn import MultimodalNN
from src.models.transcriptomics_nn import TranscriptomicsNN
from src.preprocess.data_loader import (
    prepare_chemical_data,
    prepare_transcriptomics_data,
)


def train_multimodal_model(
    chem_model,
    trans_model,
    multimodal_model,
    chem_data_loader,
    trans_data_loader,
    optimizer,
    criterion,
    epochs,
):
    chem_model.train()
    trans_model.train()
    multimodal_model.train()

    for epoch in range(epochs):
        print("Starting epoch", epoch + 1)
        total_loss = 0
        for chem_data, (trans_data, target) in zip(chem_data_loader, trans_data_loader):
            optimizer.zero_grad()

            chem_embedding = chem_model(chem_data)
            trans_embedding = trans_model(trans_data)

            output = multimodal_model(chem_embedding, trans_embedding)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(chem_data_loader)}")


# Example usage
if __name__ == "__main__":
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
    chem_model = GNN(
        num_node_features=15, num_edge_features=4, hidden_dim=64, output_dim=128
    )
    trans_model = TranscriptomicsNN(
        input_dim=transcriptomics_df.shape[1], hidden_dim=512, output_dim=128
    )
    multimodal_model = MultimodalNN(
        chem_output_dim=128, trans_output_dim=128, hidden_dim=256, output_dim=1
    )

    optimizer = torch.optim.Adam(
        list(chem_model.parameters())
        + list(trans_model.parameters())
        + list(multimodal_model.parameters()),
        lr=0.001,
    )
    criterion = torch.nn.MSELoss()
    print("Start training the MODELS!!!")

    # Train the model
    train_multimodal_model(
        chem_model,
        trans_model,
        multimodal_model,
        chem_data_loader,
        transcriptomics_data_loader,
        optimizer,
        criterion,
        epochs=10,
    )
