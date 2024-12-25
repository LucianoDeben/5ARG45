import torch
import torch.nn as nn


class DrugFeatureEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.smiles_encoder = TransformerSMILESEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, smiles_tokens):
        """
        Args:
            smiles_tokens (torch.Tensor): Tokenized SMILES tensors (B, L).
        Returns:
            torch.Tensor: Encoded SMILES representation.
        """
        return self.smiles_encoder(smiles_tokens)


class Perturbinator(nn.Module):
    """
    A model that encodes (unperturbed gene expression) + (drug SMILES tokens)
    and predicts (perturbed gene expression).
    """

    def __init__(self, gene_dim, gene_hidden_dim=512, drug_hidden_dim=128):
        super(Perturbinator, self).__init__()

        # Gene expression encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, gene_hidden_dim),
            nn.ReLU(),
            nn.Linear(gene_hidden_dim, gene_hidden_dim),
            nn.ReLU(),
        )

        # Drug feature encoder (accepts tokenized SMILES)
        self.smiles_encoder = DrugFeatureEncoder(
            embed_dim=128,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(gene_hidden_dim + drug_hidden_dim, gene_hidden_dim),
            nn.ReLU(),
            nn.Linear(gene_hidden_dim, gene_dim),
        )

    def forward(self, gene_expression, smiles_tokens):
        """
        Forward pass to predict perturbed gene expression.

        Args:
            gene_expression (torch.Tensor): Baseline gene expression (B, gene_dim).
            smiles_tokens (torch.Tensor): Tokenized SMILES tensors (B, L).
        Returns:
            torch.Tensor: Predicted perturbed gene expression (B, gene_dim).
        """
        gene_emb = self.gene_encoder(gene_expression)
        smiles_emb = self.smiles_encoder(smiles_tokens)

        combined_emb = torch.cat([gene_emb, smiles_emb], dim=-1)
        return self.fusion(combined_emb)


class TransformerSMILESEncoder(nn.Module):
    """
    Transformer-based SMILES encoder that takes pre-tokenized SMILES tensors.
    """

    def __init__(
        self,
        embed_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        max_length=512,
        vocab_size=128,
    ):
        """
        Args:
            embed_dim (int): Size of token embeddings.
            hidden_dim (int): Size of the Transformer feedforward layer.
            num_layers (int): Number of Transformer encoder layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            max_length (int): Maximum sequence length for positional encoding.
            vocab_size (int): Size of the tokenizer vocabulary.
        """
        super(TransformerSMILESEncoder, self).__init__()

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_length, embed_dim)
        )  # Max positional encoding

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, smiles_tokens):
        """
        Args:
            smiles_tokens (torch.Tensor): Tokenized SMILES tensors (B, L).
        Returns:
            torch.Tensor: Encoded SMILES representation (B, embed_dim).
        """
        # Token embeddings
        embeddings = self.token_embedding(smiles_tokens)

        # Add positional encoding
        embeddings = embeddings + self.positional_encoding[:, : embeddings.size(1), :]

        # Pass through Transformer encoder
        transformer_output = self.transformer(
            embeddings.transpose(0, 1)
        )  # (L, B, embed_dim)

        # Pool to get a single vector per SMILES
        pooled_output = self.pooling(
            transformer_output.transpose(0, 1).permute(0, 2, 1)
        ).squeeze(-1)

        return pooled_output


if __name__ == "__main__":
    # Add src to path
    import logging

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    import os
    import sys

    import torch

    # Add the src directory to the Python path
    sys.path.append(os.path.abspath(os.path.join("..", "src")))
    import pandas as pd
    from deepchem.feat.smiles_tokenizer import SmilesTokenizer
    from torch.utils.data import DataLoader, random_split

    from data_sets import PerturbationDataset
    from evaluation import evaluate_multimodal_model
    from perturbinator import Perturbinator
    from training import train_multimodal_model
    from utils import create_smiles_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the datasets
    controls = "../data/raw/compound_control_dataset.h5"
    perturbations = "../data/raw/compound_pertubation_dataset.h5"

    # Load the SMILES data and create a dictionary mapping
    smiles_df = pd.read_csv("../data/raw/compoundinfo_beta.txt", sep="\t")
    smiles_dict = create_smiles_dict(smiles_df)

    # Load the SMILES tokenizer
    vocab_file = "../data/raw/vocab.txt"
    smiles_tokenizer = SmilesTokenizer(vocab_file=vocab_file)

    max_length = 128

    dataset = PerturbationDataset(
        controls_file=controls,
        perturbations_file=perturbations,
        smiles_dict=smiles_dict,
        plate_column="det_plate",
        normalize=True,
        n_rows=1000,
        pairing="random",
        landmark_only=True,
        tokenizer=smiles_tokenizer,
        max_smiles_length=max_length,
    )

    # Split the dataset into train, validation, and test sets
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    total_len = len(dataset)
    train_size = int(train_ratio * total_len)
    val_size = int(val_ratio * total_len)
    test_size = total_len - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Log the sizes of the datasets
    logging.debug(
        f"Train Dataset: {len(train_dataset)}, Val Dataset: {len(val_dataset)}, Test Dataset: {len(test_dataset)}"
    )

    batch_size = 256
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Initialize the Perturbinator
    model = Perturbinator(
        gene_dim=978,
        gene_hidden_dim=512,
        drug_hidden_dim=128,
    )

    # Criterion and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3
    )

    train_losses, val_losses = train_multimodal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=20,
        device=device,
        gradient_clipping=1.0,
        early_stopping_patience=5,
        model_name="Perturbinator",
        use_mixed_precision=True,
    )

    # Log the final train and validation losses
    logging.debug("Final train and validation losses:")
    logging.debug(f"Train Losses: {train_losses}")
    logging.debug(f"Validation Losses: {val_losses}")

    # Evaluate on test set
    test_metrics = evaluate_multimodal_model(
        model, test_loader, criterion=criterion, device=device
    )
    logging.debug(f"Test Metrics:")
    logging.debug(f"MSE: {test_metrics['MSE']}, MAE: {test_metrics['MAE']}")
    logging.debug(
        f"R²: {test_metrics['R2']}, Pearson Correlation: {test_metrics['PCC']}"
    )
