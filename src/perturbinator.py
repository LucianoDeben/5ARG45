import torch
import torch.nn as nn
import torch.nn.init as init

class Perturbinator(nn.Module):
    """
    A model that encodes (unperturbed gene expression) + (drug SMILES tokens)
    and predicts (perturbed gene expression).
    """

    def __init__(
        self,
        gene_input_dim=978,
        gene_embed_dim=256,
        drug_embed_dim=256,
        pert_output_dim=978,
        fusion_type="concat",
        vocab_size=128,
    ):
        super(Perturbinator, self).__init__()

        self.gene_embed_dim = gene_embed_dim
        self.drug_embed_dim = drug_embed_dim
        self.gene_input_dim = gene_input_dim
        self.pert_output_dim = pert_output_dim
        self.fusion_type = fusion_type
        self.vocab_size = vocab_size

        # Gene expression encoder
        self.gene_encoder = GeneFeatureEncoder(
            self.gene_input_dim,
            hidden_dims=[
                512,
                512,
                self.gene_embed_dim,
            ],
        )

        # Drug feature encoder (accepts tokenized SMILES)
        self.drug_encoder = DrugFeatureEncoder(
            drug_embed_dim=self.drug_embed_dim,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
            vocab_size=self.vocab_size,
        )

        # Fusion layer
        self.fusion = GeneDrugFuser(
            self.gene_embed_dim,
            self.drug_embed_dim,
            self.pert_output_dim,
            self.fusion_type,
        )

    def forward(self, gene_expression, smiles_tensor, dosage):
        """
        Forward pass to predict perturbed gene expression.

        Args:
            gene_expression (torch.Tensor): Baseline gene expression (B, gene_dim).
            smiles_tokens (torch.Tensor): Tokenized SMILES tensors (B, L).
        Returns:
            torch.Tensor: Predicted perturbed gene expression (B, gene_dim).
        """
        gene_emb = self.gene_encoder(gene_expression)
        drug_emb = self.drug_encoder(smiles_tensor, dosage)

        return self.fusion(gene_emb, drug_emb)


class DrugFeatureEncoder(nn.Module):
    def __init__(
        self,
        drug_embed_dim=512,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        vocab_size=128,
    ):
        super().__init__()
        self.smiles_encoder = TransformerSMILESEncoder(
            vocab_size=vocab_size,
            drug_embed_dim=drug_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        # Dosage scaling (gamma) and shifting (beta) MLPs
        self.dosage_mlp_gamma = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Input is scalar dosage
            nn.ReLU(),
            nn.Linear(hidden_dim, drug_embed_dim),  # Scale matches embedding dim
            nn.Sigmoid(),  # Sigmoid gating for gamma
        )
        self.dosage_mlp_beta = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, drug_embed_dim),  # No gating for beta
        )

    def forward(self, smiles_tensor, dosage):
        """
        Forward pass for drug feature encoding with dosage modulation.

        Args:
            smiles_tensor (torch.Tensor): Tokenized SMILES tensors (B, L).
            dosage (torch.Tensor): Dosage values (B, 1).

        Returns:
            torch.Tensor: Modulated drug embeddings (B, embed_dim).
        """
        smiles_emb = self.smiles_encoder(smiles_tensor)  # Shape: (B, embed_dim)

        # Dosage modulation
        gamma = self.dosage_mlp_gamma(dosage.unsqueeze(-1))  # Shape: (B, embed_dim)
        beta = self.dosage_mlp_beta(dosage.unsqueeze(-1))  # Shape: (B, embed_dim)

        # Apply sigmoid gating modulation
        modulated_smiles_emb = gamma * smiles_emb + beta
        return modulated_smiles_emb


class TransformerSMILESEncoder(nn.Module):
    """
    Transformer-based SMILES encoder that takes pre-tokenized SMILES tensors.
    """

    def __init__(
        self,
        drug_embed_dim=512,
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
        self.token_embedding = nn.Embedding(vocab_size, drug_embed_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_length, drug_embed_dim)
        )  # Max positional encoding

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=drug_embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=nn.GELU(),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(drug_embed_dim)

        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Apply Xavier initialization
        self.apply(initialize_weights)

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

        # # Apply layer normalization
        embeddings = self.layer_norm(embeddings)

        # Pass through Transformer encoder
        transformer_output = self.transformer(
            embeddings.transpose(0, 1)
        )  # (L, B, embed_dim)

        # # Apply layer normalization after transformer
        transformer_output = self.layer_norm(transformer_output)

        # Pool to get a single vector per SMILES
        pooled_output = self.pooling(
            transformer_output.transpose(0, 1).permute(0, 2, 1)
        ).squeeze(-1)

        return pooled_output


class GeneFeatureEncoder(nn.Module):
    def __init__(self, gene_dim, hidden_dims, dropout_rate=0.2):
        """
        Args:
            gene_dim (int): Input dimension of gene features.
            hidden_dims (list of int): List of hidden dimensions for each layer.
            dropout_rate (float): Dropout rate.
        """
        super(GeneFeatureEncoder, self).__init__()

        layers = []
        input_dim = gene_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        self.gene_encoder = nn.Sequential(*layers)

        # Apply Xavier initialization
        self.apply(initialize_weights)

    def forward(self, x):
        return self.gene_encoder(x)


class GeneDrugFuser(nn.Module):
    def __init__(self, gene_embed_dim, drug_embed_dim, pert_embed_dim, fusion_type="concat"):
        super(GeneDrugFuser, self).__init__()
        self.fusion_type = fusion_type

        if self.fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(gene_embed_dim + drug_embed_dim, gene_embed_dim),
                nn.ReLU(),
                nn.Linear(gene_embed_dim, pert_embed_dim),
            )
        elif self.fusion_type == "attention":
            # Cross-attention: Gene as Query, Drug as Key/Value
            self.attention = nn.MultiheadAttention(
                embed_dim=drug_embed_dim,  # Match drug embedding dimension
                num_heads=4,  # Number of attention heads
                batch_first=True  # Ensures input/output shape matches (B, L, D)
            )
            self.fusion = nn.Sequential(
                nn.Linear(gene_embed_dim, pert_embed_dim),  # Final fused representation
                nn.ReLU(),
            )
        elif self.fusion_type == "gating":
            self.gate = nn.Sequential(
                nn.Linear(gene_embed_dim + drug_embed_dim, gene_embed_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(gene_embed_dim, pert_embed_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(
                "Unsupported fusion type. Choose from 'concat', 'attention', or 'gating'."
            )

    def forward(self, gene_features, drug_features):
        if self.fusion_type == "concat":
            combined_features = torch.cat((gene_features, drug_features), dim=1)
            return self.fusion(combined_features)
        elif self.fusion_type == "attention":
            # Expand dimensions for attention (add sequence dimension)
            gene_features = gene_features.unsqueeze(1)  # Shape: (B, 1, gene_embed_dim)
            drug_features = drug_features.unsqueeze(1)  # Shape: (B, 1, drug_embed_dim)

            # Cross-attention: Gene as Query, Drug as Key/Value
            attended_features, _ = self.attention(
                query=gene_features,  # Shape: (B, 1, drug_embed_dim)
                key=drug_features,    # Shape: (B, 1, drug_embed_dim)
                value=drug_features   # Shape: (B, 1, drug_embed_dim)
            )

            # Remove the sequence dimension
            attended_features = attended_features.squeeze(1)  # Shape: (B, drug_embed_dim)

            # Fuse attended features with gene features
            return self.fusion(attended_features)
        elif self.fusion_type == "gating":
            combined_features = torch.cat((gene_features, drug_features), dim=1)
            gated_features = self.gate(combined_features) * gene_features
            return self.fusion(gated_features)



def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.TransformerEncoderLayer):
        for param in module.parameters():
            if param.dim() > 1:
                init.xavier_uniform_(param)


if __name__ == "__main__":
    
    # Add src to path
    import logging
    import wandb

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    import os
    import sys
    from config import init_wandb

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
    
    wandb.init(
    project="5ARG45",
    name="perturbinator",
    mode="offline",
    )

    wandb.config = {
        "lr": 0.001,
        "architecture": "Perturbinator",
        "dataset": "LINCS/CTRPv2",
        "epochs": 20,
        "batch_size": 1024,
    }
    
    config = wandb.config   
    
    # config = init_wandb()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the datasets
    controls = "../data/raw/compound_control_dataset.h5"
    perturbations = "../data/raw/compound_perturbation_dataset.h5"

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
        n_rows=20000,
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

    batch_size = config.get("batch_size", 1024)
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
        gene_input_dim=978,
        gene_embed_dim=256,
        drug_embed_dim=256,
        pert_output_dim=978,
        fusion_type="gating",
    )

    # Criterion and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=1e-5)
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
    logging.debug(f"Train Losses: {train_losses[-1]}")
    logging.debug(f"Validation Losses: {val_losses[-1]}")

    # Evaluate on test set
    test_metrics = evaluate_multimodal_model(
        model, test_loader, criterion=criterion, device=device
    )
    logging.debug(f"Test Metrics:")
    logging.debug(f"MSE: {test_metrics['MSE']}, MAE: {test_metrics['MAE']}")
    logging.debug(
        f"R²: {test_metrics['R2']}, Pearson Correlation: {test_metrics['PCC']}"
    )
    
    wandb.log({"MSE": test_metrics["MSE"], "MAE": test_metrics["MAE"],"R2": test_metrics["R2"], "PCC": test_metrics["PCC"]})

    wandb.finish()