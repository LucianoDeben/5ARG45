import torch
import torch.nn as nn
import torch.nn.init as init


class Perturbinator(nn.Module):
    """
    A model that encodes (unperturbed gene expression) + (drug SMILES tokens)
    and predicts (perturbed gene expression, viability, or both).
    """

    def __init__(
        self,
        gene_input_dim=978,
        gene_embed_dim=256,
        drug_embed_dim=256,
        fusion_dim=512,
        pert_output_dim=978,
        task_type="multi-task",
        fusion_type="concat",
        vocab_size=128,
    ):
        super(Perturbinator, self).__init__()

        self.gene_embed_dim = gene_embed_dim
        self.drug_embed_dim = drug_embed_dim
        self.gene_input_dim = gene_input_dim
        self.fusion_dim = fusion_dim
        self.pert_output_dim = pert_output_dim
        self.task_type = task_type
        self.fusion_type = fusion_type
        self.vocab_size = vocab_size

        # Gene expression encoder
        self.gene_encoder = GeneFeatureEncoder(
            self.gene_input_dim,
            hidden_dims=[512, 512, self.gene_embed_dim],
        )

        # Drug feature encoder
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
            gene_embed_dim=self.gene_embed_dim,
            drug_embed_dim=self.drug_embed_dim,
            fusion_dim=self.fusion_dim,
            fusion_type=self.fusion_type,
        )

        # Decoder
        self.decoder = Decoder(
            fusion_dim=self.fusion_dim,
            gene_output_dim=self.pert_output_dim,
            task_type=self.task_type,
        )

        # Apply weight initialization explicitly
        self.apply(initialize_weights)

    def forward(self, gene_expression, smiles_tensor, dosage):
        """
        Forward pass to predict task-specific outputs.

        Args:
            gene_expression (torch.Tensor): Baseline gene expression (B, gene_dim).
            smiles_tensor (torch.Tensor): Tokenized SMILES tensors (B, L).
            dosage (torch.Tensor): Dosage values (B, 1).

        Returns:
            dict: Predictions based on the selected task type.
        """
        # Encode gene expression and drug features
        gene_emb = self.gene_encoder(gene_expression)
        drug_emb = self.drug_encoder(smiles_tensor, dosage)

        # Fuse embeddings
        fused_emb = self.fusion(gene_emb, drug_emb)

        # Decode into task-specific predictions
        output = self.decoder(fused_emb)
        return output


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

        # Check dimensions for broadcasting compatibility
        assert (
            gamma.size() == smiles_emb.size()
        ), "Mismatch in gamma and smiles_emb dimensions"
        assert (
            beta.size() == smiles_emb.size()
        ), "Mismatch in beta and smiles_emb dimensions"

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

        # Apply weight initialization explicitly
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

        assert embeddings.size(1) <= self.positional_encoding.size(
            1
        ), "Token sequence length exceeds maximum positional encoding length"

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

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

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
    """
    Combines gene and drug embeddings using various fusion strategies.
    """

    def __init__(
        self, gene_embed_dim, drug_embed_dim, fusion_dim, fusion_type="concat"
    ):
        """
        Args:
            gene_embed_dim (int): Dimension of gene embeddings.
            drug_embed_dim (int): Dimension of drug embeddings.
            fusion_dim (int): Dimension of the fused embedding.
            fusion_type (str): Type of fusion strategy ("concat", "attention", "gating").
        """
        super(GeneDrugFuser, self).__init__()
        self.gene_embed_dim = gene_embed_dim
        self.drug_embed_dim = drug_embed_dim
        self.fusion_dim = fusion_dim
        self.fusion_type = fusion_type

        # Initialize fusion layers
        self.concat_layer = nn.Sequential(
            nn.Linear(gene_embed_dim + drug_embed_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )
        # TODO: Add attention assertion for gene and drug embedding dimensions
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=4, batch_first=True
        )
        self.attention_fc = nn.Sequential(
            nn.Linear(gene_embed_dim, fusion_dim),
            nn.ReLU(),
        )

        self.gating_layer = nn.Sequential(
            nn.Linear(gene_embed_dim + drug_embed_dim, gene_embed_dim),
            nn.Sigmoid(),
        )
        self.gating_fc = nn.Sequential(
            nn.Linear(gene_embed_dim, fusion_dim),
            nn.ReLU(),
        )

        # Validate fusion type
        if fusion_type not in ["concat", "attention", "gating"]:
            raise ValueError(
                "Unsupported fusion type. Choose 'concat', 'attention', or 'gating'."
            )

    def forward(self, gene_features, drug_features):
        """
        Forward pass to fuse gene and drug embeddings.

        Args:
            gene_features (torch.Tensor): Gene embeddings (B, gene_embed_dim).
            drug_features (torch.Tensor): Drug embeddings (B, drug_embed_dim).

        Returns:
            torch.Tensor: Fused embedding (B, fusion_dim).
        """
        if self.fusion_type == "concat":
            return self._fuse_concat(gene_features, drug_features)
        elif self.fusion_type == "attention":
            return self._fuse_attention(gene_features, drug_features)
        elif self.fusion_type == "gating":
            return self._fuse_gating(gene_features, drug_features)

    def _fuse_concat(self, gene_features, drug_features):
        """
        Concatenates gene and drug embeddings, followed by a fully connected layer.
        """
        combined_features = torch.cat((gene_features, drug_features), dim=1)
        return self.concat_layer(combined_features)

    def _fuse_attention(self, gene_features, drug_features):
        """
        Uses attention to fuse gene and drug embeddings.
        """
        # Add sequence dimension for attention
        gene_features = gene_features.unsqueeze(1)  # Shape: (B, 1, gene_embed_dim)
        drug_features = drug_features.unsqueeze(1)  # Shape: (B, 1, drug_embed_dim)

        # Apply attention (query: gene, key/value: drug)
        attended_features, _ = self.attention_layer(
            query=gene_features,
            key=drug_features,
            value=drug_features,
        )

        # Remove sequence dimension and apply final linear transformation
        attended_features = attended_features.squeeze(1)  # Shape: (B, drug_embed_dim)
        return self.attention_fc(attended_features)

    def _fuse_gating(self, gene_features, drug_features):
        """
        Uses gating to modulate gene embeddings based on drug embeddings.
        """
        combined_features = torch.cat((gene_features, drug_features), dim=1)
        gated_features = self.gating_layer(combined_features) * gene_features
        return self.gating_fc(gated_features)


class Decoder(nn.Module):
    """
    Decodes fused embeddings into task-specific outputs.
    Supports gene expression prediction, viability prediction, or both.
    """

    def __init__(self, fusion_dim, gene_output_dim=978, task_type="multi-task"):
        """
        Args:
            fusion_dim (int): Dimension of the fused embedding.
            gene_output_dim (int): Number of output genes for gene expression prediction.
            task_type (str): Task type ("multi-task", "gene-expression", "viability").
        """
        super(Decoder, self).__init__()
        self.task_type = task_type

        # Shared layers (optional, can also be task-specific)
        self.shared_fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        if self.task_type == "multi-task":
            # Viability branch
            self.viability_fc = nn.Sequential(
                nn.Linear(512, 1),  # Single scalar for viability
                nn.Sigmoid(),  # Range [0, 1]
            )
            # Gene expression branch
            self.gene_expression_fc = nn.Sequential(
                nn.Linear(512, gene_output_dim),  # Vector for gene expression
                nn.ReLU(),  # Ensure non-negative values
            )
        elif self.task_type == "gene-expression":
            # Gene expression prediction only
            self.gene_expression_fc = nn.Sequential(
                nn.Linear(512, gene_output_dim),
                nn.ReLU(),
            )
        elif self.task_type == "viability":
            # Viability prediction only
            self.viability_fc = nn.Sequential(
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(
                "Invalid task_type. Choose from 'multi-task', 'gene-expression', or 'viability'."
            )

        # Apply weight initialization
        self.apply(initialize_weights)

    def forward(self, fused_emb):
        """
        Forward pass to decode fused embeddings.

        Args:
            fused_emb (torch.Tensor): Fused embeddings (B, fusion_dim).

        Returns:
            dict: Predictions based on task type.
        """
        shared_out = self.shared_fc(fused_emb)

        if self.task_type == "multi-task":
            viability = torch.clamp(self.viability_fc(shared_out), 1e-7, 1 - 1e-7)
            gene_expression = self.gene_expression_fc(shared_out)
            return {"viability": viability, "gene_expression": gene_expression}
        elif self.task_type == "gene-expression":
            return {"gene_expression": self.gene_expression_fc(shared_out)}
        elif self.task_type == "viability":
            return {
                "viability": torch.clamp(self.viability_fc(shared_out), 1e-7, 1 - 1e-7)
            }


def initialize_weights(module):
    """
    Applies weight initialization to supported layers.
    """
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):  # For convolutional layers
        init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.TransformerEncoderLayer):
        for param in module.parameters():
            if param.dim() > 1:  # Weight matrices
                init.xavier_uniform_(param)
    elif isinstance(module, nn.BatchNorm1d):  # BatchNorm
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()


class MultiTaskLoss(nn.Module):
    def __init__(self, gene_loss_weight=1.0, viability_loss_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.gene_loss_weight = gene_loss_weight
        self.viability_loss_weight = viability_loss_weight

    def forward(self, outputs, gene_labels=None, viability_labels=None):
        total_loss = 0.0

        # Gene expression loss
        if "gene_expression" in outputs and gene_labels is not None:
            gene_loss = torch.nn.functional.mse_loss(
                outputs["gene_expression"], gene_labels
            )
            total_loss += self.gene_loss_weight * gene_loss

        # Viability prediction loss
        if "viability" in outputs and viability_labels is not None:
            viability_loss = torch.nn.functional.mse_loss(
                outputs["viability"].squeeze(), viability_labels
            )
            total_loss += self.viability_loss_weight * viability_loss

        # Ensure at least one loss is computed
        if total_loss == 0.0:
            raise ValueError(
                "No valid labels or outputs provided for loss calculation."
            )

        return total_loss


if __name__ == "__main__":

    # Add src to path
    import logging
    import os
    import sys

    import wandb

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    import pandas as pd
    import torch
    from deepchem.feat.smiles_tokenizer import SmilesTokenizer
    from torch.utils.data import DataLoader, random_split

    from config import init_wandb
    from data_sets import PerturbationDataset
    from evaluation import evaluate_multimodal_model
    from perturbinator import Perturbinator
    from training import train_multimodal_model
    from utils import create_smiles_dict

    # # Initialize Weights & Biases
    # wandb.init(
    #     project="5ARG45",
    #     name="perturbinator_multitask",
    #     mode="offline",
    # )
    # wandb.config = {
    #     "lr": 0.001,
    #     "architecture": "Perturbinator",
    #     "dataset": "LINCS/CTRPv2",
    #     "epochs": 10,
    #     "batch_size": 512,
    # }

    config = init_wandb()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets and metadata
    controls = "../data/raw/compound_control_dataset.h5"
    perturbations = "../data/raw/compound_perturbation_dataset.h5"
    smiles_df = pd.read_csv("../data/raw/compoundinfo_beta.txt", sep="\t")
    smiles_dict = create_smiles_dict(smiles_df)

    # Load the SMILES tokenizer
    vocab_file = "../data/raw/vocab.txt"
    smiles_tokenizer = SmilesTokenizer(vocab_file=vocab_file)

    # Create dataset
    dataset = PerturbationDataset(
        controls_file=controls,
        perturbations_file=perturbations,
        smiles_dict=smiles_dict,
        plate_column="det_plate",
        normalize="min-max",
        n_rows=50,
        pairing="random",
        landmark_only=True,
        tokenizer=smiles_tokenizer,
        max_smiles_length=128,
    )

    # Split dataset
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

    batch_size = config.get("batch_size", 1024)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = Perturbinator(
        gene_input_dim=978,
        gene_embed_dim=256,
        drug_embed_dim=256,
        fusion_dim=512,
        pert_output_dim=978,
        task_type="gene-expression",  # Multitask learning
        fusion_type="gating",
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # Define multitask loss
    criterion = MultiTaskLoss(gene_loss_weight=1.0, viability_loss_weight=1.0)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.get("lr", 1e-3), weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3
    )

    # Train the model
    train_losses, val_losses = train_multimodal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=config.get("epochs", 25),
        device=device,
        gradient_clipping=1.0,
        early_stopping_patience=5,
        model_name="Perturbinator",
        use_mixed_precision=True,
    )

    # Log training results
    logging.info("Final train and validation losses:")
    logging.info(f"Train Loss: {train_losses[-1]:.4f}")
    logging.info(f"Validation Loss: {val_losses[-1]:.4f}")

    # Evaluate on test set
    test_metrics = evaluate_multimodal_model(
        model, test_loader, criterion=criterion, device=device
    )

    logging.info("Test Metrics:")

    # Log gene expression metrics if available
    if "gene_expression_metrics" in test_metrics:
        logging.info(
            f"Gene Expression Metrics: {test_metrics['gene_expression_metrics']}"
        )
        wandb.log(
            {
                "Gene Expression MSE": test_metrics["gene_expression_metrics"].get(
                    "MSE", None
                ),
                "Gene Expression MAE": test_metrics["gene_expression_metrics"].get(
                    "MAE", None
                ),
                "Gene Expression R2": test_metrics["gene_expression_metrics"].get(
                    "R2", None
                ),
                "Gene Expression PCC": test_metrics["gene_expression_metrics"].get(
                    "PCC", None
                ),
            }
        )

    # Log viability metrics if available
    if "viability_metrics" in test_metrics:
        logging.info(f"Viability Metrics: {test_metrics['viability_metrics']}")
        wandb.log(
            {
                "Viability MSE": test_metrics["viability_metrics"].get("MSE", None),
                "Viability MAE": test_metrics["viability_metrics"].get("MAE", None),
                "Viability R2": test_metrics["viability_metrics"].get("R2", None),
                "Viability PCC": test_metrics["viability_metrics"].get("PCC", None),
            }
        )

    wandb.finish()
