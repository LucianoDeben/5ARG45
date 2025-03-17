# models/molecular/encoders.py
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to token embeddings for transformer models.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MolecularMLP(nn.Module):
    """
    Simple MLP encoder for processing molecular fingerprints.
    
    This encoder is designed to work with pre-computed fingerprints 
    (like ECFP/Morgan fingerprints) and optional dosage information.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 128,
        activation: str = "relu",
        normalize: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize the MolecularMLP encoder.
        
        Args:
            input_dim: Dimension of the input features (fingerprint size)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of the output representation
            activation: Activation function to use ('relu', 'gelu', etc.)
            normalize: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.needs_dosage = False  # Flag for MultimodalModel
        
        # Define activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "silu" or activation.lower() == "swish":
            self.activation = nn.SiLU()
        elif activation.lower() == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Add linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add batch norm if requested
            if normalize:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            # Add activation
            layers.append(self.activation)
            
            # Add dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create encoder
        self.encoder = nn.Sequential(*layers)
        
        logger.debug(
            f"Initialized MolecularMLP with input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, output_dim={output_dim}"
        )
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass for molecular fingerprint data.
        
        Args:
            x: Either:
                - Tensor [batch_size, input_dim] of molecular fingerprints
                - Dictionary with 'molecular' key containing fingerprint tensor
        
        Returns:
            Tensor [batch_size, output_dim] with molecular representation
        """
        # Handle dictionary input format for compatibility with multimodal model
        if isinstance(x, dict):
            if "molecular" in x:
                x = x["molecular"]
            elif "fingerprint" in x:
                x = x["fingerprint"]
            else:
                raise ValueError("Expected dictionary with 'molecular' or 'fingerprint' key")
        
        # Ensure proper shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        return self.encoder(x)


class MolecularWithDosageEncoder(nn.Module):
    """
    MLP encoder that explicitly integrates dosage information with molecular fingerprints.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 128,
        dosage_integration: str = "concat",
        activation: str = "relu",
        normalize: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize the MolecularWithDosageEncoder.
        
        Args:
            input_dim: Dimension of the input features (fingerprint size)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of the output representation
            dosage_integration: How to integrate dosage ('concat', 'scale', 'bilinear')
            activation: Activation function to use
            normalize: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dosage_integration = dosage_integration.lower()
        self.needs_dosage = True  # Flag for MultimodalModel
        
        # Fingerprint encoder to intermediate dimension
        fingerprint_output_dim = hidden_dims[-1]
        self.fingerprint_encoder = MolecularMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else [input_dim//2],
            output_dim=fingerprint_output_dim,
            activation=activation,
            normalize=normalize,
            dropout=dropout
        )
        
        # Dosage integration
        if self.dosage_integration == "concat":
            self.projection = nn.Linear(fingerprint_output_dim + 1, output_dim)
        elif self.dosage_integration == "scale":
            self.dosage_gates = nn.Linear(1, fingerprint_output_dim)
            self.projection = nn.Linear(fingerprint_output_dim, output_dim)
        elif self.dosage_integration == "bilinear":
            self.bilinear = nn.Bilinear(fingerprint_output_dim, 1, fingerprint_output_dim)
            self.projection = nn.Linear(fingerprint_output_dim, output_dim)
        else:
            raise ValueError(f"Invalid dosage integration: {dosage_integration}")
        
        logger.debug(
            f"Initialized MolecularWithDosageEncoder with input_dim={input_dim}, "
            f"output_dim={output_dim}, dosage_integration={dosage_integration}"
        )
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], dosage: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for molecular fingerprint data with dosage.
        
        Args:
            x: Either:
                - Tensor [batch_size, input_dim] of molecular fingerprints
                - Dictionary with 'molecular' key containing fingerprint tensor and 'dosage' key
            dosage: Optional tensor [batch_size, 1] with dosage information
        
        Returns:
            Tensor [batch_size, output_dim] with dosage-integrated molecular representation
        """
        # Extract fingerprint and dosage from dictionary if needed
        if isinstance(x, dict):
            if "molecular" in x:
                fingerprint = x["molecular"]
                if dosage is None and "dosage" in x:
                    dosage = x["dosage"]
            elif "fingerprint" in x:
                fingerprint = x["fingerprint"]
                if dosage is None and "dosage" in x:
                    dosage = x["dosage"]
            else:
                raise ValueError("Expected dictionary with 'molecular'/'fingerprint' and 'dosage' keys")
        else:
            fingerprint = x
            
        # Ensure dosage is available
        if dosage is None:
            raise ValueError("Dosage information is required but not provided")
            
        # Ensure dosage has proper shape
        if dosage.dim() == 1:
            dosage = dosage.unsqueeze(1)
            
        # Process fingerprint
        fingerprint_embedding = self.fingerprint_encoder(fingerprint)
        
        # Integrate with dosage
        if self.dosage_integration == "concat":
            x = torch.cat([fingerprint_embedding, dosage], dim=1)
            return self.projection(x)
        elif self.dosage_integration == "scale":
            scaling = torch.sigmoid(self.dosage_gates(dosage))
            x = fingerprint_embedding * scaling
            return self.projection(x)
        elif self.dosage_integration == "bilinear":
            x = self.bilinear(fingerprint_embedding, dosage)
            x = F.relu(x)
            return self.projection(x)


class SMILESTransformerEncoder(nn.Module):
    """
    Process SMILES using pretrained or custom transformers.
    
    This encoder uses transformer architectures for processing SMILES strings,
    either using pretrained models or training from scratch.
    """

    def __init__(
        self,
        output_dim: int,
        pretrained_model: Optional[str] = None,
        custom_vocab_file: Optional[str] = None,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 128,
    ):
        super(SMILESTransformerEncoder, self).__init__()
        self.output_dim = output_dim
        self.max_length = max_length
        self.needs_dosage = False  # Flag for MultimodalModel

        if pretrained_model:
            # Use pretrained transformer model (e.g., ChemBERTa)
            logger.info(f"Using pretrained model: {pretrained_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.transformer = AutoModel.from_pretrained(pretrained_model)
            self.model_dim = self.transformer.config.hidden_size
        else:
            # Custom transformer implementation
            logger.info("Using custom transformer implementation")
            # Initialize tokenizer similar to SMILESEncoder
            try:
                self.tokenizer = SmilesTokenizer(vocab_file=custom_vocab_file)
                self.vocab_size = self.tokenizer.vocab_size
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
                self.vocab_size = 100

            self.model_dim = embedding_dim
            self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
            self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

            encoder_layers = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layers, num_layers=num_layers
            )

        # Output projection
        self.projection = nn.Linear(self.model_dim, output_dim)

    def forward(
        self, x: Union[Dict[str, Union[torch.Tensor, List[str]]], torch.Tensor]
    ) -> torch.Tensor:
        # Process input
        if isinstance(x, dict):
            if "smiles" not in x:
                raise ValueError("Expected dictionary with 'smiles' key")

            smiles_data = x["smiles"]
            if isinstance(smiles_data, list):
                # Tokenize SMILES strings
                if hasattr(self, "tokenizer") and self.tokenizer is not None:
                    tokens = self.tokenizer(
                        smiles_data,
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = tokens["input_ids"]
                    attention_mask = tokens["attention_mask"]
                else:
                    raise RuntimeError("Tokenizer not available")
            else:
                # Already tokenized
                input_ids = smiles_data
                attention_mask = x.get("attention_mask")
        else:
            # Direct tensor input
            input_ids = x
            attention_mask = None

        # Process through transformer
        if hasattr(self, "embedding"):
            # Custom transformer
            embedded = self.embedding(input_ids)
            embedded = self.pos_encoder(embedded)

            if attention_mask is not None:
                src_key_padding_mask = ~attention_mask.bool()
                output = self.transformer(
                    embedded, src_key_padding_mask=src_key_padding_mask
                )
            else:
                output = self.transformer(embedded)

            # Mean pooling
            output = output.mean(dim=1)
        else:
            # Pretrained transformer
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            # Use [CLS] token or mean pooling
            output = outputs.last_hidden_state.mean(dim=1)

        # Project to output dimension
        return self.projection(output)


# Factory function for creating molecular encoders
def create_molecular_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of molecular encoders.
    
    Args:
        encoder_type: Type of encoder ("mlp", "dosage", "transformer")
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output features
        **kwargs: Additional arguments specific to encoder types
    
    Returns:
        A molecular encoder module
    """
    if encoder_type == "mlp":
        return MolecularMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=kwargs.get("activation", "relu"),
            normalize=kwargs.get("normalize", True),
            dropout=kwargs.get("dropout", 0.3)
        )
    elif encoder_type == "dosage":
        return MolecularWithDosageEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dosage_integration=kwargs.get("dosage_integration", "concat"),
            activation=kwargs.get("activation", "relu"),
            normalize=kwargs.get("normalize", True),
            dropout=kwargs.get("dropout", 0.3)
        )
    elif encoder_type == "transformer":
        return SMILESTransformerEncoder(
            output_dim=output_dim,
            pretrained_model=kwargs.get("pretrained_model"),
            custom_vocab_file=kwargs.get("custom_vocab_file"),
            embedding_dim=kwargs.get("embedding_dim", 256),
            num_layers=kwargs.get("num_layers", 6),
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1),
            max_length=kwargs.get("max_length", 128)
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")