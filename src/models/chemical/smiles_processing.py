# models/chemical/smiles_processing.py
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

logger = logging.getLogger(__name__)


class SMILESEncoder(nn.Module):
    """
    Process SMILES strings using sequence-based neural networks.

    This encoder converts SMILES strings into fixed-size representations
    using either CNN or RNN architectures. It supports pre-tokenized
    sequences or can tokenize raw SMILES strings directly.

    Attributes:
        embedding_dim: Dimension of token embeddings
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of the output representation
        architecture: Type of architecture to use ('cnn' or 'rnn')
        vocab_file: Path to vocabulary file for tokenization
        max_length: Maximum SMILES sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        architecture: str = "cnn",
        vocab_file: str = "data/raw/vocab.txt",
        max_length: int = 128,
        dropout: float = 0.2,
    ):
        """
        Initialize the SMILESEncoder.

        Args:
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of the output representation
            architecture: Type of architecture to use ('cnn' or 'rnn')
            vocab_file: Path to vocabulary file for tokenization
            max_length: Maximum SMILES sequence length
            dropout: Dropout rate
        """
        super(SMILESEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.architecture = architecture.lower()
        self.max_length = max_length
        self.dropout = dropout

        # Initialize tokenizer
        try:
            self.tokenizer = SmilesTokenizer(vocab_file=vocab_file)
            self.vocab_size = self.tokenizer.vocab_size
            logger.debug(f"Initialized tokenizer with vocab size {self.vocab_size}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer, using default vocabulary: {e}")
            # Default to approximate vocabulary size if tokenizer fails
            self.tokenizer = None
            self.vocab_size = 100  # Approximate size for common SMILES tokens

        # Token embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)

        # Architecture-specific layers
        if self.architecture == "cnn":
            self.processor = nn.Sequential(
                nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
        elif self.architecture == "rnn":
            self.processor = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0,
                bidirectional=True,
            )
            # Adjust for bidirectional
            self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. Use 'cnn' or 'rnn'."
            )

        # Output projection
        self.output = nn.Linear(hidden_dim, output_dim)

        logger.debug(
            f"Initialized SMILESEncoder with {architecture} architecture, "
            f"embedding_dim={embedding_dim}, output_dim={output_dim}"
        )

    def tokenize_smiles(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Tokenize a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to tokenize

        Returns:
            Tensor of token indices of shape [batch_size, max_length]
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available. Check if vocab file exists.")

        token_indices = []
        for smiles in smiles_list:
            # Tokenize and convert to indices, padding to max_length
            tokens = self.tokenizer.tokenize(smiles)
            indices = self.tokenizer.convert_tokens_to_ids(tokens)

            # Padding or truncation
            if len(indices) > self.max_length:
                indices = indices[: self.max_length]
            else:
                indices = indices + [0] * (self.max_length - len(indices))

            token_indices.append(indices)

        return torch.tensor(token_indices, dtype=torch.long)

    def forward(
        self, x: Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]]
    ) -> torch.Tensor:
        """
        Forward pass for SMILES data.

        Args:
            x: Either:
                - Tensor [batch_size, seq_length] of pre-tokenized SMILES
                - Dictionary with 'smiles' key containing either:
                    - List of SMILES strings to be tokenized
                    - Tensor of pre-tokenized SMILES

        Returns:
            Tensor [batch_size, output_dim] with SMILES embedding
        """
        # Handle different input types
        if isinstance(x, dict):
            if "smiles" not in x:
                raise ValueError("Expected dictionary with 'smiles' key")

            smiles_data = x["smiles"]
            if isinstance(smiles_data, list):
                # Tokenize SMILES strings
                smiles_sequences = self.tokenize_smiles(smiles_data)
            else:
                # Already tokenized
                smiles_sequences = smiles_data
        else:
            # Direct tensor input
            smiles_sequences = x

        # Validate input
        if smiles_sequences.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor for SMILES sequences, got shape {smiles_sequences.shape}"
            )

        if smiles_sequences.dtype not in (torch.long, torch.int):
            raise TypeError(
                f"Expected Long or Int tensor for SMILES sequences, got {smiles_sequences.dtype}"
            )

        # Process through embedding layer
        embedded = self.embedding(smiles_sequences)

        # Process through architecture-specific layers
        if self.architecture == "cnn":
            # Transpose for CNN: [batch, seq_len, embed_dim] -> [batch, embed_dim, seq_len]
            embedded_transposed = embedded.transpose(1, 2)
            output = self.processor(embedded_transposed).squeeze(-1)
        elif self.architecture == "rnn":
            # Process through LSTM
            output, (h_n, _) = self.processor(embedded)
            # Combine forward and backward directions
            output = torch.cat([h_n[-2], h_n[-1]], dim=1)
            output = F.relu(self.hidden_projection(output))

        # Apply dropout and project to output dimension
        output = F.dropout(output, p=self.dropout, training=self.training)
        return self.output(output)


class SMILESWithDosageEncoder(nn.Module):
    """
    Integrated encoder that combines SMILES representations with dosage information.

    This module processes SMILES strings alongside dosage information
    to create a comprehensive molecular representation.

    Attributes:
        smiles_encoder: The SMILESEncoder module
        output_dim: Dimension of the final representation
        dosage_integration: Method to integrate dosage information
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        architecture: str = "cnn",
        dosage_integration: str = "concat",
        vocab_file: str = "data/raw/vocab.txt",
        max_length: int = 128,
        dropout: float = 0.2,
    ):
        """
        Initialize the SMILESWithDosageEncoder.

        Args:
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of the output representation
            architecture: Type of architecture to use ('cnn' or 'rnn')
            dosage_integration: How to integrate dosage ('concat', 'scale', 'bilinear')
            vocab_file: Path to vocabulary file for tokenization
            max_length: Maximum SMILES sequence length
            dropout: Dropout rate
        """
        super(SMILESWithDosageEncoder, self).__init__()
        self.dosage_integration = dosage_integration.lower()

        # SMILES encoder (output to hidden_dim to allow for integration)
        self.smiles_encoder = SMILESEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Intermediate output
            architecture=architecture,
            vocab_file=vocab_file,
            max_length=max_length,
            dropout=dropout,
        )

        # Dosage integration
        if self.dosage_integration == "concat":
            self.projection = nn.Linear(hidden_dim + 1, output_dim)
        elif self.dosage_integration == "scale":
            self.dosage_gates = nn.Linear(1, hidden_dim)
            self.projection = nn.Linear(hidden_dim, output_dim)
        elif self.dosage_integration == "bilinear":
            self.bilinear = nn.Bilinear(hidden_dim, 1, hidden_dim)
            self.projection = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Invalid dosage integration: {dosage_integration}")

        self.output_dim = output_dim
        logger.debug(
            f"Initialized SMILESWithDosageEncoder with dosage_integration={dosage_integration}, "
            f"architecture={architecture}, output_dim={output_dim}"
        )

    def forward(self, data: Dict[str, Union[torch.Tensor, List[str]]]) -> torch.Tensor:
        """
        Forward pass for SMILES data with dosage.

        Args:
            data: Dictionary containing:
                - 'smiles': Either list of SMILES strings or tensor of tokenized SMILES
                - 'dosage': Tensor [batch_size, 1] with dosage information

        Returns:
            Tensor [batch_size, output_dim] with integrated molecular embedding
        """
        if "smiles" not in data or "dosage" not in data:
            raise ValueError("Expected dictionary with 'smiles' and 'dosage' keys")

        # Extract data
        smiles_data = data["smiles"]
        dosage = data["dosage"]

        # Ensure dosage is proper shape
        if dosage.dim() == 1:
            dosage = dosage.unsqueeze(1)

        # Process SMILES
        smiles_embedding = self.smiles_encoder(smiles_data)

        # Integrate with dosage
        if self.dosage_integration == "concat":
            x = torch.cat([smiles_embedding, dosage], dim=1)
            return self.projection(x)
        elif self.dosage_integration == "scale":
            scaling = torch.sigmoid(self.dosage_gates(dosage))
            x = smiles_embedding * scaling
            return self.projection(x)
        elif self.dosage_integration == "bilinear":
            x = self.bilinear(smiles_embedding, dosage)
            x = F.relu(x)
            return self.projection(x)


class AttentionSMILESEncoder(nn.Module):
    """
    Process SMILES strings using attention mechanisms for better representation.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        vocab_file: str = "data/raw/vocab.txt",
        max_length: int = 128,
        dropout: float = 0.2,
    ):
        super(AttentionSMILESEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.dropout = dropout

        # Initialize tokenizer
        try:
            self.tokenizer = SmilesTokenizer(vocab_file=vocab_file)
            self.vocab_size = self.tokenizer.vocab_size
        except Exception as e:
            logger.warning(f"Failed to load tokenizer, using default vocabulary: {e}")
            self.tokenizer = None
            self.vocab_size = 100

        # Token embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Output projection
        self.output = nn.Linear(embedding_dim, output_dim)

    def tokenize_smiles(
        self, smiles_list: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a list of SMILES strings.

        Returns:
            Tuple of (token_indices, attention_mask)
        """
        # Implementation similar to SMILESEncoder.tokenize_smiles
        # Plus creating attention mask for padding tokens

        token_indices = []
        attention_masks = []

        for smiles in smiles_list:
            tokens = self.tokenizer.tokenize(smiles)
            indices = self.tokenizer.convert_tokens_to_ids(tokens)

            # Create attention mask (0 for padding, 1 for actual tokens)
            mask = [1] * len(indices)

            # Padding or truncation
            if len(indices) > self.max_length:
                indices = indices[: self.max_length]
                mask = mask[: self.max_length]
            else:
                pad_length = self.max_length - len(indices)
                indices = indices + [0] * pad_length
                mask = mask + [0] * pad_length

            token_indices.append(indices)
            attention_masks.append(mask)

        return (
            torch.tensor(token_indices, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.bool),
        )

    def forward(
        self, x: Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]]
    ) -> torch.Tensor:
        # Handle different input types like in SMILESEncoder
        # Then process through the transformer with attention

        # Get token sequences and attention mask
        if isinstance(x, dict):
            if "smiles" not in x:
                raise ValueError("Expected dictionary with 'smiles' key")

            smiles_data = x["smiles"]
            if isinstance(smiles_data, list):
                smiles_sequences, attention_mask = self.tokenize_smiles(smiles_data)
            else:
                # Assume pre-tokenized with attention mask
                smiles_sequences = smiles_data
                attention_mask = x.get("attention_mask", None)
        else:
            # Direct tensor input
            smiles_sequences = x
            attention_mask = None

        # Embedding with positional encoding
        embedded = self.embedding(smiles_sequences)
        embedded = self.pos_encoder(embedded)

        # Apply transformer with attention mask if available
        if attention_mask is not None:
            # Convert boolean mask to transformer-compatible mask
            src_key_padding_mask = ~attention_mask
            output = self.transformer_encoder(
                embedded, src_key_padding_mask=src_key_padding_mask
            )
        else:
            output = self.transformer_encoder(embedded)

        # Use [CLS] token output or mean pooling
        output = output.mean(dim=1)  # Mean pooling across sequence length

        # Apply final projection
        return self.output(output)


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embedding.
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
