# models/chemical/transformers.py

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from transformers import AutoModel, AutoTokenizer

from src.models.chemical.smiles_processing import PositionalEncoding

logger = logging.getLogger(__name__)


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
