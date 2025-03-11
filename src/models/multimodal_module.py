# src/models/multimodal_module.py
import torch
from typing import Dict, Any, Optional
from src.models.base_module import DrugResponseModule

class MultimodalDrugResponseModule(DrugResponseModule):
    def __init__(
        self,
        transcriptomics_encoder: torch.nn.Module,
        molecular_encoder: torch.nn.Module,
        fusion_module: torch.nn.Module,
        prediction_head: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        # Create the complete model
        model = MultimodalModel(
            transcriptomics_encoder=transcriptomics_encoder,
            molecular_encoder=molecular_encoder,
            fusion_module=fusion_module,
            prediction_head=prediction_head
        )
        
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Save individual components for potential separate optimization
        self.transcriptomics_encoder = transcriptomics_encoder
        self.molecular_encoder = molecular_encoder
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head

class MultimodalModel(torch.nn.Module):
    def __init__(
        self,
        transcriptomics_encoder: torch.nn.Module,
        molecular_encoder: torch.nn.Module,
        fusion_module: torch.nn.Module,
        prediction_head: torch.nn.Module,
    ):
        super().__init__()
        self.transcriptomics_encoder = transcriptomics_encoder
        self.molecular_encoder = molecular_encoder
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        transcriptomics_features = self.transcriptomics_encoder(x["transcriptomics"])
        molecular_features = self.molecular_encoder(x["molecular"])
        
        # Include dosage information if needed
        if "dosage" in x:
            fused_features = self.fusion_module(
                transcriptomics_features, molecular_features, x["dosage"]
            )
        else:
            fused_features = self.fusion_module(
                transcriptomics_features, molecular_features
            )
        
        output = self.prediction_head(fused_features)
        return output