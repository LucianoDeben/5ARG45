# src/models/multimodal_model.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any

from src.models.base_module import DrugResponseModule

class MultimodalModel(nn.Module):
    """
    Multimodal model that combines transcriptomics and molecular encoders
    through a fusion module.
    """
    def __init__(
        self,
        transcriptomics_encoder: nn.Module,
        molecular_encoder: nn.Module,
        fusion_module: nn.Module,
        prediction_head: nn.Module,
    ):
        super().__init__()
        self.transcriptomics_encoder = transcriptomics_encoder
        self.molecular_encoder = molecular_encoder
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the multimodal model.
        
        Args:
            x: Dictionary containing 'transcriptomics', 'molecular', and optionally 'dosage'
        
        Returns:
            Tensor with predicted viability values
        """
        # Extract transcriptomics features
        transcriptomics_features = self.transcriptomics_encoder(x["transcriptomics"])
        
        # Extract molecular features (which may include dosage information)
        if "molecular" in x and x["molecular"] is not None:
            molecular_input = x["molecular"]
            # Some molecular models may need dosage separately
            if hasattr(self.molecular_encoder, "needs_dosage") and self.molecular_encoder.needs_dosage:
                molecular_features = self.molecular_encoder(molecular_input, x.get("dosage"))
            else:
                molecular_features = self.molecular_encoder(molecular_input)
        else:
            # Create a zero tensor with appropriate dimensions if molecular data is missing
            device = transcriptomics_features.device
            batch_size = transcriptomics_features.size(0)
            
            # Determine output dimension of molecular encoder
            if hasattr(self.molecular_encoder, "output_dim"):
                mol_dim = self.molecular_encoder.output_dim
            else:
                # Default dimension if not specified
                mol_dim = 128
                
            molecular_features = torch.zeros(batch_size, mol_dim, device=device)
        
        # Include dosage information for fusion if needed
        if hasattr(self.fusion_module, "needs_dosage") and self.fusion_module.needs_dosage and "dosage" in x:
            fused_features = self.fusion_module(
                transcriptomics_features, molecular_features, x["dosage"]
            )
        else:
            fused_features = self.fusion_module(
                transcriptomics_features, molecular_features
            )
        
        # Final prediction
        output = self.prediction_head(fused_features)
        return output


class MultimodalDrugResponseModule(DrugResponseModule):
    """
    Lightning module for multimodal drug response prediction.
    Combines transcriptomics and molecular data encoders with a fusion module.
    """
    
    def __init__(
        self,
        transcriptomics_encoder: nn.Module,
        molecular_encoder: nn.Module,
        fusion_module: nn.Module,
        prediction_head: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: Optional[float] = None,
        **kwargs
    ):
        # Create the complete model
        model = MultimodalModel(
            transcriptomics_encoder=transcriptomics_encoder,
            molecular_encoder=molecular_encoder,
            fusion_module=fusion_module,
            prediction_head=prediction_head
        )
        
        # Initialize parent class
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            **kwargs
        )
        
        # Save individual components for potential separate optimization
        # or feature analysis
        self.transcriptomics_encoder = transcriptomics_encoder
        self.molecular_encoder = molecular_encoder
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head
    
    def configure_optimizers(self):
        """
        Configure optimizers with potential for different learning rates
        for different components.
        """
        # Default optimizer configuration
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    def get_feature_importance(self, batch):
        """
        Optional method to compute feature importance for interpretability.
        Implement based on specific attribution methods needed.
        """
        pass