# src/models/transcriptomics_module.py
from src.models.base_module import DrugResponseModule

class TranscriptomicsModule(DrugResponseModule):
    """Module specifically for transcriptomics-based models"""
    
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
    
    def forward(self, x):
        # For transcriptomics models, extract the transcriptomics data if needed
        if isinstance(x, dict) and "transcriptomics" in x:
            return self.model(x["transcriptomics"])
        return self.model(x)