# src/models/molecular_module.py
from src.models.base_module import DrugResponseModule

class MolecularModule(DrugResponseModule):
    """Module specifically for molecular-based models"""
    
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
    
    def forward(self, x):
        # For molecular models, extract the molecular data if needed
        if isinstance(x, dict):
            if "molecular" in x:
                # Some models might need dosage information
                if "dosage" in x and hasattr(self.model, "needs_dosage") and self.model.needs_dosage:
                    return self.model(x["molecular"], x["dosage"])
                return self.model(x["molecular"])
        return self.model(x)