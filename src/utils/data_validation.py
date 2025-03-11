# utils/data_validation.py
from typing import Any, Dict, List, Union, Tuple


def validate_batch(
    batch: Union[Dict[str, Any], Tuple, List], required_keys: List[str] = ["viability"]
) -> None:
    """
    Validate that a batch has the expected structure.
    
    Args:
        batch: Batch of data (dict, tuple, or list)
        required_keys: Keys required if batch is a dictionary
    """        
    # For dictionary-based datasets
    if not isinstance(batch, dict):
        raise ValueError(f"Batch must be a dict, got {type(batch)}")
    
    # Check required keys for dictionary batches
    for key in required_keys:
        if key not in batch:
            raise ValueError(f"Batch missing required key: {key}")