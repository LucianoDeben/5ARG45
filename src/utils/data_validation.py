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
    # For TranscriptomicsDataset, which returns (x, y) tuples
    if isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise ValueError(f"Tuple/list batch should have 2 elements, got {len(batch)}")
        # The second element should be the target/viability
        if not hasattr(batch[1], "shape"):
            raise ValueError("Second element of tuple/list batch should be a tensor")
        return
        
    # For dictionary-based datasets
    if not isinstance(batch, dict):
        raise ValueError(f"Batch must be a dict, tuple, or list, got {type(batch)}")
    
    # Check required keys for dictionary batches
    for key in required_keys:
        if key not in batch:
            raise ValueError(f"Batch missing required key: {key}")