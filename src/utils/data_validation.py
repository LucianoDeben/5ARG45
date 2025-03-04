# utils/data_validation.py
from typing import Any, Dict


def validate_batch(
    batch: Dict[str, Any], required_keys: List[str] = ["viability"]
) -> None:
    if not isinstance(batch, dict):
        raise ValueError(f"Batch must be a dict, got {type(batch)}")
    for key in required_keys:
        if key not in batch:
            raise ValueError(f"Batch missing required key: {key}")
