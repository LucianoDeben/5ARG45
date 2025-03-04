# utils/loss.py
from typing import Callable, Dict

import torch.nn as nn


def create_criterion(config: Dict, section: str = "training") -> Callable:
    loss_type = config.get(section, {}).get("loss", "mse").lower()
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss: {loss_type}")
