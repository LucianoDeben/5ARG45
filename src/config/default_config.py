# config/default_config.py
from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    """
    Return the default configuration for the LINCS/CTRP drug response prediction project.

    Returns:
        Dict containing default hyperparameters, model architecture, and training settings.
    """
    config = {
        # Data settings
        "data": {
            "gctx_file": "data/processed/LINCS.gctx",
            "feature_space": "landmark",
            "nrows": None,
        },
        # Model settings
        "model": {
            "transcriptomics_input_dim": 978,
            "transcriptomics_hidden_dims": [512, 256],
            "transcriptomics_output_dim": 128,
            "chemical_input_dim": 1025,  # Updated from 256
            "chemical_hidden_dims": [256, 128],
            "chemical_output_dim": 128,
            "fusion_output_dim": 256,
            "fusion_strategy": "concat",
            "predictor_hidden_dims": [64, 32],
            "normalize": True,
            "dropout": 0.3,
            "activation": "relu",
            "use_batch_norm": True,
        },
        # Training settings
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42,
            "group_by": None,
        },
        # Chemical representation
        "chemical": {
            "representation": "fingerprint",
            "fingerprint_size": 1024,
            "radius": 2,  # ECFP4 equivalent
        },
        # Experiment tracking
        "experiment": {
            "project_name": "lincs_ctrp_prediction",
            "run_name": None,
            "track": True,  # Enable W&B tracking
        },
    }
    return config
