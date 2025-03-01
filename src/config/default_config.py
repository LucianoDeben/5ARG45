"""Default configuration for the LINCS/CTRP drug response prediction project."""

from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration settings.

    Returns:
        Dictionary containing default configuration
    """
    config = {
        # Data settings
        "data": {
            "gctx_file": "../data/processed/LINCS.gctx",
            "feature_space": "landmark",
            "nrows": None,
            "normalize": "zscore",
            "random_seed": 42,
            "cache_data": True,
            "use_multiprocessing": True,
            "num_workers": 4,
        },
        # Model architecture
        "model": {
            # Transcriptomics encoder (978 landmark genes)
            "transcriptomics_input_dim": 978,
            "transcriptomics_hidden_dims": [512, 256],
            "transcriptomics_output_dim": 128,
            # Chemical encoder (1024 bit fingerprint + 1 dosage feature)
            "chemical_input_dim": 1025,
            "chemical_hidden_dims": [256, 128],
            "chemical_output_dim": 128,
            # Fusion module
            "fusion_output_dim": 256,
            "fusion_strategy": "concat",
            "predictor_hidden_dims": [128, 64],
            # General settings
            "normalize": True,
            "dropout": 0.3,
            "activation": "gelu",
            "use_batch_norm": True,
            "layer_norm": True,
            "residual_connections": True,
        },
        # Training settings
        "training": {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "loss": "mse",
            # Data splitting
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42,
            # Learning rate scheduling
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1e-6,
            # Early stopping
            "early_stopping": True,
            "patience": 10,
            "min_delta": 0.001,
            # Gradient clipping
            "clip_grad_norm": True,
            "max_grad_norm": 1.0,
            # Mixed precision training
            "use_amp": True,
            # Regularization
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
        },
        # Chemical representation
        "chemical": {
            "representation": "fingerprint",
            "fingerprint_size": 1024,  # Standard ECFP size
            "radius": 2,  # ECFP4
            "use_chirality": True,
            "use_features": True,
            "sanitize": True,
        },
        # Experiment tracking
        "experiment": {
            "project_name": "lincs_ctrp_prediction",
            "run_name": None,
            "track": True,
            "tags": ["multimodal", "drug-response"],
            "version": "1.0.0",
            "save_checkpoints": True,
            "checkpoint_freq": 5,
            "keep_n_checkpoints": 3,
        },
    }

    return config
