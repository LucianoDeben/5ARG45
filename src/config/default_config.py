# config/default_config.py
from typing import Any, Dict

from config.constants import CURRENT_SCHEMA_VERSION, DEFAULT_PATHS, LOGGING_CONFIG


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
            # Transcriptomics encoder
            "transcriptomics_input_dim": 978,
            "transcriptomics_hidden_dims": [512, 256],
            "transcriptomics_output_dim": 128,
            # Chemical encoder
            "chemical_input_dim": 1025,
            "chemical_hidden_dims": [256, 128],
            "chemical_output_dim": 128,
            # Fusion module
            "fusion_output_dim": 256,
            "fusion_strategy": "cross_attention",
            "fusion_num_heads": 4,
            "fusion_dropout": 0.1,
            # Prediction head
            "predictor_hidden_dims": [128, 64],
            "predictor_dropout": 0.3,
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
            # Basic training parameters
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "loss": "huber",
            # Data splitting
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42,
            "group_by": "cell_mfc_name",
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
            "fingerprint_size": 1024,
            "radius": 2,
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
            "version": CURRENT_SCHEMA_VERSION,
            "save_checkpoints": True,
            "checkpoint_freq": 5,
            "keep_n_checkpoints": 3,
        },
        # Paths
        "paths": DEFAULT_PATHS,
        # Logging
        "logging": LOGGING_CONFIG,
        # Hardware
        "hardware": {
            "device": "cuda",
            "precision": "mixed",
            "num_workers": 4,
            "pin_memory": True,
            "benchmark_cudnn": True,
        },
    }

    return config
