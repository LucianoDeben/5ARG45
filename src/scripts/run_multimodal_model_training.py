# run_multimodal_model_training.py
import os
import yaml
import argparse
import torch
import logging
import numpy as np
import pytorch_lightning as pl

# Import custom modules
from src.config.config_utils import get_default_config
from src.data.loaders import GCTXLoader
from src.data.datasets import DatasetFactory
from src.models.transcriptomics.encoders import create_transcriptomic_encoder
from src.models.molecular.encoders import create_molecular_encoder
from src.models.integration.fusers import create_feature_fusion
from src.models.prediction.predictors import create_viability_predictor
from src.models.multimodal_models import MultimodalModel, MultimodalDrugResponseModule
from src.training.trainer import MultiRunTrainer

logger = logging.getLogger(__name__)

def main(config_path=None, **kwargs):
    """
    Main training script for multimodal drug response prediction.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments to override config
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
    
    # Override config with any provided kwargs
    for key, value in kwargs.items():
        # Assuming nested keys are dot-separated
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            current = current[k]
        current[keys[-1]] = value
    
    # Set random seed for reproducibility
    pl.seed_everything(config['data']['random_seed'])
    
    # Load data
    logger.info("Loading LINCS and MixSeq data...")
    lincs_loader = GCTXLoader(config['data']['lincs_file'])
    mixseq_loader = GCTXLoader(config['data']['mixseq_file'])
    
    # Create model components
    transcriptomics_encoder = create_transcriptomic_encoder(
        encoder_type=config['model']['transcriptomics_encoder_type'],
        input_dim=config['model']['transcriptomics_input_dim'],
        hidden_dims=config['model']['transcriptomics_hidden_dims'],
        output_dim=config['model']['transcriptomics_output_dim'],
        activation=config['model']['activation'],
        dropout=config['model']['dropout'],
        normalize=config['model']['normalize']
    )
    
    molecular_encoder = create_molecular_encoder(
        encoder_type=config['model']['molecular_encoder_type'],
        input_dim=config['model']['molecular_input_dim'],
        hidden_dims=config['model']['molecular_hidden_dims'],
        output_dim=config['model']['molecular_output_dim'],
        activation=config['model']['activation'],
        dropout=config['model']['dropout']
    )
    
    fusion_module = create_feature_fusion(
        fusion_type=config['model']['fusion_type'],
        modality_dims=(
            config['model']['transcriptomics_output_dim'], 
            config['model']['molecular_output_dim']
        ),
        output_dim=config['model']['fusion_output_dim'],
        strategy=config['model']['fusion_strategy']
    )
    
    prediction_head = create_viability_predictor(
        predictor_type=config['model']['predictor_type'],
        input_dim=config['model']['fusion_output_dim'],
        hidden_dims=config['model']['predictor_hidden_dims']
    )
    
    # Create complete multimodal model
    multimodal_model = MultimodalModel(
        transcriptomics_encoder=transcriptomics_encoder,
        molecular_encoder=molecular_encoder,
        fusion_module=fusion_module,
        prediction_head=prediction_head
    )
    
    dataset_kwargs = {
        "feature_space": config["data"]["feature_space"],
        "dataset_type": "transcriptomics",
        "nrows": config["data"]["nrows"],
        "test_size": config["training"]["test_size"],
        "val_size": config["training"]["val_size"],
        "chunk_size": config["data"]["chunk_size"],
        "group_by": config["training"]["group_by"],
        "stratify_by": config["training"]["stratify_by"],
    } 
    
    
    # Setup trainer
    trainer = MultiRunTrainer(
        module_class=MultimodalDrugResponseModule,
        module_kwargs={
            'model': multimodal_model,
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay']
        },
        dataset_kwargs=dataset_kwargs,
        gctx_loader=lincs_loader,
        num_runs=config['training']['num_runs'],
        max_epochs=config['training']['epochs'],
        output_dir=config['paths']['model_dir'],
        experiment_name=config['experiment']['project_name']
    )
    
    # Run training
    results = trainer.train()
    
    # Log results
    logger.info("Training complete. Results summary:")
    for metric, value in results['mean'].items():
        logger.info(f"{metric}: {value}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Drug Response Prediction")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    main(args.config)