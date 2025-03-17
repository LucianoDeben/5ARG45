# scripts/train_multimodal_model.py
import os
import argparse
import yaml
import logging
import torch
from datetime import datetime

# Enable Tensor Core optimization for better performance on NVIDIA GPUs
torch.set_float32_matmul_precision('medium')

# Import custom modules
from src.config.config_utils import setup_logging, load_config
from src.data.loaders import GCTXLoader
from src.data.datasets import DatasetFactory

# Import model components
from src.models.transcriptomics.encoders import TranscriptomicEncoder
from src.models.molecular.transformers import SMILESTransformerEncoder
from src.models.integration.fusion import FeatureFusion
from src.models.integration.attention import CrossModalAttention
from src.models.prediction.viability_prediction import ViabilityPredictor
from src.models.multimodal_models import MultimodalDrugResponseModule

# Import training utilities
from src.training.trainer import MultiRunTrainer

# Setup logging
logger = logging.getLogger(__name__)


def create_molecular_encoder(config):
    """Create molecular encoder based on config settings."""
    mol_type = config["model"]["molecular_encoder_type"]
    
    if mol_type == "transformer":
        return SMILESTransformerEncoder(
            output_dim=config["model"]["molecular_output_dim"],
            embedding_dim=config["model"].get("molecular_embedding_dim", 256),
            num_layers=config["model"].get("molecular_num_layers", 6),
            num_heads=config["model"].get("molecular_num_heads", 8),
            dropout=config["model"]["dropout"],
        )
    elif mol_type == "descriptors" or mol_type == "fingerprint":
        # Import late to avoid circular imports
        from src.models.molecular.smiles_processing import MolecularMLP
        
        return MolecularMLP(
            input_dim=config["model"]["molecular_input_dim"],
            hidden_dims=config["model"]["molecular_hidden_dims"],
            output_dim=config["model"]["molecular_output_dim"],
            dropout=config["model"]["dropout"],
            activation=config["model"]["activation"],
            normalize=config["model"]["use_batch_norm"],
        )
    else:
        raise ValueError(f"Unsupported molecular encoder type: {mol_type}")


def create_fusion_module(config, transcriptomics_dim, molecular_dim):
    """Create fusion module based on config settings."""
    fusion_type = config["model"]["fusion_type"]
    fusion_strategy = config["model"]["fusion_strategy"]
    fusion_output_dim = config["model"]["fusion_output_dim"]
    
    if fusion_type == "attention":
        return CrossModalAttention(
            transcriptomics_dim=transcriptomics_dim,
            chemical_dim=molecular_dim,
            hidden_dim=fusion_output_dim,
            num_heads=config["model"].get("fusion_num_heads", 4),
            dropout=config["model"]["dropout"],
        )
    else:  # simple fusion
        return FeatureFusion(
            t_dim=transcriptomics_dim,
            c_dim=molecular_dim,
            output_dim=fusion_output_dim,
            strategy=fusion_strategy,
            projection=True,
            dropout=config["model"]["dropout"],
        )


def create_multimodal_model(config):
    """Create multimodal model based on config settings."""
    # Create transcriptomics encoder
    transcriptomics_encoder = TranscriptomicEncoder(
        input_dim=config["model"]["transcriptomics_input_dim"],
        hidden_dims=config["model"]["transcriptomics_hidden_dims"],
        output_dim=config["model"]["transcriptomics_output_dim"],
        normalize=config["model"]["use_batch_norm"],
        dropout=config["model"]["dropout"],
        activation=config["model"]["activation"],
        residual=config["model"].get("use_residual", False),
    )
    
    # Create molecular encoder
    molecular_encoder = create_molecular_encoder(config)
    
    # Create fusion module
    fusion_module = create_fusion_module(
        config, 
        config["model"]["transcriptomics_output_dim"],
        config["model"]["molecular_output_dim"]
    )
    
    # Create prediction head
    prediction_head = ViabilityPredictor(
        input_dim=config["model"]["fusion_output_dim"],
        hidden_dims=config["model"]["predictor_hidden_dims"],
        dropout=config["model"]["dropout"],
        activation=config["model"]["activation"],
        use_batch_norm=config["model"]["use_batch_norm"],
        output_activation="sigmoid",
        uncertainty=config["model"].get("use_uncertainty", False),
    )
    
    # Create multimodal model
    model = MultimodalDrugResponseModule(
        transcriptomics_encoder=transcriptomics_encoder,
        molecular_encoder=molecular_encoder,
        fusion_module=fusion_module,
        prediction_head=prediction_head,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-5),
        grad_clip=config["training"]["max_grad_norm"] if config["training"]["clip_grad_norm"] else None,
    )
    
    return model


def train_multimodal_model(config_path=None):
    """Train a multimodal drug response prediction model."""
    # Load configuration
    config = load_config(config_path)
    
    # Create timestamp for experiment naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config["experiment"]["run_name"].replace("${timestamp}", timestamp)
    if "${RUN_NAME:-" in run_name:
        run_name = f"multimodal_run_{timestamp}"
    
    # Create output directory
    base_output_dir = os.path.join(config["paths"]["results_dir"], run_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(base_output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Training multimodal model with config: {config_path}")
    logger.info(f"Results will be saved to: {base_output_dir}")
    
    # Load data
    logger.info("Loading LINCS and MixSeq data...")
    lincs_loader = GCTXLoader(config["data"]["lincs_file"])
    mixseq_loader = GCTXLoader(config["data"]["mixseq_file"])
    
    # Create model
    model = create_multimodal_model(config)
    
    # Create MixSeq dataset for external testing
    logger.info("Creating MixSeq dataset for external testing...")
    _, _, test_ds_mixseq = DatasetFactory.create_and_split_datasets(
        gctx_loader=mixseq_loader,
        dataset_type="multimodal",
        feature_space=config["data"]["feature_space"],
        nrows=config["data"]["nrows"],
        test_size=0.9,
        val_size=0.05,
        random_state=config["training"]["random_state"],
        chunk_size=config["data"]["chunk_size"],
    )
    
    # Create MixSeq DataLoader
    mixseq_loader_dl = torch.utils.data.DataLoader(
        test_ds_mixseq, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=config["data"]["num_workers"]
    )
    
    # Prepare dataset parameters
    dataset_kwargs = {
        "feature_space": config["data"]["feature_space"],
        "dataset_type": "multimodal",
        "nrows": config["data"]["nrows"],
        "test_size": config["training"]["test_size"],
        "val_size": config["training"]["val_size"],
        "random_state": config["training"]["random_state"],
        "chunk_size": config["data"]["chunk_size"],
        "group_by": config["training"]["group_by"],
        "stratify_by": config["training"]["stratify_by"],
    }
    
    # Set up MultiRunTrainer
    trainer = MultiRunTrainer(
        module_class=MultimodalDrugResponseModule,
        module_kwargs={
            "transcriptomics_encoder": model.transcriptomics_encoder,
            "molecular_encoder": model.molecular_encoder,
            "fusion_module": model.fusion_module,
            "prediction_head": model.prediction_head,
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"].get("weight_decay", 1e-5),
            "grad_clip": config["training"]["max_grad_norm"] if config["training"]["clip_grad_norm"] else None,
        },
        gctx_loader=lincs_loader,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs={
            "batch_size": config["training"]["batch_size"],
            "shuffle": True,
            "num_workers": config["data"]["num_workers"],
        },
        num_runs=config["training"]["num_runs"],
        max_epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        output_dir=base_output_dir,
        experiment_name=f"{config['experiment']['project_name']}_{run_name}",
        base_seed=config["training"]["random_state"],
        gpu_if_available=(config["inference"]["device"] == "cuda"),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        external_test_loaders={"mixseq": mixseq_loader_dl},
        visualizations_to_generate=[
            "predictions", "learning_curves", "residual", "error_distribution", 
            "boxplot", "violinplot", "calibration"
        ],
    )
    
    # Run training
    logger.info("Starting multi-run training...")
    results = trainer.train()
    
    # Log summary results
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)
    logger.info("\nAggregated Test Metrics:")
    for metric, value in results["mean"].items():
        if metric.startswith("test_"):
            logger.info(f"{metric.replace('test_', '')}: {value:.4f} ± {results['std'][metric]:.4f}")
    
    logger.info("\nExternal Test Metrics (MixSeq):")
    for metric, value in results["mean"].items():
        if metric.startswith("mixseq_"):
            logger.info(f"{metric.replace('mixseq_', '')}: {value:.4f} ± {results['std'][metric]:.4f}")
    
    logger.info(f"\nResults saved to {trainer.experiment_dir}")
    
    return results, trainer.experiment_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal drug response prediction model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Train model
    train_multimodal_model(args.config)