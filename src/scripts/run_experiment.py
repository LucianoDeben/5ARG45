# scripts/run_experiment.py
import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.loaders import GCTXDataLoader
from src.data.datasets import DatasetFactory
from src.models.transcriptomics.encoders import TranscriptomicsEncoderFactory
from src.models.chemical.encoders import ChemicalEncoderFactory
from src.models.integration.fusion import FusionModuleFactory
from src.models.prediction.heads import PredictionHeadFactory
from src.models.multimodal_module import MultimodalDrugResponseModule
from src.training.trainer import MultiRunTrainer
from src.evaluation.evaluator import DrugResponseEvaluator
from src.utils.logging import setup_logging

def main(config_path: str):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(config["logging"])
    
    # Load data
    gctx_loader = GCTXDataLoader(
        gctx_path=config["data"]["gctx_path"],
        meta_path=config["data"]["meta_path"]
    )
    
    # Create datasets
    train_ds, val_ds, test_ds = DatasetFactory.create_and_split_multimodal(
        gctx_loader=gctx_loader,
        **config["dataset"]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["evaluation"]["num_workers"]
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["evaluation"]["num_workers"]
    )
    
    # Create model components
    transcriptomics_encoder = TranscriptomicsEncoderFactory.create(
        **config["model"]["transcriptomics_encoder"]
    )
    
    chemical_encoder = ChemicalEncoderFactory.create(
        **config["model"]["chemical_encoder"]
    )
    
    fusion_module = FusionModuleFactory.create(
        **config["model"]["fusion_module"]
    )
    
    prediction_head = PredictionHeadFactory.create(
        **config["model"]["prediction_head"]
    )
    
    # Create module kwargs
    module_kwargs = {
        "transcriptomics_encoder": transcriptomics_encoder,
        "molecular_encoder": chemical_encoder,
        "fusion_module": fusion_module,
        "prediction_head": prediction_head,
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
    }
    
    # Create and run trainer
    trainer = MultiRunTrainer(
        module_class=MultimodalDrugResponseModule,
        module_kwargs=module_kwargs,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        num_runs=config["experiment"]["num_runs"],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        log_dir=config["experiment"]["log_dir"],
        experiment_name=config["experiment"]["name"],
        seed=config["experiment"]["seed"],
    )
    
    results = trainer.train()
    
    # Run detailed evaluation on the best model
    best_model_idx = results.get("test_loss_mean", float("inf"))
    best_model = trainer.best_models[best_model_idx]
    
    evaluator = DrugResponseEvaluator(
        model=best_model,
        dataloader=test_loader,
        output_dir=os.path.join(config["experiment"]["log_dir"], config["experiment"]["name"], "evaluation")
    )
    
    detailed_metrics = evaluator.evaluate()
    
    # Print final results
    print("\nExperiment Results Summary:")
    print(f"Number of runs: {config['experiment']['num_runs']}")
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nDetailed Evaluation Metrics (Best Model):")
    for metric, value in detailed_metrics.items():
        print(f"{metric}: {value:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drug response prediction experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    main(args.config)