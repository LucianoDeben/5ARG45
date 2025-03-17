# scripts/run_benchmarking.py
import os
import argparse
import yaml
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

torch.set_float32_matmul_precision('medium')

from src.config.config_utils import setup_logging, get_default_config
from src.data.loaders import GCTXLoader
from src.data.datasets import DatasetFactory
from src.training.trainer import MultiRunTrainer
from src.models.base_module import DrugResponseModule  # Updated to use base module

# Import models
from src.models.unimodal.flexible_fcnn import FlexibleFCNN
from src.models.unimodal.ridge_regression import RidgeRegression

logger = logging.getLogger(__name__)

def train_model(model_type, config, lincs_loader, mixseq_loader, base_output_dir):
    """Run training for a specific model type and return results."""
    
    # Get the input dimension and feature names from config or data
    input_dim = config["model"]["transcriptomics_input_dim"]
    #TODO: Allow for feature extraction by adding it to the datasets itself!
    feature_names = config.get("data", {}).get("feature_names", [f"feat_{i}" for i in range(input_dim)])
    logger.info(f"Creating {model_type.upper()} model with input dimension: {input_dim}")

    # Create model based on model_type
    if model_type.lower() == "fcnn":
        model = FlexibleFCNN(
            input_dim=input_dim,
            hidden_dims=config["model"]["transcriptomics_hidden_dims"] + [config["model"]["transcriptomics_output_dim"]],
            output_dim=1,
            activation_fn=config["model"]["activation"],
            dropout_prob=config["model"]["dropout"],
            residual=False,
            norm_type="batchnorm" if config["model"]["use_batch_norm"] else "none",
            weight_init="kaiming",
            feature_names=feature_names,
        )
    elif model_type.lower() == "ridge":
        model = RidgeRegression(
            input_dim=input_dim,
            output_dim=1,
            alpha=1.0,  
            feature_names=feature_names,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create Lightning module with feature names
    module_kwargs = {
        "model": model,
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"] if model_type.lower() != "ridge" else 0.0,
        "grad_clip": config["training"]["max_grad_norm"] if config["training"]["clip_grad_norm"] else None,
    }
    
    # Create timestamp and run name for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_prefix = model_type.lower()
    run_name = f"{model_prefix}_{config['experiment']['run_name'].replace('${timestamp}', timestamp)}"
    if "${RUN_NAME:-" in run_name:
        run_name = f"{model_prefix}_run_{timestamp}"
    
    logger.info("Creating MixSeq dataset for external testing...")
    _, _, test_ds_mixseq = DatasetFactory.create_and_split_datasets(
        data_loader=mixseq_loader,
        dataset_type="transcriptomics",
        feature_space=config["data"]["feature_space"],
        nrows=config["data"]["nrows"],
        test_size=0.9,
        val_size=0.05,
        random_state=config["data"]["random_seed"],
        chunk_size=config["data"]["chunk_size"],
        group_by=config["training"]["group_by"],
        stratify_by=config["training"]["stratify_by"],
    )
    
    # Create MixSeq DataLoader
    mixseq_loader_dl = torch.utils.data.DataLoader(test_ds_mixseq, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
    
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
    
    # Set up MultiRunTrainer with pre-created dataloaders
    trainer = MultiRunTrainer(
        module_class=DrugResponseModule,  # Updated to use base module
        module_kwargs=module_kwargs,
        gctx_loader=lincs_loader,
        dataset_kwargs=dataset_kwargs,
        num_runs=config["training"]["num_runs"],
        max_epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        output_dir=base_output_dir,
        experiment_name=f"{config['experiment']['project_name']}_{run_name}",
        base_seed=config["data"]["random_seed"],
        gpu_if_available=(config["inference"]["device"] == "cuda"),
        external_test_loaders={"mixseq": mixseq_loader_dl},
        visualizations_to_generate=["predictions", "boxplot", "feature_importance", "prediction_intervals"],  # Selective visualizations
    )
    
    # Run training
    logger.info(f"Starting multi-run training for {model_type.upper()} model...")
    results = trainer.train()
    
    # Log detailed results
    logger.info("\n" + "="*50)
    logger.info(f"TRAINING COMPLETE FOR {model_type.upper()}")
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
    
    return {
        "model_type": model_type,
        "results": results,
        "experiment_dir": trainer.experiment_dir,
    }

def compare_models(model_results, output_dir):
    """Generate comparison visualizations and reports for multiple models."""
    logger.info("Generating model comparison...")
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create DataFrame for comparison
    comparison_data = []
    all_metrics = set()
    
    # Collect all available metrics
    for model_result in model_results:
        for key in model_result["results"]["mean"].keys():
            all_metrics.add(key)
    logger.info(f"Available metrics: {sorted(list(all_metrics))}")
    
    # Separate test and external metrics
    test_metrics = {m for m in all_metrics if m.startswith("test_")}
    mixseq_metrics = {m for m in all_metrics if m.startswith("mixseq_")}
    
    # Create comparison data
    for model_result in model_results:
        model_type = model_result["model_type"]
        results = model_result["results"]
        
        model_data = {"Model": model_type.upper()}
        
        # Add mean and std metrics
        for metric in all_metrics:
            if metric in results["mean"]:
                model_data[f"{metric}_mean"] = results["mean"][metric]
            if metric in results["std"]:
                model_data[f"{metric}_std"] = results["std"][metric]
        
        comparison_data.append(model_data)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(comparison_dir, "model_comparison.csv"), index=False)
    logger.info(f"Saved model comparison to {os.path.join(comparison_dir, 'model_comparison.csv')}")
    
    # Extract metric bases
    metric_bases = {metric.split('_')[1] for metric in test_metrics}
    
    # Generate comparison visualizations
    for metric_base in metric_bases:
        test_metric_mean = f"test_{metric_base}_mean"
        test_metric_std = f"test_{metric_base}_std"
        mixseq_metric_mean = f"mixseq_{metric_base}_mean"
        mixseq_metric_std = f"mixseq_{metric_base}_std"
        
        if test_metric_mean in comparison_df.columns and mixseq_metric_mean in comparison_df.columns:
            plt.figure(figsize=(12, 6))
            models = comparison_df["Model"].tolist()
            x = np.arange(len(models))
            width = 0.35
            
            plt.bar(x - width/2, comparison_df[test_metric_mean], width, label='LINCS Test',
                    yerr=comparison_df[test_metric_std], alpha=0.7, capsize=5)
            plt.bar(x + width/2, comparison_df[mixseq_metric_mean], width, label='MixSeq',
                    yerr=comparison_df[mixseq_metric_std], alpha=0.7, capsize=5)
            
            plt.xlabel('Model')
            plt.ylabel(metric_base.upper())
            plt.title(f'Comparison of {metric_base.upper()} across Models')
            plt.xticks(x, models)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"comparison_{metric_base}.png"))
            plt.close()
        elif test_metric_mean in comparison_df.columns:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(comparison_df)), comparison_df[test_metric_mean],
                    yerr=comparison_df[test_metric_std], alpha=0.7, capsize=5)
            plt.xlabel('Model')
            plt.ylabel(metric_base.upper())
            plt.title(f'{metric_base.upper()} on LINCS Test Set')
            plt.xticks(range(len(comparison_df)), comparison_df["Model"])
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"comparison_{metric_base}_test_only.png"))
            plt.close()
    
    # Log comparison summary
    logger.info("\n" + "="*50)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*50)
    
    if test_metrics:
        logger.info("\nTest Set Performance:")
        for metric in sorted(test_metrics):
            metric_base = metric.split('_')[1]
            logger.info(f"\n{metric_base.upper()}:")
            for _, row in comparison_df.iterrows():
                logger.info(f"  {row['Model']}: {row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}")
    
    if mixseq_metrics:
        logger.info("\nMixSeq Performance:")
        for metric in sorted(mixseq_metrics):
            metric_base = metric.split('_')[1]
            logger.info(f"\n{metric_base.upper()}:")
            for _, row in comparison_df.iterrows():
                logger.info(f"  {row['Model']}: {row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}")
    
    logger.info(f"\nComparison saved to {comparison_dir}")
    return comparison_df

def main(config_path=None, models_to_run=None):
    """Run benchmarking experiments for specified models."""
    # logger = setup_logging()
    
    # Load configuration
    config = yaml.safe_load(open(config_path, 'r')) if config_path and os.path.exists(config_path) else get_default_config()
    models_to_run = models_to_run or ["fcnn", "ridge"]
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(config["paths"]["results_dir"], f"benchmark_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading LINCS and MixSeq data...")
    lincs_loader = GCTXLoader(config["data"]["lincs_file"])
    mixseq_loader = GCTXLoader(config["data"]["mixseq_file"])
    
    # Run training for each model
    all_results = []
    for model_type in models_to_run:
        logger.info(f"\n{'-'*70}\nBenchmarking {model_type.upper()} model\n{'-'*70}")
        model_results = train_model(model_type, config, lincs_loader, mixseq_loader, base_output_dir)
        all_results.append(model_results)
    
    # Compare models
    comparison = compare_models(all_results, base_output_dir)
    
    logger.info("\nBenchmarking complete!")
    logger.info(f"All results saved to {base_output_dir}")
    return all_results, comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking experiments")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--models", type=str, nargs="+", choices=["fcnn", "ridge"], 
                        default=["fcnn", "ridge"], help="Models to benchmark")
    args = parser.parse_args()
    main(args.config, args.models)