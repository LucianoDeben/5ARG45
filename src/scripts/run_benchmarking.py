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

# Enable Tensor Core optimization for better performance on A100 GPUs
torch.set_float32_matmul_precision('medium')

from src.config.config_utils import setup_logging, get_default_config
from src.data.loaders import GCTXDataLoader
from src.data.datasets import DatasetFactory
from src.training.trainer import MultiRunTrainer
from src.models.base_module import DrugResponseModule

# Import models
from src.models.unimodal.flexible_fcnn import FlexibleFCNN
from src.models.unimodal.ridge_regression import RidgeRegression

def main(config_path=None, models_to_run=None):
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
    
    # Default models to run if not specified
    if not models_to_run:
        models_to_run = ["fcnn", "ridge"]
    
    # Create base output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(config["paths"]["results_dir"], f"benchmark_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    logger.info(f"Saving all results to: {base_output_dir}")
    
    # Load LINCS and MixSeq data
    logger.info("Loading datasets...")
    lincs_loader = GCTXDataLoader(config["data"]["lincs_file"])
    mixseq_loader = GCTXDataLoader(config["data"]["mixseq_file"])
    
    # Store results for all models
    all_results = []
    
    # Run each model type
    for model_type in models_to_run:
        logger.info(f"\n{'-'*70}\nTraining {model_type.upper()} model\n{'-'*70}")
        
        # Get the input dimension from config
        input_dim = config["model"]["transcriptomics_input_dim"]
        logger.info(f"Using input dimension from config: {input_dim}")
        
        # Create model based on model_type
        if model_type.lower() == "fcnn":
            model = FlexibleFCNN(
                input_dim=input_dim,
                hidden_dims=config["model"]["transcriptomics_hidden_dims"],
                output_dim=1,
                activation_fn=config["model"]["activation"],
                dropout_prob=config["model"]["dropout"],
                residual=config["model"].get("residual", False),
                norm_type="batchnorm" if config["model"]["use_batch_norm"] else "none",
                weight_init=config["model"].get("weight_init", "kaiming"),
            )
        elif model_type.lower() == "ridge":
            model = RidgeRegression(
                input_dim=input_dim,
                output_dim=1,
                alpha=config["model"].get("alpha", 1.0)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create Lightning module
        module_kwargs = {
            "model": model,
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"] if model_type.lower() != "ridge" else 0.0,
        }
        
        # Create timestamp and model-specific run name
        model_prefix = model_type.lower()
        run_name = f"{model_prefix}_{config['experiment']['run_name'].replace('${timestamp}', timestamp)}"
        if "${RUN_NAME:-" in run_name:
            run_name = f"{model_prefix}_run_{timestamp}"
        
        # Create datasets and dataloaders
        logger.info("Creating LINCS dataset for training...")
        train_ds, val_ds, test_ds = DatasetFactory.create_and_split_transcriptomics(
            gctx_loader=lincs_loader,
            feature_space=config["data"]["feature_space"],
            nrows=config["data"]["nrows"],
            test_size=config["training"]["test_size"],
            val_size=config["training"]["val_size"],
            random_state=config["training"]["random_state"],
            chunk_size=config["data"]["chunk_size"],
            group_by=config["training"]["group_by"],
            stratify_by=config["training"]["stratify_by"],
        )
        
        logger.info("Creating MixSeq dataset for external testing...")
        _, _, test_ds_mixseq = DatasetFactory.create_and_split_transcriptomics(
            gctx_loader=mixseq_loader,
            feature_space=config["data"]["feature_space"],
            nrows=config["data"]["nrows"],
            test_size=0.9,
            val_size=0.05,
            random_state=config["training"]["random_state"],
            chunk_size=config["data"]["chunk_size"],
            group_by=config["training"]["group_by"],
            stratify_by=config["training"]["stratify_by"],
        )
        
        # Create DataLoaders
        dataloader_kwargs = {
            "batch_size": config["training"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
            "shuffle": True,
        }
        train_loader = torch.utils.data.DataLoader(train_ds, **dataloader_kwargs)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
        mixseq_loader_dl = torch.utils.data.DataLoader(test_ds_mixseq, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
        
        # Set up MultiRunTrainer with pre-created dataloaders
        trainer = MultiRunTrainer(
            module_class=DrugResponseModule,
            module_kwargs=module_kwargs,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            num_runs=config["training"]["num_runs"],
            max_epochs=config["training"]["epochs"],
            patience=config["training"]["patience"],
            log_dir=base_output_dir,
            experiment_name=f"{config['experiment']['project_name']}_{run_name}",
            seed=config["training"]["random_state"],
        )
        
        # Run training
        logger.info(f"Starting multi-run training for {model_type.upper()} model...")
        results = trainer.train()
        
        # Evaluate on external test set (MixSeq)
        logger.info("Evaluating on external test set (MixSeq)...")
        external_results = []
        for model_idx, best_model in enumerate(trainer.best_models):
            # Create Lightning trainer for evaluation
            import pytorch_lightning as pl
            eval_trainer = pl.Trainer(logger=False)
            mixseq_result = eval_trainer.test(best_model, dataloaders=mixseq_loader_dl)[0]
            external_results.append(mixseq_result)
        
        # Aggregate external results
        mixseq_metrics = {}
        for result in external_results:
            for key, value in result.items():
                if key not in mixseq_metrics:
                    mixseq_metrics[key] = []
                mixseq_metrics[key].append(value)
        
        mixseq_mean = {f"mixseq_{k}": np.mean(v) for k, v in mixseq_metrics.items()}
        mixseq_std = {f"mixseq_{k}": np.std(v) for k, v in mixseq_metrics.items()}
        
        # Add MixSeq results to the overall results
        results["mean"].update(mixseq_mean)
        results["std"].update(mixseq_std)
        
        # Print model results
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
        
        logger.info(f"\nResults saved to {trainer.log_dir}/{trainer.experiment_name}")
        
        # Store results for comparison
        all_results.append({
            "model_type": model_type,
            "results": results,
            "experiment_dir": os.path.join(trainer.log_dir, trainer.experiment_name)
        })
    
    # Generate model comparison if we have multiple models
    if len(all_results) > 1:
        logger.info("\nGenerating model comparison...")
        comparison_dir = os.path.join(base_output_dir, "model_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Create comparison DataFrame
        comparison_data = []
        all_metrics = set()
        
        # Collect all metrics
        for model_result in all_results:
            for key in model_result["results"]["mean"].keys():
                all_metrics.add(key)
        
        # Create comparison data
        for model_result in all_results:
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
        
        # Generate comparison visualizations
        test_metrics = {m for m in all_metrics if m.startswith("test_")}
        mixseq_metrics = {m for m in all_metrics if m.startswith("mixseq_")}
        metric_bases = {metric.split('_')[1] for metric in test_metrics}
        
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
        
        # Print comparison summary
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
    
    logger.info("\nBenchmarking complete!")
    logger.info(f"All results saved to {base_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking experiments")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--models", type=str, nargs="+", choices=["fcnn", "ridge"], 
                        default=["fcnn", "ridge"], help="Models to benchmark")
    args = parser.parse_args()
    
    main(args.config, args.models)