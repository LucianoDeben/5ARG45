import logging
from typing import Dict, List, Any

import numpy as np
from torch.utils.data import Dataset
from results import CVResults
from trainers import PyTorchTrainer, Trainer

class RunResultsAccumulator:
    """Accumulate results for each run and compute summaries."""
    
    def __init__(self):
        self.results = {}  # model_name: List[Dict[metric_name: value]]

    def add_result(self, model_name: str, run_number: int, metrics_dict: Dict):
        """Add results for a specific model and run."""
        if model_name not in self.results:
            self.results[model_name] = []
        self.results[model_name].append(metrics_dict)

    def get_summaries(self) -> Dict:
        """Compute mean and standard deviation for each metric across runs for each model."""
        summaries = {}
        for model_name, runs in self.results.items():
            mean = {}
            std = {}
            # Extract test metrics (keys starting with "test_")
            test_metric_keys = [key for key in runs[0].keys() if key.startswith("test_")]
            for metric_key in test_metric_keys:
                metric_name = metric_key.split("_")[1]  # Remove "test_" prefix
                values = [run[metric_key] for run in runs]
                mean[metric_name] = np.mean(values)
                std[metric_name] = np.std(values)
            summaries[model_name] = {"mean": mean, "std": std}
        return summaries

class ModelEvaluator:
    """Orchestrates training and evaluation across multiple runs for different models."""
    
    def __init__(self, n_runs: int = 5, random_state: int = 42):
        self.n_runs = n_runs
        self.random_state = random_state
        self.run_results_accumulator = RunResultsAccumulator()
        logging.info(f"Initialized ModelEvaluator with {n_runs} runs")

    def evaluate_model(self, trainer: Trainer, dataset, model_name: str, source: str = "gene", **kwargs):
        """Evaluate a single model across multiple runs."""
        for run in range(self.n_runs):
            current_rs = self.random_state + run if self.random_state is not None else None
            # Perform split for this run
            from splitters import train_val_test_split  # Replace with actual import
            X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
                dataset, source=source, random_state=current_rs, **kwargs
            )
            # Create datasets if necessary (for PyTorch)
            if isinstance(trainer, PyTorchTrainer):
                train_dataset = Dataset(X_train, y_train)
                val_dataset = Dataset(X_val, y_val)
                test_dataset = Dataset(X_test, y_test)
            else:
                # For SklearnTrainer, use the arrays directly
                pass

            # Train and evaluate
            if isinstance(trainer, PyTorchTrainer):
                train_metrics = trainer.train(train_dataset, val_dataset, **kwargs)
                test_metrics = trainer.evaluate(test_dataset, **kwargs)
            else:
                train_metrics = trainer.train(X_train, y_train, X_val, y_val, **kwargs)
                test_metrics = trainer.evaluate(X_test, y_test, **kwargs)

            # Combine metrics
            run_results = {**train_metrics, **test_metrics}
            self.run_results_accumulator.add_result(model_name, run, run_results)

    def get_cv_results(self) -> CVResults:
        """Get the CVResults object with summarized results."""
        summaries = self.run_results_accumulator.get_summaries()
        return CVResults(summaries)

    def evaluate_multiple_models(self, trainers: Dict[str, Trainer], dataset, **kwargs):
        """Evaluate multiple models and return the CVResults object."""
        for model_name, trainer in trainers.items():
            logging.info(f"Evaluating model: {model_name}")
            self.evaluate_model(trainer, dataset, model_name, **kwargs)
        return self.get_cv_results()