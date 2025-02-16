# --------------------- Usage Example for scikit-learn ---------------------
import logging
from data_sets import LINCSDataset, stratified_group_split
from evaluation import get_regression_metrics
from training import NestedCVTrainer


if __name__ == "__main__":
    # Assume LINCSDataset and nested_stratified_group_split are defined and imported.
    test_file = "../data/processed/LINCS.gctx"
    dataset = LINCSDataset(
        gctx_path=test_file,
        in_memory=True,  # or False for lazy loading
        normalize="z-score",
        feature_space="landmark"  # or e.g., ["landmark", "best inferred"]
    )
    
    nested_splits = stratified_group_split(
        dataset, n_outer_splits=5, n_inner_splits=4, random_state=42
    )
    
    # For a scikit-learn model (e.g., LinearRegression)
    from sklearn.linear_model import LinearRegression
    
    def sklearn_model_builder():
        return LinearRegression()
    
    trainer = NestedCVTrainer(
        dataset=dataset,
        nested_splits=nested_splits,
        model_builder=sklearn_model_builder,
        model_type="sklearn",
        evaluation_fn=get_regression_metrics
    )
    
    outer_metrics, (overall_mean, overall_std), inner_details = trainer.run()
    
    metric_names = ["MSE", "MAE", "R²", "Pearson"]
    for i, name in enumerate(metric_names):
        logging.info(f"Overall {name}: mean = {overall_mean[i]:.4f}, std = {overall_std[i]:.4f}")