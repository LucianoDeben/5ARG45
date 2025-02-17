import logging
import numpy as np
from data_sets import LINCSDataset
from utils import nested_stratified_group_split_fixed  # Fixed nested splits (single fold assignment)
from metrics import get_regression_metrics
from cv_trainers import SklearnNestedCVTrainer
from results import CVResults  # The updated CVResults class
from sklearn.linear_model import LinearRegression, Ridge, Lasso

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Define feature space options.
    # "all" means all allowed gene types (i.e. landmark, best inferred, inferred)
    feature_space_options = ["landmark", ["landmark", "best inferred"], "all"]
    
    # Define model builders for scikit-learn models.
    model_builders = {
        "LinearRegression": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=0.5),
        "Lasso": lambda: Lasso(alpha=0.1)
    }
    
    # Create a common set of nested splits using the full dataset's row metadata.
    # (Assuming row_metadata is independent of feature_space selection.)
    # We'll use the dataset built with the default feature space (say, "landmark") for splitting.
    base_dataset = LINCSDataset(
        gctx_path="../data/processed/LINCS.gctx",
        in_memory=True,
        normalize="z-score",
        feature_space="landmark"
    )
    nested_splits = nested_stratified_group_split_fixed(base_dataset, n_folds=5, random_state=42)
    
    results = {}
    for feature_space in feature_space_options:
        # Create a label for the feature space.
        if isinstance(feature_space, list):
            feat_label = ",".join(feature_space)
        else:
            feat_label = feature_space
        for model_name, builder in model_builders.items():
            combined_key = (feat_label, model_name)
            logging.info(f"Evaluating {combined_key}")
            # Reinitialize dataset with the given feature_space.
            dataset = LINCSDataset(
                gctx_path="../data/processed/LINCS.gctx",
                in_memory=True,
                normalize="z-score",
                feature_space=feature_space
            )
            # Use the same nested_splits computed above (since row metadata is unchanged)
            trainer = SklearnNestedCVTrainer(
                dataset=dataset,
                nested_splits=nested_splits,
                evaluation_fn=get_regression_metrics,
                model_builder=builder,
                use_inner_cv=False
            )
            outer_metrics, (overall_mean, overall_std), _ = trainer.run()
            results[combined_key] = {"mean": overall_mean, "std": overall_std}
    
    cv_results = CVResults(results)
    print(cv_results.get_results_df())
    cv_results.plot_metric("MSE")
