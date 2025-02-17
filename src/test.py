import logging
import torch
import torch.nn as nn
import torch.optim as optim
from data_sets import LINCSDataset
from utils import stratified_group_split
from metrics import get_regression_metrics
from cv_trainers import PyTorchNestedCVTrainer
from models import FlexibleFCNN  # your model definition

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    test_file = "../data/processed/LINCS.gctx"
    dataset = LINCSDataset(
        gctx_path=test_file,
        in_memory=True,
        normalize="z-score",
        feature_space="landmark"
    )
    
    nested_splits = stratified_group_split(dataset, n_outer_splits=5, n_inner_splits=4, random_state=42)
    
    def pytorch_model_builder():
        X_all, _ = dataset.to_numpy()
        input_dim = X_all.shape[1]
        return FlexibleFCNN(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128, 64],
            output_dim=1,
            activation_fn="relu",
            dropout_prob=0.2,
            residual=False,
            norm_type="batchnorm",
            weight_init="kaiming"
        )
    
    train_params = {
        "epochs": 10,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gradient_clipping": 5.0,
        "early_stopping_patience": 10,
        "use_mixed_precision": False,
        "criterion": nn.MSELoss(),
        "optimizer_fn": optim.Adam
    }
    
    trainer = PyTorchNestedCVTrainer(
        dataset=dataset,
        nested_splits=nested_splits,
        evaluation_fn=get_regression_metrics,
        model_builder=pytorch_model_builder,
        train_params=train_params,
        use_inner_cv=False
    )
    
    outer_metrics, (overall_mean, overall_std), inner_details = trainer.run()
    metric_names = overall_mean.keys()
    for name in metric_names:
        logging.info(f"PyTorch Overall {name}: mean = {overall_mean[name]:.4f}, std = {overall_std[name]:.4f}")

