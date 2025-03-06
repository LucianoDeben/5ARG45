# %% Import necessary libraries and modules
from datetime import datetime
from src.config.config_utils import init_wandb, load_config, setup_logging
from src.data.datasets import DatasetFactory
from src.data.feature_transforms import create_feature_transform
from src.data.loaders import GCTXDataLoader
from src.data.preprocessing import LINCSCTRPDataProcessor
from src.training.trainer import MultiRunTrainer
from src.utils.experiment_tracker import ExperimentTracker
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# %% Setup logging
logger = setup_logging()
logger.info("Starting multimodal drug response prediction workflow...")

# %% Load config
try:
    config = load_config("config.yaml")
except Exception as e:
    logger.error(f"Failed to load config: {e}")

# %% Setup experiment logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = f"multimodal_drug_response_{timestamp}"

try:
    exp_logger = ExperimentTracker(
        experiment_name=exp_name,
        config=config,
        log_dir=config["paths"]["log_dir"],
        use_tensorboard=False,
        use_wandb= False,
        wandb_project=config["experiment"]["project_name"],
    )
    if config["experiment"]["track"]:
        init_wandb(config)
        logger.info("WandB experiment tracking enabled.")
except Exception as e:
    logger.error(f"Error setting up experiment logging: {e}")
# %% Prepare Dataloaders   
# Step 1: Load data using GCTXDataLoader
gctx_loader = GCTXDataLoader(config["data"]["gctx_file"])

# Step 2: Preprocess data using LINCSCTRPDataProcessor
processor = LINCSCTRPDataProcessor(
    gctx_file=config["data"]["gctx_file"],
    feature_space=config["data"]["feature_space"],
    nrows=config["data"].get("nrows"),
    imputation_strategy=config["data"].get("imputation_strategy", "mean"),
    handle_outliers=config["data"].get("handle_outliers", False),
    outlier_threshold=config["data"].get("outlier_threshold", 3.0),
)
transcriptomics, metadata = processor.preprocess()

# Step 3: Create feature transform for molecular data
transform_molecular = create_feature_transform(
    config["molecular"].get("transform_molecular", "fingerprint"),
    fingerprint_size=config["molecular"].get("fingerprint_size", 1024),
    fingerprint_radius=config["molecular"].get("fingerprint_radius", 2),
)

# Step 4: Create and split datasets using DatasetFactory
train_ds, val_ds, test_ds = DatasetFactory.create_and_split_transcriptomics(
    gctx_loader=gctx_loader,
    feature_space=config["data"]["feature_space"],
    nrows=config["data"].get("nrows"),
    test_size=config["training"]["test_size"],
    val_size=config["training"]["val_size"],
    random_state=config["data"]["random_seed"],
    group_by=config["training"]["group_by"],
    stratify_by=config["training"]["stratify_by"],
    transform=None,
    chunk_size=config["data"].get("chunk_size"),
)

# Log the dataset sizes
logger.info(f"Training dataset size: {len(train_ds)}")
logger.info(f"Validation dataset size: {len(val_ds)}") 
logger.info(f"Test dataset size: {len(test_ds)}")

# Step 5: Create PyTorch DataLoaders
train_loader = DataLoader(
    train_ds,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=config["data"].get("num_workers", 0),
)
val_loader = DataLoader(
    val_ds,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["data"].get("num_workers", 0),
)
test_loader = DataLoader(
    test_ds,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["data"].get("num_workers", 0),
)

logger.info("Data preparation completed successfully.")

# %% Add Ridge Regression with alpha=1.0 and benchmarking with MultiRunTrainer

# First, let's define a proper Ridge regression model with L2 regularization (alpha=1.0)
class RidgeRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, alpha: float = 1.0):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def get_l2_regularization(self):
        """Calculate L2 regularization term for Ridge regression"""
        l2_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2) ** 2
        return self.alpha * l2_reg / 2

class SimpleFCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Custom loss function for Ridge regression that includes L2 regularization
class RidgeLoss(nn.Module):
    def __init__(self):
        super(RidgeLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, outputs, targets, model):
        mse_loss = self.mse(outputs, targets)
        l2_reg = model.get_l2_regularization()
        return mse_loss + l2_reg

# Create wrapper models for MultiRunTrainer
class FCNNWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCNNWrapper, self).__init__()
        self.model = SimpleFCNN(input_dim, hidden_dim, output_dim)
    
    def forward(self, x):
        # Handle both tensor input and dictionary/tuple input
        if isinstance(x, dict) and 'transcriptomics' in x:
            return self.model(x['transcriptomics'])
        elif isinstance(x, dict) and len(x) > 0:
            # Get the first non-target key
            for key in x:
                if key != 'viability' and key != 'targets':
                    return self.model(x[key])
        # Default case - x is already a tensor
        return self.model(x)

class RidgeRegressionWrapper(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=1.0):
        super(RidgeRegressionWrapper, self).__init__()
        self.model = RidgeRegression(input_dim, output_dim, alpha)
    
    def forward(self, x):
        # Handle both tensor input and dictionary/tuple input
        if isinstance(x, dict) and 'transcriptomics' in x:
            return self.model(x['transcriptomics'])
        elif isinstance(x, dict) and len(x) > 0:
            # Get the first non-target key
            for key in x:
                if key != 'viability' and key != 'targets':
                    return self.model(x[key])
        # Default case - x is already a tensor
        return self.model(x)
    
    def get_l2_regularization(self):
        return self.model.get_l2_regularization()

# Create a criterion adapter for the RidgeLoss to work with the MultiRunTrainer
class RidgeCriterionAdapter:
    def __init__(self, model):
        self.ridge_loss = RidgeLoss()
        self.model = model
        
    def __call__(self, outputs, targets):
        # Ensure shapes match
        if outputs.dim() > targets.dim():
            targets = targets.view(-1, 1)
        return self.ridge_loss(outputs, targets, self.model)

# Setup device and common parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_runs = 20  # Number of runs for benchmarking
input_dim = train_ds[0][0].shape[0]  # Input dimension from the dataset
hidden_dim = 128  # Can be read from config if already defined

# Setup ExperimentTracker for benchmarking
benchmark_logger = ExperimentTracker(
    experiment_name=f"{exp_name}_benchmark",
    config=config,
    log_dir=config["paths"]["log_dir"],
    use_tensorboard=False,
    use_wandb=False
)

# %% Run benchmarking for FCNN
logger.info("Starting FCNN benchmark with 20 runs...")

fcnn_trainer = MultiRunTrainer(
    model_class=FCNNWrapper,
    model_kwargs={
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": 1
    },
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer_class= lambda params: optim.Adam(params, lr=config["training"]["learning_rate"]),
    criterion=nn.MSELoss(),
    device=device,
    experiment_tracker=benchmark_logger,
    config=config,
    num_runs=num_runs,
    save_models=True,
    output_dir=f"{config['paths']['results_dir']}/fcnn_benchmark"
)

fcnn_results = fcnn_trainer.run_training(
    epochs=200
)

# %% Run benchmarking for Ridge Regression
logger.info("Starting Ridge Regression benchmark with 20 runs...")

# Create a separate Ridge model for initializing the criterion adapter
ridge_model = RidgeRegressionWrapper(input_dim=input_dim, output_dim=1, alpha=1.0)
ridge_criterion = RidgeCriterionAdapter(ridge_model)

ridge_trainer = MultiRunTrainer(
    model_class=RidgeRegressionWrapper,
    model_kwargs={
        "input_dim": input_dim,
        "output_dim": 1,
        "alpha": 1.0
    },
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer_class=lambda params: optim.SGD(params, lr=config["training"]["learning_rate"]),
    criterion=ridge_criterion,
    device=device,
    experiment_tracker=benchmark_logger,
    config=config,
    num_runs=num_runs,
    save_models=True,
    output_dir=f"{config['paths']['results_dir']}/ridge_benchmark"
)

ridge_results = ridge_trainer.run_training(
    epochs=200
)

# %% Compare model results
logger.info("=== Model Comparison ===")
for metric in ["r2", "rmse", "mae", "pearson"]:
    fcnn_mean = fcnn_results.get("test", {}).get(f"{metric}_mean", float('nan'))
    fcnn_std = fcnn_results.get("test", {}).get(f"{metric}_std", float('nan'))
    ridge_mean = ridge_results.get("test", {}).get(f"{metric}_mean", float('nan'))
    ridge_std = ridge_results.get("test", {}).get(f"{metric}_std", float('nan'))
    
    logger.info(f"{metric.upper()}: FCNN = {fcnn_mean:.4f} ± {fcnn_std:.4f}, Ridge = {ridge_mean:.4f} ± {ridge_std:.4f}")

# Save comparison results
import json
comparison = {
    "fcnn": fcnn_results,
    "ridge": ridge_results,
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}

comparison_file = f"{config['paths']['results_dir']}/model_comparison.json"
with open(comparison_file, 'w') as f:
    json.dump(comparison, f, indent=2)

logger.info(f"Comparison results saved to {comparison_file}")