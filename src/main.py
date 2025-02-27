from data_sets import LINCSDataset
from evaluators import ModelEvaluator
from splitters import train_val_test_split
import decoupler as dc
from sklearn.linear_model import LinearRegression
from trainers import PyTorchTrainer, SklearnTrainer
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    dataset = LINCSDataset(gctx_path="../data/processed/LINCS.gctx", feature_space="landmark")
    
    tf_net = dc.get_collectri()  # or load your own TF network
    dataset.run_tf_inference(net=tf_net, method="ulm")
    
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
    dataset, source="tf", random_state=42, group_column="cell_mfc_name"
)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    
    sklearn_model = LinearRegression()
    sklearn_trainer = SklearnTrainer(sklearn_model, name="Linear Regression")
    
    class MultimodalModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.fc(x)

    pytorch_model = MultimodalModel(input_dim=dataset.X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
    pytorch_trainer = PyTorchTrainer(pytorch_model, loss_fn, optimizer, name="PyTorch Model")
    
    evaluator = ModelEvaluator(n_runs=5, random_state=42)
    trainers = {"Linear Regression": sklearn_trainer, "PyTorch Model": pytorch_trainer}
    cv_results = evaluator.evaluate_multiple_models(trainers, dataset, source="tf")
    
