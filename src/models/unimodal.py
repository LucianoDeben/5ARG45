import torch.nn as nn
import torch.nn.functional as F


def init_weights(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout

        # Apply weight initialization
        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # Linear -> BatchNorm -> ReLU
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # No activation on the output layer
        return x


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * input_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout

        # Apply weight initialization
        self.apply(init_weights)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1d -> BatchNorm -> ReLU
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)  # Apply dropout
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # No activation on the output layer
        return x
