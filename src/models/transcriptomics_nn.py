import torch.nn as nn
import torch.nn.functional as F


class TranscriptomicsNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(TranscriptomicsNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x):
        # First layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.fc3(x)
        return x
