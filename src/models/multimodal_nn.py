import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalNN(nn.Module):
    def __init__(
        self, chem_output_dim, trans_output_dim, hidden_dim, output_dim, dropout=0.1
    ):
        super(MultimodalNN, self).__init__()
        self.fc1 = nn.Linear(chem_output_dim + trans_output_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, chem_embedding, trans_embedding):
        # Concatenate embeddings
        x = torch.cat((chem_embedding, trans_embedding), dim=1)

        # First layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.fc3(x)
        return x
