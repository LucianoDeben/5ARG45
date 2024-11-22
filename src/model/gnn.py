import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)  # Global mean pooling
        x = self.fc(x)
        return x
