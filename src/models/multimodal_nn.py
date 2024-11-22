import torch
import torch.nn.functional as F


class MultimodalNN(torch.nn.Module):
    def __init__(self, chem_output_dim, trans_output_dim, hidden_dim, output_dim):
        super(MultimodalNN, self).__init__()
        self.fc1 = torch.nn.Linear(chem_output_dim + trans_output_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, chem_embedding, trans_embedding):
        x = torch.cat((chem_embedding, trans_embedding), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
