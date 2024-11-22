import torch
import torch.nn.functional as F


class TranscriptomicsNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TranscriptomicsNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
