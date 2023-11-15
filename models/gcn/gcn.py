import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphConvolutionModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphConvolutionModel, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # Apply global mean pooling to reduce spatial dimensions
        x = global_mean_pool(x, data.batch)

        x = self.fc(x)

        return x
