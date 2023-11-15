import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool


class GraphTransformer(nn.Module):
    def __init__(
        self, num_features, num_classes, num_layers, hidden_dim, num_heads, dropout
    ):
        super(GraphTransformer, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    num_features, hidden_dim, num_heads=num_heads, dropout=dropout
                )
            )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x = global_mean_pool(x, data.batch)

        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)