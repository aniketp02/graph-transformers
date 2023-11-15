import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from datasets.dataset_graph_patternNet import PatternNetGraphDataset
from datasets.dataset_eurosat import EuroSatDataset
from models.transformers.graph_transformer import GraphTransformer

from config.config import CNNTrainingOptions
import numpy as np
import random

# Set your model hyperparameters
num_features = 1  # Number of input features (adjust as needed)
num_classes = 12  # Number of classes (adjust as needed)
learning_rate = 0.001
num_layers = 1
hidden_dim = 64
num_heads = 4
dropout = 0.5

# load config
opt = CNNTrainingOptions().parse_args()
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

# Initialize your Graph Transformer model
model = GraphTransformer(
    num_features, num_classes, num_layers, hidden_dim, num_heads, dropout
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)
# exit()

# load training data
if opt.dataset_name == "PatternNet":
    train_data = PatternNetGraphDataset(root=opt.data_root)
elif opt.dataset_name == "EuroSAT":
    train_data = EuroSatDataset(opt.filelists, opt.data_root)
else:
    raise ("Dataset Not Supported!")

training_data_loader = DataLoader(
    dataset=train_data, batch_size=opt.batch_size, shuffle=True
)
train_data_length = len(training_data_loader)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop for PatternNet dataset
for epoch in range(opt.epochs):
    model.train()
    for data in training_data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape, data.y.shape)
        # exit()
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        print(f"loss is {loss.item()}")

    print(f"PatternNet Epoch [{epoch + 1}/{opt.epochs}] Loss: {loss.item()}")
    # Save your trained model for later use
    torch.save(model.state_dict(), "checkpoints/graph_transformer_model.pth")