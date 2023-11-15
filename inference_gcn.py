import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from datasets.dataset_graph_patternNet import PatternNetGraphDataset
from models.gcn.gcn import GraphConvolutionModel

# Set your model hyperparameters
num_features = 1  # Number of input features (adjust as needed)
num_classes = 12  # Number of classes (adjust as needed)

# Initialize your GCN model
model = GraphConvolutionModel(num_features, num_classes)  # Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the trained model checkpoint
checkpoint_path = "graph_convolution_model.pth"
model_state_dict = torch.load(checkpoint_path)

model.load_state_dict(model_state_dict)
model.eval()

# Set up data loaders for inference
inference_data = PatternNetGraphDataset(root="/home/ubuntu/infer_patternNet")

patternnet_loader = DataLoader(inference_data, batch_size=1, shuffle=False)


# Define a function to perform inference
def infer(model, data_loader, dataset_name):
    results = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            output = model(data)
            print("output:", output[0].shape)
            predicted = torch.argmax(output[0])
            print("predicted:", predicted)
            true_label = torch.argmax(data.y[0])
            results.append(
                f"{dataset_name} - True label {true_label}: Predicted Class: {predicted.item()}"
            )
    return results


# Perform inference on PatternNet
patternnet_results = infer(model, patternnet_loader, "PatternNet")

# Save inference results to text files
with open("results/patternnet_inference_results.txt", "w") as f:
    f.write("\n".join(patternnet_results))

print("Inference completed and results saved.")
