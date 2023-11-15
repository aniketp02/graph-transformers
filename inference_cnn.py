import imghdr
import os
import torch
import cv2
import torchvision
from torch.utils.data import DataLoader
from datasets.dataset_patternNet import PatternNetDataset
from models.cnn.cnn import CNN
from utils.data_utils import get_key_patternet

from PIL import ImageFont
from PIL import ImageDraw

if __name__ == "__main__":
    # Set the path to the trained model checkpoint
    checkpoint_path = "checkpoints/cnn/net_cnn_model_epoch_98.pth"

    # Set the path to the directory where you want to save the inference results
    output_dir = "results/cnn"
    os.makedirs(output_dir, exist_ok=True)

    # Load the trained model checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Initialize the CNN model
    net = CNN(input_channels=3).cuda()
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    # Set the data root and filelists path for inference
    data_root = "/home/ubuntu/PatternNet1/raw"
    filelists_path = "filelists/test_patternet.txt"  # If required

    # Create a dataset for inference
    infer_dataset = PatternNetDataset(
        filelists_path, data_root
    )  # You can modify the dataset constructor as needed

    # Create a data loader for inference
    infer_data_loader = DataLoader(
        dataset=infer_dataset, batch_size=1, shuffle=False
    )  # Batch size is set to 1 for inference

    # Perform inference and save results
    with torch.no_grad():
        for i, data in enumerate(infer_data_loader):
            label, img = data

            # Get the model's inference
            outputs = net(img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

            img = torchvision.transforms.ToPILImage()(img[0])
            # print(img.shape)
            # exit()
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 12
            )
            print(predicted_class)
            key = get_key_patternet(predicted_class)
            print(key)
            draw.text(
                (0, 0),
                f"Predicted Class: {key}",
                (0, 0, 0),
                font=font,
            )
            img = img.convert("1")
            img.save(os.path.join(output_dir, f"{i}.png"))

            # Save inference results
            # result_filename = str(label) + ".txt"
            # result_path = os.path.join(output_dir, result_filename)
            # with open(result_path, "w") as f:
            #     f.write(f"Image: {label}\nPredicted Class: {predicted_class}")

            print(f"Processed image {i+1}/{len(infer_data_loader)}")

    print("Inference completed. Results are saved in", output_dir)
