import os.path as osp
import glob
from pathlib import Path

import cv2
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data

from utils.data_utils import get_label_patternet


class PatternNetGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PatternNetGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        # print(self.raw_dir)
        raw_files = glob.glob(f"{self.raw_dir}/*/*.jpg")
        return raw_files

    @property
    def processed_file_names(self):
        processed_files = glob.glob(f"{self.processed_dir}/*/*.pt")
        return processed_files

    def process(self):
        for idx, raw_path in enumerate(self.raw_file_names):
            # img_path = osp.join(self.root, raw_path).split('\n')[0]
            img_path = raw_path
            print("raw path is :", raw_path)

            label_file = raw_path.split('raw/')[-1]
            label_name = label_file.split('/')[0]
            print("label name is :", label_name)

            labels = get_label_patternet(label_name)
            labels = torch.FloatTensor(labels).cuda()

            img = cv2.imread(img_path)
            H, W, C = img.shape

            edge_index, pos = torch_geometric.utils.grid(H, W)

            img = img.reshape(H * W, C)
            img = torch.FloatTensor(img)

            y = labels.unsqueeze(0).repeat(H * W, 1)

            data = Data(x=img, edge_index=edge_index, y=y, pos=pos)

            processed_directory = osp.join(self.processed_dir, label_file)
            processed_file = label_file.replace(label_file.split('.')[-1], 'pt')
            processed_file_path = osp.join(self.processed_dir, processed_file)

            # print("processed file path :", processed_file_path)
            # break

            Path(processed_directory).mkdir(parents=True, exist_ok=True)

            torch.save(data, processed_file_path)


    def len(self):
        return len(self.raw_file_names)
    
    def get(self, idx):
        data_path = self.processed_file_names[idx].split('\n')[0]
        # print("data path is ", data_path)
        data = torch.load(data_path)
        return data