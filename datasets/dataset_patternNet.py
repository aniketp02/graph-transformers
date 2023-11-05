''' Extension of the Dataset class of pytorch '''

import torch
import os
import cv2

from torch.utils.data import Dataset
# from torch_geometric.data import Dataset

from utils.data_utils import get_label_patternet


class PatternNetDataset(Dataset):
    def __init__(self, filelist_path, data_root):
        super(PatternNetDataset, self).__init__()
        self.filelist  = open(filelist_path, 'r')
        self.data = self.filelist.readlines()
        self.data_root = data_root
        self.length = len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.data[index]).split('\n')[0]
        label_name = self.data[index].split('/')[0]

        img = cv2.imread(img_path)
        img = torch.FloatTensor(img)
        img = img.permute(2, 0, 1).cuda()

        labels = get_label_patternet(label_name)
        labels = torch.FloatTensor(labels).cuda()

        return labels, img

    def __len__(self):
        return self.length