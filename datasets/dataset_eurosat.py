import torch
import os
import cv2
# from PIL import Image
from libtiff import TIFF

from torch.utils.data import Dataset

from utils.data_utils import get_label_eurosat

class EuroSatDataset(Dataset):
    def __init__(self, filelist_path, data_root):
        self.data_root = data_root
        self.filelist  = open(filelist_path, 'r')
        self.data = self.filelist.readlines()
        self.length = len(self.data)

        # self.classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        # self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # self.data = self._load_data()
        # self.transform = transform

    # def _load_data(self):
    #     data = []
    #     for cls in self.classes:
    #         class_dir = os.path.join(self.data_root, cls)
    #         for filename in os.listdir(class_dir):
    #             img_path = os.path.join(class_dir, filename)
    #             data.append((img_path, self.class_to_idx[cls]))
    #     return data

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.data[index]).split('\n')[0]
        label_name = self.data[index].split('/')[0]
        print(img_path)
        tif = TIFF.open(img_path)
        img = tif.read_image().astype('int16')

        for i, image in enumerate(tif.iter_images()):
            print(image.shape)
            print(i)
            # cv2.imwrite(f"{i}.png", image)
            # pass

        tif = TIFF.open('filename.tif', mode='w')
        tif.write_image(image)

        print(img.shape)
        # print(img)
        img = torch.FloatTensor(img)
        img = img.permute(2, 0, 1).cuda()

        labels = get_label_eurosat(label_name)
        labels = torch.FloatTensor(labels).cuda()

        return labels, img
