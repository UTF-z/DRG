import sys, os
sys.path.append('.')
import pickle
import cv2
import pandas
import torch
import numpy as np
from lib.utils.logger import logger
from lib.const import Queries
from torch.utils.data import Dataset
from lib.dataset.data_prepare import DataAugmentation


class DRGrading(Dataset):

    def __init__(self, img_dir, split):
        self.imgs = []
        if split == 'train':
            data_path = os.path.join(img_dir, 'train.pkl')
        elif split == 'val':
            data_path = os.path.join(img_dir, 'val.pkl')

        with open(data_path, 'rb') as f:
            self.imgs, self.labels = pickle.load(f)
        assert len(self.imgs) == len(self.labels)
        f, s, t = (self.labels == 0).sum(), (self.labels == 1).sum(), (self.labels == 2).sum()
        print(f, s, t)
        logger.info(f'Dataset successfully loaded! Total length: {len(self.imgs)}')

    def set_device(self, device):
        self.device = device

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return {
            Queries.IMG: torch.from_numpy(img).to(torch.float32).unsqueeze(0).to(self.device),
            Queries.LABEL: torch.tensor(self.labels[idx]).to(self.device)
        }


if __name__ == '__main__':
    img_path = os.path.join('assets', 'DRG_data')
    data = DRGrading(img_path, 'train')
    for i in range(5):
        print(data.imgs[i].shape, data.imgs[i].dtype, type(data.imgs[i]), data.labels[i])
        print(data.imgs[i].min(), data.imgs[i].max())
        print("<<<<<<<<<<<<<<<<")
    print(len(data))
