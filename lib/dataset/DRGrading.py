import sys, os
import pickle
import cv2
import pandas
import torch
import numpy as np
from lib.utils.logger import logger
from lib.const import Queries


class DRGrading:

    def __init__(self, img_dir, gt_dir, split, mode='use_full_loading', amount=100):
        self.imgs = []
        if split == 'train':
            img_dir = os.path.join(img_dir, 'train')
        elif split == 'test':
            img_dir = os.path.join(img_dir, 'test')
        elif split == 'val':
            img_dir = os.path.join(img_dir, 'val')

        gt_files = os.listdir(gt_dir)
        self.gtfile = pandas.read_csv(os.path.join(gt_dir, gt_files[0]))
        for image_name in self.gtfile['image name']:
            img_path = os.path.normpath(os.path.join(img_dir, image_name))
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"File doesn't exist, check path {img_path}.")
                exit()
            self.imgs.append(cv2.imread(img_path))
        self.labels = np.array(self.gtfile['DR grade'])
        self.imgs = np.array(self.imgs)
        if mode == 'small_data':
            logger.warning(f"You are using small amount of data, amount = {amount}.")
            sel_idx = np.array(np.arange(amount))
            sel_idx = np.random.permutation(sel_idx)
            self.labels = self.labels[sel_idx]
            self.imgs = self.imgs[sel_idx]
        elif mode == 'use_full_loading':
            pass
        else:
            logger.error("Unknown mode, 'use_full_loading', 'small_data' allowed.")

        logger.info('Dataset successfully loaded!')
    
    def set_device(self, device):
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            Queries.IMG: torch.from_numpy(self.imgs[idx]).to(torch.float32).transpose(2, 0).to(self.device),
            Queries.LABEL: torch.tensor(self.labels[idx]).to(self.device)
        }


if __name__ == '__main__':
    img_path = os.path.join('assets', 'images')
    gts_path = os.path.join('assets', 'gts')
    data = DRGrading(img_path, gts_path, 'train')
    for i in range(len(data.imgs)):
        print(data.imgs[i].shape, data.labels[i])