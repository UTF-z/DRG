import sys, os
sys.path.append('.')
import pickle
import cv2
import pandas
import torch
import numpy as np
from lib.utils.logger import DRGLogger
from lib.const import Queries
from torch.utils.data import Dataset
from lib.dataset.data_processing import DataAugmentation


class DRGrading(Dataset):

    def __init__(self, img_dir, split):
        super(DRGrading, self).__init__()
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
        DRGLogger.info(f'Dataset successfully loaded! Total length: {len(self.imgs)}')

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

class TripletDataset(Dataset):

    def __init__(self, img_dir, split, aug='raw'):
        super(TripletDataset, self).__init__()
        self.imgs = []
        self.split = split
        if split == 'train':
            if aug == 'raw':
                DRGLogger.info("DATASET: Using Raw Data")
                data_path = os.path.join(img_dir, 'train.pkl')
            elif aug == 'fl':
                DRGLogger.info("DATASET: Using Flipped_aug Data")
                data_path = os.path.join(img_dir, 'train_aug_fl.pkl')
            elif aug == 'all':
                DRGLogger.info("DATASET: Using All_aug Data")
                data_path = os.path.join(img_dir, 'train_aug_all.pkl')
        elif split == 'val':
            data_path = os.path.join(img_dir, 'val.pkl')
        elif split == 'test':
            data_path = os.path.join(img_dir, 'test.pkl')
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        imgs = np.array([e[Queries.IMG] for e in self.data])
        self.mean = imgs.mean()
        self.std = np.sqrt(imgs.var())
        DRGLogger.info("Dataset: " + split + "--" + str(len(self.data)))
    
    def set_device(self, device):
        self.device = device
    
    def get_mean_and_std(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx][Queries.IMG]).to(torch.float32).unsqueeze(0).to(self.device)
        if self.split == 'test':
            label = torch.tensor(-1).to(self.device)
        else:
            label = torch.tensor(self.data[idx][Queries.LABEL]).to(self.device)
        name = self.data[idx][Queries.NAME]
        return {
            Queries.IMG: img,
            Queries.LABEL: label,
            Queries.NAME: name
        }


class TripletDataset_online_augment(Dataset):

    def __init__(self, img_dir, split, aug='raw'):
        super(TripletDataset_online_augment, self).__init__()
        self.imgs = []
        self.split = split
        if split == 'train':
            if aug == 'raw':
                data_path = os.path.join(img_dir, 'train.pkl')
            elif aug == 'fl':
                data_path = os.path.join(img_dir, 'train_aug_fl.pkl')
        elif split == 'val':
            data_path = os.path.join(img_dir, 'val.pkl')
        elif split == 'test':
            data_path = os.path.join(img_dir, 'test.pkl')
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        if split == 'train':
            self.data = DataAugmentation.augment_online(self.data)
        
        imgs = np.array([e[Queries.IMG] for e in self.data])
        self.mean = imgs.mean()
        self.std = np.sqrt(imgs.var())
        print(self.mean, self.std)
        DRGLogger.info("Dataset: " + split + "--" + str(len(self.data)))
    
    def set_device(self, device):
        self.device = device
    
    def get_mean_and_std(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx][Queries.IMG]).to(torch.float32).unsqueeze(0).to(self.device)
        if self.split == 'test':
            label = torch.tensor(-1).to(self.device)
        else:
            label = torch.tensor(self.data[idx][Queries.LABEL]).to(self.device)
        name = self.data[idx][Queries.NAME]
        return {
            Queries.IMG: img,
            Queries.LABEL: label,
            Queries.NAME: name
        }













