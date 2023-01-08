import sys, os
sys.path.append('.')
import pickle
import cv2
import pandas
import skimage
import numpy as np
from tqdm import tqdm
from lib.utils.logger import logger
from lib.const import Queries
from lib.preprocess.ButterworthHighPassFilter import BHF
from lib.preprocess.HighPassFilter import HPF
from torch.utils.data import Dataset
from matplotlib import pyplot
import pickle

def ButterworthHighPassFilter(image, d=20, n=1):

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    s1 = s1 = np.log(np.abs(fshift))

    def make_transform_matrix(d):
        transform_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):

                def cal_distance(pa, pb):
                    from math import sqrt

                    dis = sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transform_matrix[i, j] = 1 / (1 + (d / dis)**(2 * n))
        return transform_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img

def HistEqualizer(img):
    img = cv2.equalizeHist(img)
    return img

def GaussianBlur(img):
    img = cv2.GaussianBlur(img, (5, 5), 9)
    return img

def GaussianNoise(img):
    img = skimage.util.random_noise(img, mode='gaussian')
    return img

def translate(img, shift=10, direction='right', roll=True): 
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right' 
    img = img.copy() 
    if direction == 'right': 
        right_slice = img[:, -shift:].copy() 
        img[:, shift:] = img[:, :-shift] 
        if roll: 
            img[:,:shift] = np.fliplr(right_slice) 
    if direction == 'left': 
        left_slice = img[:, :shift].copy() 
        img[:, :-shift] = img[:, shift:] 
        if roll: 
            img[:, -shift:] = left_slice 
    if direction == 'down': 
        down_slice = img[-shift:, :].copy() 
        img[shift:, :] = img[:-shift,:] 
        if roll: 
            img[:shift, :] = down_slice 
    if direction == 'up': 
        upper_slice = img[:shift, :].copy() 
        img[:-shift, :] = img[shift:, :] 
        if roll: 
            img[-shift:,:] = upper_slice 

def make_translate(shift, direction, roll):
    def trans(img):
        return translate(img, shift, direction, roll)
    return trans

def flip_img(img, mode='ud'): 
    if mode == 'ud':
        return np.flipud(img)
    elif mode == 'lr':
        return np.fliplr(img)

def make_flip(mode):
    def flip(img):
        return flip_img(img, mode)
    return flip

processings = [GaussianNoise, make_flip('lr'), make_flip('ud'), make_translate(20, 'left', False)]

class DataAugmentation:

    def __init__(self, gt_dir, img_dir, target_dir):
        self.gt_dir = gt_dir
        self.img_dir = img_dir
        self.target_dir = target_dir
        gt_files = os.listdir(gt_dir)
        self.imgs = []
        self.gtfile = pandas.read_csv(os.path.join(gt_dir, gt_files[0]))
        for image_name in self.gtfile['image name']:
            img_path = os.path.normpath(os.path.join(img_dir, image_name))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"File doesn't exist, check path {img_path}.")
                exit()
            self.imgs.append(img)
        self.labels = np.array(self.gtfile['DR grade'])
        self.imgs = np.array(self.imgs)
    
    def split(self, train_val_ratio=3):
        idx_0 = self.labels == 0
        idx_1 = self.labels == 1
        idx_2 = self.labels == 2
        grade_0_img = self.imgs[idx_0]
        grade_1_img = self.imgs[idx_1]
        grade_2_img = self.imgs[idx_2]
        grade_0_label = self.labels[idx_0]
        grade_1_label = self.labels[idx_1]
        grade_2_label = self.labels[idx_2]

        perm_0 = np.random.permutation(grade_0_img.shape[0])
        split_0 = len(perm_0) * train_val_ratio // (train_val_ratio + 1)
        perm_1 = np.random.permutation(grade_1_img.shape[0])
        split_1 = len(perm_1) * train_val_ratio // (train_val_ratio + 1)
        perm_2 = np.random.permutation(grade_2_img.shape[0])
        split_2 = len(perm_2) * train_val_ratio // (train_val_ratio + 1)

        train_0_img, val_0_img = grade_0_img[perm_0[:split_0]], grade_0_img[perm_0[split_0:]]
        train_0_label, val_0_label = grade_0_label[perm_0[:split_0]], grade_0_label[perm_0[split_0:]]
        train_1_img, val_1_img = grade_1_img[perm_1[:split_1]], grade_1_img[perm_1[split_1:]]
        train_1_label, val_1_label = grade_1_label[perm_1[:split_1]], grade_1_label[perm_1[split_1:]]
        train_2_img, val_2_img = grade_2_img[perm_2[:split_2]], grade_2_img[perm_2[split_2:]]
        train_2_label, val_2_label = grade_2_label[perm_2[:split_2]], grade_2_label[perm_2[split_2:]]

        train_img = np.concatenate((train_0_img, train_1_img, train_2_img), axis=0)
        train_label = np.concatenate((train_0_label, train_1_label, train_2_label), axis=0)
        val_img = np.concatenate((val_0_img, val_1_img, val_2_img), axis=0)
        val_label = np.concatenate((val_0_label, val_1_label, val_2_label), axis=0)
        assert len(train_img) == len(train_label)
        perm_train = np.random.permutation(len(train_img))
        perm_val = np.random.permutation(len(val_img))
        train_img = train_img[perm_train]
        train_label = train_label[perm_train]
        val_img = val_img[perm_val]
        val_label = val_label[perm_val]
        train_data, val_data =  (train_img, train_label), (val_img, val_label)
        train_data = self.augment(train_data)
        os.makedirs(self.target_dir, exist_ok=True)
        with open(os.path.join(self.target_dir, "train.pkl"), 'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(self.target_dir, "val.pkl"), 'wb') as f:
            pickle.dump(val_data, f)
    
    def augment(self, data):
        imgs = data[0]
        labels = data[1]
        for i in tqdm(range(len(imgs))):
            img, label = imgs[i], labels[i]
            for proc in processings:
                img = img.astype(np.uint8)
                img = proc(img)
                np.append(imgs, [img], axis=0)
                np.append(labels, [label], axis=0)
        random_idx = np.random.permutation(len(imgs))
        imgs = imgs[random_idx]
        labels = labels[random_idx]
        return (imgs, labels)
    
    def vis_process(self, process, num=2):
        idx = np.random.choice(len(self.imgs), num, replace=False)
        imgs = list(self.imgs[idx])
        processed_imgs = [process(img) for img in imgs]
        for img in processed_imgs:
            pyplot.imshow(img)



if __name__ == '__main__':
    gt_dir = 'assets/gts'
    img_dir = 'assets/images/train'
    target_dir = 'assets/DRG_data'
    data_augmentation = DataAugmentation(gt_dir, img_dir, target_dir)
    # data_augmentation.split(train_val_ratio=3)
    DataAugmentation.vis_process(make_translate(20, 'right', True), 2)
        

