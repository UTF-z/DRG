import sys, os

sys.path.append('.')
import pickle
import cv2
import pandas
import skimage
import numpy as np
import copy
from tqdm import tqdm
from lib.utils.logger import DRGLogger
from lib.const import Queries
from lib.preprocess.ButterworthHighPassFilter import BHF
from lib.preprocess.HighPassFilter import HPF
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
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
            img[:, :shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift, :]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:, :] = upper_slice
    return img


def make_translate(shift, direction, roll):

    def trans(img):
        return translate(img, shift, direction, roll)

    return trans


def flip_img(img, mode='ud'):
    if mode == 'ud':
        img = np.flipud(img)
        img = np.ascontiguousarray(img)
        return img
    elif mode == 'lr':
        img = np.fliplr(img)
        img = np.ascontiguousarray(img)
        return img


def make_flip(mode):

    def flip(img):
        return flip_img(img, mode)

    return flip


processings = [
    GaussianNoise,
    make_flip('lr'),
    make_flip('ud'),
    make_translate(20, 'left', False),
    make_translate(20, 'right', False),
    make_translate(20, 'up', False),
    make_translate(20, 'down', False)
]


class DataAugmentation:

    def __init__(self, img_dir, target_dir, gt_dir = None):
        if gt_dir is not None:
            self.gt_dir = gt_dir
            self.img_dir = img_dir
            self.target_dir = target_dir
            gt_files = os.listdir(gt_dir)
            self.imgs = []
            self.gtfile = pandas.read_csv(os.path.join(gt_dir, gt_files[0]))
            for image_name in self.gtfile['image name']:
                img_path = os.path.normpath(os.path.join(img_dir, image_name))
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = img / 255.0
                if img is None:
                    DRGLogger.warning(f"File doesn't exist, check path {img_path}.")
                    exit()
                self.imgs.append(img)
            self.labels = np.array(self.gtfile['DR grade'])
            self.img_names = np.array(self.gtfile['image name'])
            self.imgs = np.array(self.imgs)
        else:
            self.img_dir = img_dir
            self.target_dir = target_dir
            self.img_names = np.array(os.listdir(self.img_dir))
            self.imgs = []
            for name in self.img_names:
                path = os.path.join(self.img_dir, name)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = img / 255.0
                if img is None:
                    DRGLogger.warning(f"File doesn't exist, check path {img_path}.")
                    exit()
                self.imgs.append(img)
            self.imgs = np.array(self.imgs)
            self.labels = None

    def split(self, train_val_ratio=3):
        assert self.labels is not None
        total_train_imgs = []
        total_train_names = []
        total_train_labels = []
        total_val_imgs = []
        total_val_names = []
        total_val_labels = []
        for label in range(3):
            idx = self.labels == label
            imgs = self.imgs[idx]
            names = self.img_names[idx]
            labels = self.labels[idx]
            perm = np.random.permutation(len(imgs))
            imgs = imgs[perm]
            names = names[perm]
            labels = labels[perm]
            split = len(imgs) * train_val_ratio // (train_val_ratio + 1)
            train_imgs = imgs[:split]
            train_names = names[:split]
            train_labels = labels[:split]
            val_imgs = imgs[split:]
            val_names = names[split:]
            val_labels = labels[split:]
            total_train_imgs.append(train_imgs)
            total_train_names.append(train_names)
            total_train_labels.append(train_labels)
            total_val_imgs.append(val_imgs)
            total_val_names.append(val_names)
            total_val_labels.append(val_labels)
        total_train_imgs = np.concatenate(total_train_imgs, axis=0)
        total_train_names = np.concatenate(total_train_names, axis=0)
        total_train_labels = np.concatenate(total_train_labels, axis=0)
        total_val_imgs = np.concatenate(total_val_imgs, axis=0)
        total_val_names = np.concatenate(total_val_names, axis=0)
        total_val_labels = np.concatenate(total_val_labels, axis=0)
        perm_train = np.random.permutation(len(total_train_imgs))
        perm_val = np.random.permutation(len(total_val_imgs))
        total_train_imgs = total_train_imgs[perm_train]
        total_train_labels = total_train_labels[perm_train]
        total_train_names = total_train_names[perm_train]
        total_val_imgs = total_val_imgs[perm_val]
        total_val_labels = total_val_labels[perm_val]
        total_val_names = total_val_names[perm_val]
        print(total_train_labels[:10])
        print(total_train_names[:10])
        print(total_val_labels[:10])
        print(total_val_names[:10])
        train_data = []
        val_data = []
        for i in range(len(total_train_imgs)):
            curr_img = total_train_imgs[i]
            curr_label = total_train_labels[i]
            curr_name = total_train_names[i]
            curr_data = {
                Queries.IMG: curr_img,
                Queries.LABEL: curr_label,
                Queries.NAME: curr_name
            }
            train_data.append(curr_data)
        for i in range(len(total_val_imgs)):
            curr_img = total_val_imgs[i]
            curr_label = total_val_labels[i]
            curr_name = total_val_names[i]
            curr_data = {
                Queries.IMG: curr_img,
                Queries.LABEL: curr_label,
                Queries.NAME: curr_name
            }
            val_data.append(curr_data)
        print(len(train_data), len(val_data))
        os.makedirs(self.target_dir, exist_ok=True)
        with open(os.path.join(self.target_dir, "train.pkl"), 'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(self.target_dir, "val.pkl"), 'wb') as f:
            pickle.dump(val_data, f)
    
    def generate_testset(self):
        assert self.labels is None
        test_data = []
        for i in range(len(self.imgs)):
            curr_img = self.imgs[i]
            curr_name = self.img_names[i]
            curr_data = {
                Queries.IMG: curr_img,
                Queries.NAME: curr_name,
                Queries.LABEL: None
            }
            test_data.append(curr_data)
        print(len(test_data))
        os.makedirs(self.target_dir, exist_ok=True)
        with open(os.path.join(self.target_dir, "test.pkl"), 'wb') as f:
            pickle.dump(test_data, f)

    @staticmethod
    def augment_offline(in_path, out_path):
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        vis_path = 'tmp'
        imgs = [item[Queries.IMG] for item in data]
        labels = [item[Queries.LABEL] for item in data]
        names =[item[Queries.NAME] for item in data]
        new_imgs = copy.deepcopy(imgs)
        new_labels = copy.deepcopy(labels)
        new_names = copy.deepcopy(names)
        for i in tqdm(range(len(imgs))):
            img, label, name = imgs[i], labels[i], names[i]
            for j, proc in enumerate(processings):
                proc_img = proc(img)
                new_imgs.append(proc_img)
                new_labels.append(label)
                new_names.append(name)
                # print(j, ">>>", proc_img.mean(), proc_img.min(), proc_img.max(), proc_img.dtype)
                # vis_name = name + '-' + str(i) + '-' + proc.__name__ + '.png'
                # plt.subplot(1, 1, 1)
                # plt.imshow(proc_img, cmap='gray')
                # plt.savefig(os.path.join(vis_path, vis_name))
        random_idx = np.random.permutation(len(new_labels))
        new_imgs = np.array(new_imgs)
        new_labels = np.array(new_labels)
        new_names = np.array(new_names)
        imgs = new_imgs[random_idx]
        labels = new_labels[random_idx]
        names = new_names[random_idx]
        data_aug = []
        for i in range(len(imgs)):
            curr_img = imgs[i]
            curr_label = labels[i]
            curr_name = names[i]
            curr_data = {
                Queries.IMG: curr_img,
                Queries.LABEL: curr_label,
                Queries.NAME: curr_name
            }
            data_aug.append(curr_data)
        print(len(data_aug))
        with open(out_path, 'wb') as f:
            pickle.dump(data_aug, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def augment_online(in_data):
        new_data = []
        for data in tqdm(in_data):
            for proc in processings:
                proc_data = {
                    Queries.IMG: proc(data[Queries.IMG]),
                    Queries.LABEL: data[Queries.LABEL],
                    Queries.NAME: data[Queries.NAME]
                }
                new_data.append(proc_data)
        new_data.extend(in_data)
        return new_data

    def vis_process(self, process, num=2):
        idx = np.random.choice(len(self.imgs), num, replace=False)
        imgs = list(self.imgs[idx])
        processed_imgs = [process(img) for img in imgs]
        output_dir = 'tmp'
        for i in range(len(imgs)):
            before = imgs[i]
            after = processed_imgs[i]
            print((before - after).mean(), np.sqrt((before - after).var()))
            print(before.dtype, before.min(), before.max(), before.mean())
            print(after.dtype, after.min(), after.max(), after.mean())
            plt.subplot(1, 2, 1)
            plt.imshow(imgs[i], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(processed_imgs[i], cmap='gray')
            plt.savefig(os.path.join(output_dir, process.__name__ + str(i) + '.png'))


if __name__ == '__main__':
    gt_dir = 'assets/gts'
    img_dir = 'assets/images/train'
    target_dir = 'assets/DRG_data/Triplet'
    test_img_dir = 'assets/images/test'
    # data_augmentation = DataAugmentation(img_dir, target_dir, gt_dir=gt_dir)
    # data_augmentation.split(train_val_ratio=3)
    # data_augmentation.vis_process(GaussianNoise, 2)
    # data_augmentation.augment(r'assets/DRG_data/train.pkl', r'assets/DRG_data/train_aug_trans_ud.pkl')
    # test_data = DataAugmentation(test_img_dir, target_dir)
    # test_data.generate_testset()
    # DataAugmentation.augment_offline(r'assets/DRG_data/Triplet/train.pkl', r'assets/DRG_data/Triplet/train_aug_all_final.pkl')

    img = np.zeros((1024, 1024), dtype=np.float64)
    img[:256, :256] = 1
    data = [{
        Queries.IMG: img,
        Queries.NAME: 'haha',
        Queries.LABEL: -1
    }]
    out_data = DataAugmentation.augment_online(data)
    output_dir = 'tmp'
    for i, data in enumerate(out_data):
        name = 'test' + str(i) + '.png'
        plt.imshow(data[Queries.IMG])
        plt.savefig(os.path.join(output_dir, name))

