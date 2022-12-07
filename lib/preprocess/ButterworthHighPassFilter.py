import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


class BHF:

    def ButterworthHighPassFilter(self, image, d=20, n=1):

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

    def read(self, path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def write(self, path, img):
        return cv2.imwrite(path, img)

    def __init__(self, img_dir, gt_dir, split):
        self.img = []
        fft_dir = os.path.join(img_dir, 'processed', 'BHF')
        if split == 'train':
            img_dir = os.path.join(img_dir, 'train')
        elif split == 'test':
            img_dir = os.path.join(img_dir, 'test')
        elif split == 'val':
            img_dir = os.path.join(img_dir, 'val')
        gt_files = os.listdir(gt_dir)
        self.gtfile = pd.read_csv(os.path.join(gt_dir, gt_files[0]))
        for image_name in self.gtfile['image name']:
            img_path = os.path.normpath(os.path.join(img_dir, image_name))
            img = self.read(img_path)
            self.write(os.path.normpath(os.path.join(fft_dir, image_name)), self.ButterworthHighPassFilter(img, 20, 1))


if __name__ == "__main__":
    img_path = os.path.join('assets', 'images')
    gts_path = os.path.join('assets', 'gts')
    fft = BHF(img_path, gts_path, 'train')
