import sys
sys.path.append('.')
import pickle
import os
import numpy
from matplotlib import pyplot as plt
from lib.const import Queries

in_path = "assets/DRG_data/Triplet/train_aug_all.pkl"
out_dir = "tmp"

with open(in_path, 'rb') as f:
    data = pickle.load(f)

for i in range(10):
    sample = data[i]
    img = sample[Queries.IMG]
    label = sample[Queries.LABEL]
    name = sample[Queries.NAME]
    ax = plt.subplot(1, 1, 1)
    print(img.shape, img.dtype, img.min(), img.max(), img.mean())
    plt.imshow(img, cmap='gray')
    plt.title(name + ": " + str(label))
    output_name = f"{str(i)}.png"
    plt.savefig(os.path.join(out_dir, output_name))