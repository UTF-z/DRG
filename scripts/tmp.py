import pickle
import numpy as np
import os

dir = r'assets/DRG_data'
fl = os.path.join(dir, 'train_aug_fl.pkl')
ga = os.path.join(dir, 'train_aug_gaussian.pkl')
rl = os.path.join(dir, 'train_aug_trans_rl.pkl')
ud = os.path.join(dir, 'train_aug_trans_ud.pkl')

paths = [fl, ga, rl, ud]
data = []
labels = []
for path in paths:
    with open(path, 'rb') as f:
        d = pickle.load(f)
        imgs = d[0]
        label = d[1]
    data.append(imgs)
    labels.append(label)

data = np.concatenate(data, axis=0)
labels = np.concatenate(labels, axis=0)
idx = np.random.permutation(len(data))
data = data[idx]
labels = labels[idx]
out_path = os.path.join(dir, 'train_final_aug.pkl')
with open(out_path, 'wb') as f:
    pickle.dump((data, labels), f)
