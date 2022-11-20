from lib.dataset.DRGrading import DRGrading
from lib.models.resnet import *
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2, os
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter


def main(*args, **kwargs):
    drg_data = DRGrading('assets/images', 'assets/gts', 'train', 'small_data')
    epochs = 1
    batch_size = 8
    model = ResNet(BasicBlock, [1, 1, 1, 1], 3, True)

    print(sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024))

    dataloader = DataLoader(
        dataset=drg_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    model.to('cuda:0')
    model.train()
    exp_path = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    exp_dir = os.path.join('exp', exp_path)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    summary = SummaryWriter(exp_dir)
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            global_step = step + epoch * len(dataloader)
            optimizer.zero_grad()
            images = batch['img'].to('cuda:0')
            labels = batch['label'].to('cuda:0')
            res = model(images)
            loss = F.cross_entropy(res, labels)
            loss.backward()
            optimizer.step()
            summary.add_scalar('loss', loss, global_step)


if __name__ == '__main__':
    main()