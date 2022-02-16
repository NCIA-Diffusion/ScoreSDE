import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10

            
def DataLooper(config, batch_size):
    if config.dataset.name.lower() == 'cifar10':
        dataset = CIFAR10(
            root=os.path.join(config.dataset.root, 'train'),
            train=True,
            download=False,
            transform=T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.train.n_workers,
            drop_last=True,
        )
        while True:
            for x, _ in iter(dataloader):
                yield x
                
    elif config.dataset.x_name.lower() == 'mnist':
        dataset = MNIST(
            root=os.path.join(config.dataset.root),
            train=True,
            download=False,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ])
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.train.n_workers,
            drop_last=True,
        )
        while True:
            for x, _ in iter(dataloader):
                yield x




