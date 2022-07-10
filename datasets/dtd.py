import os
import cv2
import numpy as np
import collections
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.datasets as datasets


class DATASET(datasets.ImageFolder):
    def __init__(self, transform, mode):
        if mode=='train':
            root = os.path.join('/data/datasets/dtd/split1', 'train')
        else:
            root = os.path.join('/data/datasets/dtd/split1', 'val')
        super(DATASET, self).__init__(root, transform)


if __name__ == "__main__":
    #import datasets

    dataset = DATASET(transform=None, mode="validation")
    print(len(dataset))
    img, label = dataset.__getitem__(0)
    