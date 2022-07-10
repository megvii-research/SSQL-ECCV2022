import os
import cv2
import numpy as np
import collections
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import nori2 as nori
from .utils import imdecode


class DATASET(Dataset):
    def __init__(self, transform, mode, eval_size=50000):
        self.instance_per_epoch = {
            "train": 5000 * 256,
            "validation": eval_size,
        }[mode]
        nfname = {
            "train": "imagenet.train.nori.list",
            "validation": "imagenet.val.nori.list",
        }[mode]
        self.nid_filename = "s3://hw-share/dataset/imagenet/ILSVRC2012/" + nfname
        self.nf = nori.Fetcher()
        self.nid_labels = self.load()
        assert mode in [
            "train",
            "validation",
        ], "only support mode equals train or validation"
        self.is_train = mode == "train"
        self.transform = transform
        self.nb_classes = 1000

    def load(self):
        self.nid_labels = []
        with nori.smart_open(self.nid_filename) as f:
            for line in f:
                nid, label, _ = line.strip().split("\t")
                self.nid_labels.append((nid, int(label)))
        return self.nid_labels[: self.instance_per_epoch]

    def __len__(self):
        return self.instance_per_epoch

    def __getitem__(self, idx):
        nid, label = self.nid_labels[idx]
        data = self.nf.get(nid)
        img = imdecode(data)[:, :, :3][:, :, ::-1]  # bgr -> rgb
        img = Image.fromarray(img)
        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


if __name__ == "__main__":
    import datasets

    dataset = DATASET(transform=None, mode="validation")
    img, label = dataset.__getitem__(0)
