import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from . import dense_transforms

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'drive_data'


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for file in os.listdir(dataset_path):
            if '.png' in file:
                self.image_paths.append(file)
            elif '.csv' in file:
                with open('%s/%s' % (dataset_path, file)) as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for row in reader:
                        self.labels.append([float(r) for r in row][0:2])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open('%s/%s' % (self.dataset_path, image_path))
        label = self.labels[idx]

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = dense_transforms.ToTensor()(image)

        return (image_tensor[0], torch.Tensor(label))
