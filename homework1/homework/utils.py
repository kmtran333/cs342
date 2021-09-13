from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import os

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """

        with open(dataset_path, newline='') as csvfile:
            self.label_reader = csv.reader(csvfile, delimiter=',')

        next(self.label_reader)

        self.label_data = list(self.label_reader)
        self.label_data = self.label_data[:-1]

        os.chdir(dataset_path)

    def __len__(self):
        """
        Your code here
        """
        return len(self.label_data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """

        image_name = self.label_data[idx][0]

        return (torchvision.transforms.ToTensor(image_name), LABEL_NAMES.index(self.label_data[idx][1]))


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()