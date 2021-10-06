from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torch
import csv
import os

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path, resize=None, random_crop=None, random_horizontal_flip=False, normalize=False):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """
        self.data_path = dataset_path
        self.resize = resize
        self.random_crop = random_crop
        self.random_horizontal_flip = random_horizontal_flip
        self.normalize = normalize
        with open(os.getcwd() + '/' + dataset_path + '/' + 'labels.csv', newline='') as csvfile:
            self.label_reader = csv.reader(csvfile, delimiter=',')

            next(self.label_reader)

            self.label_data = list(self.label_reader)

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
        import torchvision
        image_name = self.label_data[idx][0]
        # trans = transforms.Compose([transforms.ToTensor()])

        transform = []
        if self.resize is not None:
            transform.append(torchvision.transforms.Resize(self.resize))
        if self.random_crop is not None:
            transform.append(torchvision.transforms.RandomResizedCrop(self.random_crop))
        if self.random_horizontal_flip:
            transform.append(torchvision.transforms.RandomHorizontalFlip())
        transform.append(torchvision.transforms.ToTensor())
        if self.normalize:
            transform.append(
                torchvision.transforms.Normalize(mean=[0.4701, 0.4308, 0.3839], std=[0.2595, 0.2522, 0.2541]))

        trans = torchvision.transforms.Compose(transform)
        with Image.open(os.getcwd() + '/' + self.data_path + '/' + image_name) as im:
            output = (trans(im), LABEL_NAMES.index(self.label_data[idx][1]))
            return output


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
