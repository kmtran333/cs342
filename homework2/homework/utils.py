from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import os

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """
        self.data_path = dataset_path
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
        image_name = self.label_data[idx][0]
        trans = transforms.Compose([transforms.ToTensor()])

        with Image.open(os.getcwd() + '/' + self.data_path + '/' + image_name) as im:
            output = (trans(im), LABEL_NAMES.index(self.label_data[idx][1]))
            return output


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
