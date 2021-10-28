import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.nn.functional as F

if __name__ == '__main__':
    # transforms = [dense_transforms.RandomHorizontalFlip(),
    #               dense_transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.7, hue=0.4),
    #               dense_transforms.ToTensor(),
    #               dense_transforms.ToHeatmap()]
    #
    # transforms = dense_transforms.Compose(transforms)
    #
    # train_data = load_detection_data('dense_data/train', transform=transforms)
    #
    # for data, label, size in train_data:
    #     det = Detector()
    #     label0 = label[0]
    #     d = det.detect(label0)
    #     print(d)
    #
    #     i=0
    #     for each_list in d:
    #         for each_tup in each_list:
    #             cx = each_tup[1]
    #             cy = each_tup[2]
    #             print(cx, cy, label0[i][cy][cx])
    #         i += 1
    #     break

    img = Image.open("test_img.jpg")
    convert = transforms.ToTensor()
    img_tensor = convert(img)

    det = Detector()
    d = det.detect(img_tensor)
    print(d)

    i=0
    for each_list in d:
        for each_tup in each_list:
            cx = each_tup[1]
            cy = each_tup[2]
            print(cx, cy, img_tensor[i][cy][cx])
        i += 1


