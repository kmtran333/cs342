import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from .models import Detector, save_model
from .utils import load_detection_data, DetectionSuperTuxDataset
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.nn.functional as F

from os import path
import sys

if __name__ == '__main__':


    model = Detector()

    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th'),
                                     map_location=torch.device('cpu')))


    def point_in_box(pred, lbl):
        px, py = pred[:, None, 0], pred[:, None, 1]
        x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
        return (x0 <= px) & (px < x1) & (y0 <= py) & (py < y1)

    model.eval()
    for img, *gts in DetectionSuperTuxDataset('dense_data/valid', min_size=0):
        with torch.no_grad():
            detections = model.detect(img)

            sys.exit()







