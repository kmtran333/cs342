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

    # img = Image.open("test_img.jpg")
    # convert = transforms.ToTensor()
    # img_tensor = convert(img)
    #
    # det = Detector()
    # d = det.detect(img_tensor)
    # print(d)
    #
    # i=0
    # for each_list in d:
    #     for each_tup in each_list:
    #         cx = each_tup[1]
    #         cy = each_tup[2]
    #         print(cx, cy, img_tensor[i][cy][cx])
    #     i += 1

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

            gts0 = gts[0]
            det0 = detections[0]
            lbl = torch.as_tensor(gts0.astype(float), dtype=torch.float32).view(-1, 4)
            d = torch.as_tensor(det0, dtype=torch.float32).view(-1, 5)
            sz = abs(lbl[:, 2] - lbl[:, 0]) * abs(lbl[:, 3] - lbl[:, 1])

            is_close = point_in_box

            self_det = []
            all_pair_is_close = is_close(d[:, 1:], lbl)
            print(lbl)
            print(d)
            print(sz)

            print(all_pair_is_close)

            sys.exit()
            if len(d):
                detection_used = torch.zeros(len(d))
                for i in range(len(lbl)):
                    if sz[i] >= self.min_size:
                        # Find a true positive
                        s, j = (d[:, 0] - 1e10 * detection_used - 1e10 * ~all_pair_is_close[:, i]).max(dim=0)
                        if not detection_used[j] and all_pair_is_close[j, i]:
                            detection_used[j] = 1
                            self_det.append((float(s), 1))





