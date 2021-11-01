import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, gamma=2., reduction='none'):
        torch.nn.Module.__init__(self)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def train(args):
    from os import path
    model = Detector()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # loss = FocalLoss()
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0/0.02929112, 1.0/0.0044619, 1.0/0.00411153]))
    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        loss_data = []

        for data, label, size in train_data:
            if device is not None:
                data, label, size = data.to(device), label.to(device), size.to(device)
            o = model(data)

            loss_val = loss.forward(o, label)

            loss_data.append(loss_val.detach().cpu().numpy())

            train_logger.add_scalar('loss', float(loss_val.detach().cpu().numpy()), global_step=global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        log(train_logger, data, label, o, global_step=global_step)
        print(epoch)
    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    parser.add_argument('-n', '--n_epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-sl', '--schedule_lr', action='store_true')

    args = parser.parse_args()

    transforms = [dense_transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1),
                  dense_transforms.RandomHorizontalFlip(),
                  dense_transforms.ToTensor(),
                  dense_transforms.ToHeatmap()]

    transforms = dense_transforms.Compose(transforms)

    train_data = load_detection_data('dense_data/train', transform=transforms)
    valid_data = load_detection_data('dense_data/valid')
    train(args)
