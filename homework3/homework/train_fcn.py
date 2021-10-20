import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb

import torchvision


def train(args):
    from os import path
    model = FCN()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """

    optimizer = torch.optim.Adam(model.parameters())

    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)

    loss = torch.nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        train_cm = ConfusionMatrix()
        train_accuracy = []
        train_iou = []
        loss_data = []

        for data, label in train_data:
            if device is not None:
                data, label = data.to(device), label.to(device).long()

            o = model(data)
            loss_val = loss(o, label)

            train_cm.add(o.argmax(1), label)
            train_accuracy.append(train_cm.global_accuracy.detach().cpu().numpy())
            train_iou.append(train_cm.iou.detach().cpu().numpy())
            loss_data.append(loss_val.detach().cpu().numpy())

            train_logger.add_scalar('loss', float(loss_val.detach().cpu().numpy()), global_step=global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        train_logger.add_scalar('accuracy', np.mean(train_accuracy), global_step=global_step)
        train_logger.add_scalar('iou', np.mean(train_iou), global_step=global_step)

        model.eval()
        valid_cm = ConfusionMatrix()
        valid_accuracy = []
        valid_iou = []
        for data, label in valid_data:
            if device is not None:
                data, label = data.to(device), label.to(device)
            o = model(data)
            valid_cm.add(o.argmax(1), label)
            valid_accuracy.append(valid_cm.global_accuracy.detach().cpu().numpy())
            valid_iou.append(valid_cm.iou.detach().cpu().numpy())

        if args.schedule_lr:
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            scheduler.step(np.mean(valid_accuracy))

        valid_logger.add_scalar('accuracy', np.mean(valid_accuracy), global_step=global_step)
        valid_logger.add_scalar('iou', np.mean(valid_iou), global_step=global_step)

        print(epoch, np.mean(train_accuracy), np.mean(valid_accuracy), np.mean(train_iou), np.mean(valid_iou))

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-sl', '--schedule_lr', action='store_true')

    args = parser.parse_args()

    transforms = [dense_transforms.RandomHorizontalFlip(),
                  dense_transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.7, hue=0.4),
                  dense_transforms.ToTensor()]

    transforms = dense_transforms.Compose(transforms)

    train_data = load_dense_data('dense_data/train', transform=transforms)
    valid_data = load_dense_data('dense_data/valid')

    train(args)
