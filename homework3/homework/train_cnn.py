from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
from torchvision import transforms
import torch.utils.tensorboard as tb
import numpy as np


def train(args):
    from os import path
    model = CNNClassifier()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code

    """

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters())

    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)

    loss = torch.nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        train_cm = ConfusionMatrix(6)
        train_accuracy = []
        loss_data = []

        for data, label in train_data:
            if device is not None:
                data, label = data.to(device), label.to(device)

            o = model(data)
            loss_val = loss(o, label)

            train_cm.add(o.argmax(1), label)
            train_accuracy.append(train_cm.global_accuracy.detach().cpu().numpy())
            loss_data.append(loss_val.detach().cpu().numpy())

            train_logger.add_scalar('loss', float(loss_val.detach().cpu().numpy()), global_step=global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        train_logger.add_scalar('accuracy', np.mean(train_accuracy), global_step=global_step)

        model.eval()
        valid_cm = ConfusionMatrix(6)
        valid_accuracy = []
        for data, label in valid_data:
            if device is not None:
                data, label = data.to(device), label.to(device)
            o = model(data)
            valid_cm.add(o.argmax(1), label)
            valid_accuracy.append(valid_cm.global_accuracy.detach().cpu().numpy())

        if args.schedule_lr:
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            scheduler.step(np.mean(valid_accuracy))

        valid_logger.add_scalar('accuracy', np.mean(valid_accuracy), global_step=global_step)

        print(epoch, np.mean(train_accuracy), np.mean(valid_accuracy))
    save_model(model)


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

    transforms = [torchvision.transforms.RandomHorizontalFlip(),
                  torchvision.transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.7, hue=0.4),
                  transforms.ToTensor()]

    transforms = torchvision.transforms.Compose(transforms)

    train_data = load_data('data/train', transform=transforms)
    valid_data = load_data('data/valid')

    train(args)
