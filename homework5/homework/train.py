from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms


def train(args):
    from os import path
    model = Planner()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    loss = torch.nn.MSELoss()

    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        loss_data = []

        for data, label in train_data:
            if device is not None:
                data, label = data.to(device), label.to(device)

            o = model(data)

            loss_val = loss(o, label)

            loss_data.append(loss_val.detach().cpu().numpy())

            train_logger.add_scalar('loss', float(loss_val.detach().cpu().numpy()), global_step=global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        log(train_logger, data, label, o, global_step=global_step)

        if args.schedule_lr:
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            scheduler.step(np.mean(loss_data))
        print(epoch)

    save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

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
                  dense_transforms.ToTensor()]

    transforms = dense_transforms.Compose(transforms)

    train_data = load_data('drive_data', transform=transforms)
    train(args)
