import time

import numpy as np
import torch
import torch.utils.tensorboard as tb

from image_agent.image_net import ImageNet, save_model
from image_agent.utils import SuperTuxDataset

from . import dense_transforms


def train():
    torch.cuda.empty_cache()

    # Setup CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    torch.backends.cudnn.benchmark = True
    log_dir = 'logs'
    model = ImageNet().to(device)

    transform = dense_transforms.Compose([
        dense_transforms.ColorJitter(brightness=0.9,
                                     contrast=0.9,
                                     saturation=0.9,
                                     hue=0.1),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor(),
    ])

    train_batch_size = 500
    valid_batch_size = 200

    data_path = 'data'
    dataset = SuperTuxDataset(data_path, transform=transform)

    train_per = .7
    train_amount = round(len(dataset) * train_per)
    val_amount = len(dataset) - train_amount
    print('train_amount: ', train_amount)
    print('val_amount: ', val_amount)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_amount, val_amount])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=8,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=valid_batch_size,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=4,
                                             pin_memory=True)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 250

    logger = tb.SummaryWriter(log_dir + '/%s' % (time.time()), flush_secs=1)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    global_step = 0
    for epoch in range(epochs):
        print(optimizer.param_groups[0]['lr'], weight_decay, epoch)

        # Incremental Saving
        if epoch % 25 == 0:
            save_model(model)

        # Training Step
        train_epoch_loss = 0.0
        model.train(True)
        for (input, target) in train_loader:
            input = input.to(device)
            target = target.type(torch.FloatTensor).to(device)
            
            output = model(input)
            loss = loss_fn(output, target)
            train_epoch_loss += loss.item()

            log(logger, input, target, output, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # Validation Step
        val_epoch_loss = 0.0
        model.train(False)
        for (input, target) in val_loader:
            input = input.to(device)
            target = target.type(torch.FloatTensor).to(device)

            output = model(input)
            loss = loss_fn(output, target)
            val_epoch_loss += loss.item()

            optimizer.zero_grad()

        train_loss = train_epoch_loss / train_dataset.__len__()
        val_loss = val_epoch_loss / val_dataset.__len__()
        logger.add_scalars("loss", {
            'train': train_loss,
            'val': val_loss,
        }, global_step)

        scheduler.step(train_loss)
        scheduler.step(val_loss)

    save_model(model)


def log(logger, img, label, pred, global_step):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF

    label = label[0].cpu().detach().numpy()
    pred = pred[0].cpu().detach().numpy()
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2

    # Puck
    ax.add_artist(
        plt.Circle(WH2 * (label[0:2] + 1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(
        plt.Circle(WH2 * (pred[0:2] + 1), 2, ec='r', fill=False, lw=1.5))

    # Goal
    # ax.add_artist(
    #     plt.Circle(WH2 * (label[2:] + 1), 2, ec='b', fill=False, lw=1.5))
    # ax.add_artist(
    #     plt.Circle(WH2 * (pred[2:] + 1), 2, ec='y', fill=False, lw=1.5))

    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    train()
