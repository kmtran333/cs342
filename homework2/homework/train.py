from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
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
    data_loaded = load_data('data/train').dataset
    train_data, train_label = [], []
    for i in range(len(data_loaded)):
        d, l = data_loaded[i]
        train_data.append(d)
        train_label.append(l)
    train_data = torch.stack(train_data)
    train_label = torch.Tensor(train_label).int()

    data_loaded_val = load_data('data/valid').dataset
    valid_data, valid_label = [], []
    for i in range(len(data_loaded_val)):
        d, l = data_loaded_val[i]
        valid_data.append(d)
        valid_label.append(l)
    valid_data = torch.stack(valid_data)
    valid_label = torch.Tensor(valid_label).int()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

    loss = torch.nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(args.n_epochs):
        permutation = torch.randperm(train_data.size(0))

        train_accuracy = []
        valid_accuracy = []
        for it in range(0, len(permutation) - args.batch+1, args.batch):
            batch_samples = permutation[it:it + args.batch]
            batch_data, batch_label = train_data[batch_samples].to(device), train_label[batch_samples].to(device)

            o = model.forward(batch_data)
            loss_val = loss(o, batch_label.long())
            acc_val = accuracy(o, batch_label)

            train_accuracy.append(acc_val.detach().cpu().numpy())

            train_logger.add_scalar('loss', float(loss_val.detach().cpu().numpy()), global_step=global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        train_logger.add_scalar('accuracy', np.mean(train_accuracy), global_step=global_step)

        valid_pred = model.forward(valid_data.to(device))
        valid_accuracy.append(accuracy(valid_pred, valid_label))
        valid_logger.add_scalar('accuracy', np.mean(valid_accuracy), global_step=global_step)
        print(epoch)
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    train(args)
