from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch


def train(args):
    model = model_factory[args.model]()
    """
    Your code here

    """
    # Load data
    data_loaded = load_data(args.path).dataset
    train_data, train_label = [], []
    for i in range(len(data_loaded)):
        d, l = data_loaded[i]
        train_data.append(d)
        train_label.append(l)
    train_data = torch.stack(train_data)
    train_label = torch.Tensor(train_label).int()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

    loss = torch.nn.CrossEntropyLoss()

    # Start training
    global_step = 0
    for epoch in range(args.n_epochs):
        # Shuffle the data
        permutation = torch.randperm(train_data.size(0))

        # Iterate
        train_accuracy = []
        for it in range(0, len(permutation)-args.batch+1, args.batch):
            batch_samples = permutation[it:it+args.batch]
            batch_data, batch_label = train_data[batch_samples], train_label[batch_samples]

            # Compute the loss
            o = model.forward(batch_data)
            loss_val = loss(o, batch_label.long())

            # train_accuracy.extend(((o>0).long() == batch_label[:,None]).cpu().detach().numpy())
            train_accuracy.append(accuracy(o, batch_label))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('--n_epochs', type=int, default=250)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('-p', '--path', default='data/train')

    args = parser.parse_args()

    train(args)
