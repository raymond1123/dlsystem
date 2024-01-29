import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    seq = nn.Sequential(nn.Linear(in_features=dim, out_features=hidden_dim), 
                        norm(dim=hidden_dim),
                        nn.ReLU(), 
                        nn.Dropout(p=drop_prob),
                        nn.Linear(in_features=hidden_dim, out_features=dim),
                        norm(dim=dim))

    res = nn.Residual(seq)
    #print()

    return nn.Sequential(res, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    seq_list = [nn.Linear(in_features=dim, out_features=hidden_dim),nn.ReLU()]
    for _ in range(num_blocks):
        res = ResidualBlock(dim=hidden_dim, 
                            hidden_dim=hidden_dim//2, 
                            norm=norm, drop_prob=drop_prob)
        seq_list.append(res)

    seq_list.append(nn.Linear(in_features=hidden_dim, out_features=num_classes))

    return nn.Sequential(*seq_list)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = nn.SoftmaxLoss()
    loss, error, terror, tloss = 0,0,0,0
    num_samples, num_iters = 0,0

    if opt:
        model.train()
        for x, y in dataloader:
            opt.reset_grad()
            batch_size, height, width, channel = x.shape
            x = x.reshape((batch_size, height*width*channel))
            logits = model(x)
            loss = loss_fn(logits, y)
            tloss += loss.numpy()
            error = np.sum(logits.numpy().argmax(axis=1)!=y.numpy())
            terror += error
            loss.backward()
            opt.step()
            num_samples += batch_size
            num_iters+=1
            #print(f'{loss=}, {error=}')
    else:
        model.eval()
        for x, y in dataloader:
            batch_size, height, width, channel = x.shape
            x = x.reshape((batch_size, height*width*channel))
            logits = model(x)
            loss = loss_fn(logits, y)
            tloss += loss.numpy()
            error = np.sum(logits.numpy().argmax(axis=1)!=y.numpy())
            terror += error
            num_samples += batch_size
            num_iters+=1
            #print(f'{loss=}, {error=}')

    return terror/num_samples, tloss/num_iters
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model = MLPResNet(dim=784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")

    train_dataloader = ndl.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)
    test_dataloader = ndl.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size,
                                           shuffle=False)
    for i in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model=model, opt=opt)
    test_err, test_loss = epoch(test_dataloader, model=model)

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
