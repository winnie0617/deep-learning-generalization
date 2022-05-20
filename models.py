from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from sklearn.model_selection import ParameterGrid
from math import log10
import numpy as np

# transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#     # values from https://github.com/kuangliu/pytorch-cifar/issues/19
#     ])
# dataset1 = datasets.CIFAR10('../data', train=True, download=True,
#     transform=transform)
# dataset2 = datasets.CIFAR10('../data', train=False,
#     transform=transform)

# #input_size = 32 = W
# #kernel_size = 3 = K
# #padding - 0 = P
# #stride = 1 = S
# #(W-K+2P)/S + 1 -->
# class Net(nn.Module):
#     def __init__(self, dp):
#         #I decreased the channel numbers significantly for faster debugging.
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 3, 3, 1)
#         self.conv2 = nn.Conv2d(3, 3, 3, 1)
#         self.dropout1 = nn.Dropout(dp)
#         self.dropout2 = nn.Dropout(dp)
#         self.fc1 = nn.Linear(588, 20)  # 3 x 14 x 14
#         self.fc2 = nn.Linear(20, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


class Net(nn.Module):
    def __init__(self, cin, width, depth, dropout):
        super(Net, self).__init__()
        self.layer = make_NiN(cin, width, depth, dropout)

    def forward(self, x):
        out = self.layer(x)
        return out


def train(model, device, train_loader, optimizer, epoch, log_interval, dry_run):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break
    train_loss = running_loss / len(train_loader)
    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss


def make_NiN(cin, width, depth, dropout):
    """Return network in network model with given width, depth and dropout
    cin: number of input channels"""

    def nin_block(in_channels, out_channels, kernel_size, strides, padding, dropout):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

    modules = []
    # First 2 blocks
    modules.extend(nin_block(cin, width, 3, 2, padding=2, dropout=dropout))
    modules.extend(nin_block(width, width, 3, 2, padding=2, dropout=dropout))
    # More blocks if needed
    i = depth - 2
    while i > 0:
        modules.extend(nin_block(width, width, 3, 2, padding=2, dropout=dropout))
        modules.extend(nin_block(width, width, 3, 2, padding=2, dropout=dropout))
        i -= 2

    modules.extend(
        [
            nn.Conv2d(width, 10, 1, 1),  # There are 10 label classes
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            # Transform the four-dimensional output into two-dimensional output with a
            # shape of (batch size, 10)
            nn.Flatten(),
            nn.LogSoftmax(dim=1),
        ]
    )

    return nn.Sequential(*modules)


def get_model(
    hp,
    dataset1,
    dataset2,
    cin,
    test_batch_size=1000,
    gamma=0.7,
    dry_run=False,
    log_interval=10,
    # save_model=False,
):
    """
    hp - dictionary: batch_size, lr
    """
    train_kwargs = {"batch_size": hp["batch_size"]}
    test_kwargs = {"batch_size": test_batch_size}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    net = Net(cin, hp["width"], hp["depth"], hp["dropout"])
    model = net.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=hp["lr"])

    summary(model, input_size=(cin, 32, 32))

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    train_loss = 0
    for epoch in range(1, hp["epochs"] + 1):
        train_loss += train(
            model, device, train_loader, optimizer, epoch, log_interval, dry_run
        )
        scheduler.step()

    train_loss /= hp["epochs"]
    test_loss = test(model, device, test_loader)
    return model, train_loss, test_loss


def get_models(hp_list, dataset, seed=1):
    """
    Given lists of hyperparameters, provide list of all models with their their associated training & testing loss

    w: list of network layer widths (tbd)
    d: list of network depth choices (tbd)
    op: list of optimizer choices (tbd)
    """

    # Load data
    torch.manual_seed(seed)

    if dataset == "CIFAR10":
        print("===== Using CIFAR10 Dataset =====")
        # values from https://github.com/kuangliu/pytorch-cifar/issues/19
        mean = (0.4914, 0.4822, 0.4465)
        sd = (0.247, 0.243, 0.261)
        torch_ds = datasets.CIFAR10
        cin = 3

    elif dataset == "MNIST":
        print("===== Using MNIST Dataset =====")
        mean = (0.1307,)
        sd = (0.3081,)
        torch_ds = datasets.MNIST
        cin = 1

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, sd)]
    )

    dataset1 = torch_ds("../data", train=True, download=True, transform=transform)
    dataset2 = torch_ds("../data", train=False, transform=transform)

    # TODO: REMOVE - takes the first 5% images of train set
    from torch.utils.data import Subset

    dataset1 = Subset(dataset1, indices=range(len(dataset1) // 20))

    # Get model per each hyperparameter combo
    model_list = []
    train_loss_list = []
    test_loss_list = []

    grid = list(ParameterGrid(hp_list))
    for hp in grid:
        print(hp)
        model, train_loss, test_loss = get_model(hp, dataset1, dataset2, cin)
        model_list.append(model)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    return grid, model_list, train_loss_list, test_loss_list


def get_weights(model):
    """Return parameters of the neural network"""
    # TODO: doesn't work for NiN
    return model.fc2.weight
