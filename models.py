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

def make_conv(cin, width, depth, dropout, pixel):
    """
    Returns a convolutional neural network of specified parameters 

    Parameters
        cin (int): input dimension
        width (int): width of a layer
        depth (int): number of layers of the network
        dropout (float): regularization probability
        pixel (int): dimension of convolution
    """
    layer = [
        nn.Conv2d(cin, width // 2, 3, 1),
        nn.ReLU(),
        nn.Conv2d(width // 2, width, 3, 1),
        nn.ReLU(),
    ]
    # (W-K+2P)/S + 1
    out_size = (pixel - 3) / 1 + 1  # after first conv
    out_size = (out_size - 3) / 1 + 1
    for _ in range(depth - 2):
        layer.extend([nn.Conv2d(width, width, 2, 1), nn.ReLU()])
        out_size = (out_size - 2) / 1 + 1

    layer.extend(
        [
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(int(width * (out_size /2)**2), 128),  # 64 x 14 x 14
            nn.Dropout(dropout),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        ]
    )
    return nn.Sequential(*layer)

def make_NiN(cin, width, depth, dropout):
    """
    Returns a network in network (NiN) model of specified parameters

    Parameters
        cin (int): input dimension
        width (int): width of a layer
        depth (int): number of NiN blocks of the network
        dropout (float): regularization probability
    """

    def nin_block(in_channels, out_channels, kernel_size, strides, padding, 
    dropout):
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

    class_label = 10
    modules.extend(
        [
            nn.Conv2d(width, class_label, 1, 1),  # There are 10 label classes
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            # Transform the four-dimensional output into two-dimensional output 
            # with a shape of (batch_size, class_label)
            nn.Flatten(),
            nn.LogSoftmax(dim=1),
        ]
    )

    return nn.Sequential(*modules)

class Net(nn.Module):
    """
    A class that defines a Net

    Attributes
        cin (int): input dimension 
        width (int): width of a layer in a net
        depth (int): number of layers of a net
        dropout (float): regularization probability
        model_name (str): one of "conv" or "NiN", which denotes the 
            architectural model of the net as either generic convolutional 
            network or Network In Network design
        pixel (int): dimension of convolution used
    """
    def __init__(self, cin, width, depth, dropout, model_name, pixel):
        super(Net, self).__init__()
        if model_name == "conv":
            maker = make_conv
        elif model_name == "NiN":
            maker = make_NiN
        self.layer = maker(cin, width, depth, dropout, pixel)

    def forward(self, x):
        out = self.layer(x)
        return out

def train(model, device, train_loader, optimizer, epoch, log_interval, dry_run):
    """
    Trains a model subject to specified parameters and returns the training 
    loss for each epoch 

    A printer is included for training loss descent in each epoch iteration

    Parameter 
        model: the untrained architecture to be trained
        device: the dataloader device to be used
        train_loader: the train loader to be used
        optimizer: the gradient descent optimizer to be used
        epoch: the number of epochs 
        log_interval: the number of batches for each epoch
        dry_run: run for one epoch if true, else continue. 
    """
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
    """
    Test a trained model and returns the testing loss

    Parameters
        model: the trained model
        device: the dataloader device to be used
        test_loader: the testing loader to be used 
    """
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

def get_model(hp, dataset1, dataset2, cin, model_name, pixel,
    test_batch_size=1000, gamma=0.7, dry_run=False, log_interval=10):
    """
    Returns the set of trained models and their correspoding training/testing
    loss

    Parameters
        hp: the hyperparameter space
        dataset1: the training dataset
        dataset2: the testing dataset
        cin: input dimension
        model_name: the model type to be used 
        pixel: dimension of convolution
        test_batch_size: the batch size on the testing set 
        gamma: a regularization parameter for an optimizer step
        dry_run: run for one epoch if true, else continue
        log_interval: the number of batches for each epoch
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
    net = Net(cin, hp["width"], hp["depth"], hp["dropout"], model_name, pixel)
    model = net.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=hp["lr"])
    summary(model, input_size=(cin, pixel, pixel))

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    train_loss = 0
    for epoch in range(1, hp["epochs"] + 1):
        train_loss += train(
            model, device, train_loader, optimizer, epoch, log_interval, dry_run
        )
        scheduler.step()

    # recalculate loss on fitted model
    train_loss = test(model, device, train_loader) 
    test_loss = test(model, device, test_loader)
    print(f"========== \n train loss: {train_loss}\n test loss: {test_loss}\n==========")
    return model, train_loss, test_loss

def get_models(hp_list, dataset, model_name, seed=1):
    """
    Given a hyperparameter space, return a list of training and testing loss 
    for each model in the hyperparameter space, as well as the list of 
    network architectural choices used  
    
    Parameters
        hp_list: a dictionary of specificed parameter list
        dataset: the dataset to be used
        model_name: the model to be used to construct the network
        seed: the seed for the optimizer
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
        pixel = 32
    elif dataset == "MNIST":
        print("===== Using MNIST Dataset =====")
        mean = (0.1307,)
        sd = (0.3081,)
        torch_ds = datasets.MNIST
        cin = 1
        pixel = 28
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, sd)]
    )

    dataset1 = torch_ds("../data", train=True, download=True, transform=transform)
    dataset2 = torch_ds("../data", train=False, transform=transform)
    # Get model per each hyperparameter permutation in hyperparameter space
    model_list = []
    train_loss_list = []
    test_loss_list = []
    grid = list(ParameterGrid(hp_list))
    for i, hp in enumerate(grid):
        print(f"{i}/{len(grid)} -- {hp}")
        model, train_loss, test_loss = get_model(
            hp, dataset1, dataset2, cin, model_name, pixel
        )
        model_list.append(model)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    return grid, model_list, train_loss_list, test_loss_list