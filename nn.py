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

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    # values from https://github.com/kuangliu/pytorch-cifar/issues/19
    ])
dataset1 = datasets.CIFAR10('../data', train=True, download=True,
    transform=transform)
dataset2 = datasets.CIFAR10('../data', train=False,
    transform=transform)

#input_size = 32 = W
#kernel_size = 3 = K
#padding - 0 = P
#stride = 1 = S
#(W-K+2P)/S + 1 -->  
class Net(nn.Module):
    def __init__(self, dp):
        #I decreased the channel numbers significantly for faster debugging.
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 3, 1)
        self.dropout1 = nn.Dropout(dp)
        self.dropout2 = nn.Dropout(dp)
        self.fc1 = nn.Linear(588, 20)  # 3 x 14 x 14
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


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
    train_loss=running_loss/len(train_loader)
    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss

def make_nn(width, depth, dropout):
    ''' Return network in network model with given width, depth and dropout'''

    def nin_block(in_channels, out_channels, kernel_size, strides, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1), nn.ReLU(),
            nn.Dropout(dropout))
        
    def stack_blocks(depth, cin):
        net = nn.Sequential(
            nin_block(cin, width, kernel_size=3, strides=2, dropout=dropout),
            nin_block(width, width, kernel_size=3, strides=2, dropout=dropout))
        i = depth - 2
        while i > 0:
            net.add(nin_block(width, width, kernel_size=3, strides=2))
            net.add(nin_block(width, width, kernel_size=3, strides=2))
            i -= 2
        return net

    cin = 3 # TODO: change for MNIST
    net = nn.Sequential(
        stack_blocks(depth, cin),
        # There are 10 label classes
        nn.Conv2d(width, 10, 1, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        # Transform the four-dimensional output into two-dimensional output with a
        # shape of (batch size, 10)
        nn.Flatten(),
        nn.LogSoftmax()
        )
    
    return net



def get_model(
    hp,
    epochs=64,
    width=2*96,
    depth=2,
    dropout=0.25,
    test_batch_size=1000,
    lr=1,
    gamma=0.7,
    no_cuda=False,
    dry_run=False,
    seed=1,
    log_interval=10,
    save_model=False,
):
    '''
    hp - dictionary: batch_size, lr
    '''
    torch.manual_seed(seed)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": hp["batch_size"]}
    test_kwargs = {"batch_size": test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    net = make_nn(width, depth, dropout)
    model = net.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=hp["lr"])

    summary(model, input_size=(3, 32, 32))

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    train_loss = 0
    for epoch in range(1, epochs + 1):
        train_loss += train(model, device, train_loader, optimizer, epoch, log_interval, dry_run)
        scheduler.step()

    train_loss /= epochs
    test_loss = test(model, device, test_loader)
    return model, train_loss, test_loss

def get_models(
    bs=[64], 
    lr=[1],
    epochs=[1],
    dp = [0.25],
    test_batch_size=1000,
    gamma=0.7,
    no_cuda=False,
    dry_run=False,
    seed=1,
    log_interval=10,
    save_model=False
    ):
    """
    Given lists of hyperparameters, provide list of all models with their their associated training & testing loss
    
    bs: list of batch size choices (done)
    lr: list of learning rate choices (done)
    epochs: list of epoch choices (done)
    dp: list of dropout probability choices (done)
    w: list of network layer widths (tbd)
    d: list of network depth choices (tbd)
    op: list of optimizer choices (tbd)
    """
    param_grid = {'batch':bs, 'lr':lr, 'epoch':epochs, 'dropout':dp}
    grid = list(ParameterGrid(param_grid))

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            # values from https://github.com/kuangliu/pytorch-cifar/issues/19
            ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform)

    model_list = []
    train_loss_list = []
    test_loss_list = []
    for hyper in grid:
        torch.manual_seed(seed)
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        train_kwargs = {"batch_size": hyper["batch"]}
        test_kwargs = {"batch_size": test_batch_size}
        if use_cuda:
            cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)  
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        
        model = Net(hyper['dropout']).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=hyper["lr"])
        summary(model, input_size=(3, 32, 32))
        scheduler = StepLR(optimizer, step_size=hyper["lr"], gamma=gamma)

        train_loss = 0
        for epoch in range(1, hyper["epoch"]+1):
            train_loss += train(model, device, train_loader, optimizer, epoch, log_interval, dry_run)
            scheduler.step()
        train_loss /= hyper["epoch"] 
        test_loss = test(model, device, test_loader)
        
        model_list.append(model)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    return model_list, train_loss_list, test_loss_list
    
def get_vc(model):
    """
    Returns VC dimension of a given model

    Tightest VC Dimension for a regular nn is O(WLlog(W)), where W = # of weights, L = # of layers (depth)
    """
    nn_weight_num = 0
    fc_weight_num = 0
    L = -1
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            #for linear activation, w_i = (input_i + 1)*output_i
            L += 1
            fc_weight_num += (layer.in_features+1)*layer.out_features
        elif isinstance(layer, nn.Conv2d):
            # in general, # of w_i = (kernel_size^2 * output_{i-1} + 1)*output_{i}, output_i = input_i-1
            # so total # of weights = \sum_{i} w_{i} 
            L += 1
            nn_weight_num += ((layer.kernel_size[0]**2)*layer.in_channels + 1)*layer.out_channels
    
    W = nn_weight_num + fc_weight_num
    print("Weight =" + str(W))
    print("Layer =" + str(L))
    return W*L*log10(L)

def get_weights(model):
    ''' Return parameters of the neural network'''
    return model.fc2.weight
    
# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=64,
#         metavar="N",
#         help="input batch size for training (default: 64)",
#     )
#     parser.add_argument(
#         "--test-batch-size",
#         type=int,
#         default=1000,
#         metavar="N",
#         help="input batch size for testing (default: 1000)",
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=14,
#         metavar="N",
#         help="number of epochs to train (default: 14)",
#     )
#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=1.0,
#         metavar="LR",
#         help="learning rate (default: 1.0)",
#     )
#     parser.add_argument(
#         "--gamma",
#         type=float,
#         default=0.7,
#         metavar="M",
#         help="Learning rate step gamma (default: 0.7)",
#     )
#     parser.add_argument(
#         "--no-cuda", action="store_true", default=False, help="disables CUDA training"
#     )
#     parser.add_argument(
#         "--dry-run",
#         action="store_true",
#         default=False,
#         help="quickly check a single pass",
#     )
#     parser.add_argument(
#         "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
#     )
#     parser.add_argument(
#         "--log-interval",
#         type=int,
#         default=10,
#         metavar="N",
#         help="how many batches to wait before logging training status",
#     )
#     parser.add_argument(
#         "--save-model",
#         action="store_true",
#         default=False,
#         help="For Saving the current Model",
#     )
#     args = parser.parse_args()

#     if args.save_model:
#         torch.save(model.state_dict(), "mnist_cnn.pt")


# if __name__ == "__main__":
#     main()
