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


class Net(nn.Module):
    def __init__(self, dp):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(dp)
        self.dropout2 = nn.Dropout(dp)
        self.fc1 = nn.Linear(12544, 128)  # 64 x 14 x 14
        self.fc2 = nn.Linear(128, 10)

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


def get_model(
    hp,
    epochs=64,
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

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            # values from https://github.com/kuangliu/pytorch-cifar/issues/19
            ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
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
    

def get_weights(model):
    ''' Return parameters of the neural network'''
    return model.fc2.weight.data.numpy()
    


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
