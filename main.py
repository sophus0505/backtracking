import numpy as np

import time

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn

import torch.optim as optim

import torchvision.transforms as transforms

from resnet import ResNet, BasicBlock, Bottleneck, ResNet18
from backtracking import LRFinder, change_lr

from utils import *


def download_data():
    """Download the CIFAR10 dataset"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return trainloader, testloader


def train(epoch, optimizer, samples=100):
    global best_loss, loss_avg, history, patient_test, patient_train, patient
    train_loss = correct = total = 0
    pation = min([patient_test, patient_train])
    print(
        f'Training started, using optimizer {type (optimizer).__name__} for {samples} iterations.\n')

    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx > samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        loss_avg = train_loss/(batch_idx+1)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)| LR: %.7f'
                     % (loss_avg, acc, correct, total, optimizer.param_groups[0]['lr']))

    print(f'Finnished 1 epoch! Loss: {loss_avg:.4f}, Acc: {acc:.4f}')


def test(epoch, optimizer, samples=100):
    global history, patient_train, patient_test, best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx > samples:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total
            loss_avg = test_loss/(batch_idx + 1)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (loss_avg, 100.*correct/total, correct, total))

        print(
            f'Finished testing!  Loss: {loss_avg:.4f}, Acc: {acc:.4f}, Num. samples: {samples}')


def run_AdaGrad(lr_start=0.1, samples=200):
    """Runs the model with the AdaGrad optimizer.

    Args:
        lr_start (float, optional): The initial learning rate. Defaults to 0.1.
        samples (int, optional): The number of iterations. Defaults to 200.
    """

    optimizer = optim.Adagrad(net.parameters(), lr=lr_start)

    start1 = time.time()
    train(1, optimizer, samples)
    end1 = time.time()
    time1 = end1 - start1

    test(1, optimizer)

    print(f'\nTime taken to train model: ~ {time1:.0f} s.')


def run_backtracking(lr_start=0.1, samples=200, device_='cpu'):
    """Runs the model with the two-way backtracking optimizer.

    Args:
        lr_start (float, optional): The initial learning rate. Defaults to 0.1.
        samples (int, optional): The number of iterations. Defaults to 200.
    """
    optimizer_BT = optim.SGD(net.parameters(), lr=lr_start)
    lr_finder_BT = LRFinder(net, optimizer_BT, criterion, device=device_)
    optimizer = optimizer_BT

    start2 = time.time()
    train(1, optimizer, samples)
    end2 = time.time()
    time2 = end2 - start2
    print(f'\nTime taken to train model: ~ {time2:.0f} s.')

    test(1, optimizer)


if __name__ == "__main__":

    # download the data from CIFAR10
    cifar_dataset = 10  # CIFAR100 or 100
    batch_size = 4
    lr_start = 1e-5  # start learning rate

    # Data
    trainloader, testloader, num_batches = dataset(cifar_dataset, batch_size)
    num_classes = cifar_dataset

    # initialize the model

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_classes = 10  # CIFAR10
    net = ResNet18(num_classes)
    net_name = 'ResNet18 '
    print('Model:', net_name)
    print('Number of parameters:', count_parameters(net),
          'numbers of Layers:', len(list(net.parameters())))

    # Train and test the models

    lr_start = 0.1

    patient_train = 0
    patient_test = 0
    patient = 0
    best_acc = 0  # best test accuracy
    best_loss = loss_avg = 1e10  # best (smallest) training loss
    # Run on CUDA

    # CUDA device
    global device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    print(device)

    criterion = nn.CrossEntropyLoss()

    # run AdaGrad
    # run_AdaGrad()

    # Run backtracking GD
    run_backtracking(device_=device)
