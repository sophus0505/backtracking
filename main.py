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


history = {}
history['acc_train'] = []
history['loss_train'] = []
history['lr'] = []

all_history = []


def train(epoch):
    global best_loss, loss_avg, history, patient_test, patient_train, patient
    train_loss = correct = total = 0
    pation = min([patient_test, patient_train])
    # print(
    #     f'Training started, using optimizer {type (optimizer).__name__} for {samples} iterations.\n')

    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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

    history['acc_train'].append(acc)
    history['loss_train'].append(loss_avg)
    history['lr'].append(optimizer.param_groups[0]['lr'])


def test(epoch):
    global history, patient_train, patient_test, best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

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


def run_AdaGrad(lr_start=0.1, epochs=10, samples=200):
    """Runs the model with the AdaGrad optimizer.

    Args:
        lr_start (float, optional): The initial learning rate. Defaults to 0.1.
        samples (int, optional): The number of iterations. Defaults to 200.
    """

    optimizer = optim.Adagrad(net.parameters(), lr=lr_start)

    for epoch in range(epochs):
        train(epoch)

    test(1)


def run_backtracking(lr_start=0.1, samples=200, device_='cpu'):
    """Runs the model with the two-way backtracking optimizer.

    Args:
        lr_start (float, optional): The initial learning rate. Defaults to 0.1.
        samples (int, optional): The number of iterations. Defaults to 200.
    """

    # Backtracking hyper-parameters
    BT = 1  # using backtracking or not
    lr_justified = True
    alpha = 1e-4
    beta = 0.5
    num_iter = 20

    optimizer_BT = optim.SGD(net.parameters(), lr=lr_start)
    lr_finder_BT = LRFinder(net, optimizer_BT, criterion, device=device_)
    optimizer = optimizer_BT

    # train(1, optimizer, samples)

    # test(1, optimizer)
    optimizer_BT = optim.SGD(net.parameters(), lr=lr_start)
    print('Start learning rate:', optimizer_BT.param_groups[0]['lr'])
    lr_finder_BT = LRFinder(net, optimizer_BT, criterion, device="cuda")

    print("Using backtrack with", optimizer_BT.__class__.__name__,
          ", alpha =", alpha, ', beta =', beta)
    lr_finder_BT.backtrack(trainloader, alpha=alpha, beta=beta,
                           num_iter=num_iter, lr_justified=lr_justified)


if __name__ == "__main__":

    # download the data from CIFAR10
    cifar_dataset = 10  # CIFAR100 or 100
    batch_size = 200
    lr_start = 1e-2  # start learning rate

    # Data
    trainloader, testloader, num_batches = dataset(cifar_dataset, batch_size)
    num_classes = cifar_dataset

    # CUDA device
    global device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    print(device)

    # initialize the model

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_classes = 10  # CIFAR10
    net = ResNet18(num_classes)
    net_name = 'ResNet18 '
    net = net.to(device)
    print('Model:', net_name)
    print('Number of parameters:', count_parameters(net),
          'numbers of Layers:', len(list(net.parameters())))

    # Train and test the models

    patient_train = 0
    patient_test = 0
    patient = 0
    best_acc = 0  # best test accuracy
    best_loss = loss_avg = 1e10  # best (smallest) training loss

    criterion = nn.CrossEntropyLoss()

    # run AdaGrad for 10 epochs
    run_AdaGrad()

    # run backtracking for 10 epochs

    BT = 1  # using backtracking or not
    lr_justified = True
    alpha = 1e-4
    beta = 0.5
    num_iter = 20

    optimizer = optim.SGD(net.parameters(), lr=lr_start)
    lr_finder_BT = LRFinder(net, optimizer, criterion, device=device)

    for epoch in range(0, 10):
        lr_finder_BT.backtrack(trainloader, alpha=alpha, beta=beta,
                               num_iter=num_iter, lr_justified=lr_justified)
        train(epoch)

    print(history)

    # Run backtracking GD
    # run_backtracking(lr_start=0.1, samples=20_000, device_=device)
