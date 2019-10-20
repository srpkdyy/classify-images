import os
import sys
import time
import numpy
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from models import *
    

def main():
    parser = argparse.ArgumentParser(description='Classify images by PyTorch')
    parser.add_argument('dataset', metavar='DIR', help='path to dataset')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, metavar='W')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('-lrd', '--lr-decay', default=0.2, type=float)
    parser.add_argument('-ss', '--step-size', default=30, type=int)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Device: ' + device)
    
    
    print('==> Preparing dataset..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_ds = datasets.ImageFolder(
        os.path.join(args.dataset, 'train'),
        transforms.Compose([
            transforms.RandomCrop(84),
            #transforms.RandomResizedCrop(96, scale=(0.5, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_ds = datasets.ImageFolder(
        os.path.join(args.dataset, 'validate'),
        transforms.Compose([
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    test_ds = datasets.ImageFolder(
        os.path.join(args.dataset, 'test'),
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )


    print('==> Building model..')
    #model = VGG11()
    #model = VGG13()
    model = VGG16()
    #model = VGG19()
    #model = CNN()
    model.to(device)
    if device == 'cuda':
        model = nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.lr_decay)

    print('==> Training model')
    start = time.time()
    for epoch in range(args.epochs):
        train(model, train_loader, epoch, optimizer, criterion, device, start)
        validate(model, val_loader, criterion, device, start)

        scheduler.step()


def train(model, train_loader, epoch, optimizer, criterion, device, start):
    print('\nEpoch: %d' % (epoch+1))

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        now = time.time()

        sys.stdout.write('\rTrain: %d/%d==> Loss: %.6f | Acc: %.2f%% (%d/%d) | Elapsed: %.2fsec'
            % (i+1, len(train_loader), train_loss/(i+1), 100.*correct/total, correct, total, now-start))


def validate(model, val_loader, criterion, device, start):
    print()
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            val_loss = loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            now = time.time()
            disp_progress('Validate', i, len(val_loader), val_loss, correct, total, now-start)


def disp_progress(mode, i, n, loss, correct, total, elpased):
    i += 1
    sys.stdout.write('\r%s: %d/%d==> Loss: %.6f | Acc: %.3f%% (%d/%d) | Elapsed: %.2fsec'
        % (mode, i, n, loss/i, 100.*correct/total, correct, total, elpased))

if __name__ == '__main__':
    main()
