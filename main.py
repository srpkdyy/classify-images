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
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=90, type=int, metavar='N')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, metavar='W')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ' + device)
    
    
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    train_ds = datasets.ImageFolder(
        os.path.join(args.data, 'train'),
        transforms.Compose([
            transforms.RandomCrop(28),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    test_ds = datasets.ImageFolder(
        os.path.join(args.data, 'test'),
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
    
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )


    print('==> Building model..')
    model = VGG16()
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    print('==> Training model..')
    for epoch in range(args.epochs):
        train(model, train_loader, epoch, optimizer, criterion, device)



def train(model, train_loader, epoch, optimizer, criterion, device):
    print('\nEpoch: %d' % epoch)

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    end = time.time()
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

        elapsed_sec = time.time() - end
        end = time.time()

        sys.stdout.write('\r%d/%d==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (i+1, len(train_loader), train_loss/(i+1), 100.*correct/total, correct, total))




if __name__ == '__main__':
    main()
