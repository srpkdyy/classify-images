import os
import sys
import copy
import time
import numpy
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms, models

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
    parser.add_argument('--save', action='store_true')
    parser.add_argument('-lrd', '--lr-decay', default=0.1, type=float)
    parser.add_argument('-ss', '--step-size', default=30, type=int)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Device: ' + device)
    
    
    print('==> Preparing dataset..')
    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_ds = datasets.ImageFolder(
        os.path.join(args.dataset, 'train'),
        transforms.Compose([
            #transforms.RandomCrop(84),
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_ds = datasets.ImageFolder(
        os.path.join(args.dataset, 'validate'),
        transforms.Compose([
            #transforms.CenterCrop(84),
            transforms.Resize(image_size),
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

    print('==> Building model..')
    #model = VGG11()
    #model = VGG13()
    #model = VGG16()
    #model = VGG19()
    #model = CNN()
    #model = GAPVGG()
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
    model.to(device)
    if device == 'cuda':
        model = nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()
    params_to_update = []
    for name, param in model.named_parameters():
        if name in ['module.classifier.6.weight', 'module.classifier.6.bias']:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    optimizer = optim.SGD(
        params_to_update, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.lr_decay)

    print('==> Training model')
    start = time.time()
    best_cfg = None
    if args.save:
        best_cfg = {
            'loss': float('inf'),
            'acc':0.,
            'model': None,
        }
    for epoch in range(args.epochs):
        train(model, train_loader, epoch, optimizer, criterion, device, start)
        validate(model, val_loader, criterion, device, start, best_cfg)

        scheduler.step()

    if args.save:
        print('\n==> Save model')
        print('    Loss: %f' % best_cfg['loss'])
        print('    Acc: %f' % best_cfg['acc'])

        torch.save(
            best_cfg['model'],
            os.path.join('weights', 'acc{}.pth'.format(int(best_cfg['acc']*1000)))
        )


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


def validate(model, val_loader, criterion, device, start, best_cfg):
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

    if best_cfg is not None and correct/total > best_cfg['acc']:
        sys.stdout.write(' --> Update check point')
        best_cfg['loss'] = val_loss
        best_cfg['acc'] = correct/total
        best_cfg['model'] = copy.deepcopy(model)





def disp_progress(mode, i, n, loss, correct, total, elpased):
    i += 1
    sys.stdout.write('\r%s: %d/%d==> Loss: %.6f | Acc: %.3f%% (%d/%d) | Elapsed: %.2fsec'
        % (mode, i, n, loss/i, 100.*correct/total, correct, total, elpased))


if __name__ == '__main__':
    main()
