import os
import time
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test-path', default=os.path.join('dataset', 'test'), metavar='DIR')
    parser.add_argument('-w', '--weights', default='', type=str, metavar='PATH')
    parser.add_argument('--move-files', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Using: ' + device)

    print('==> Preparing dataset..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_ds = utils.ImageFolderWithPath(
        args.test_path,
        transforms.Compose([
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            normalize,
        ])
    )

    
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print('==> Building model..')
    model = torch.load(args.weights)

    print('==> Inference..')
    predicted_results = test(model, test_loader, device)

    with open('log.tsv', 'w') as f:
        for path, label in zip(*predicted_results):
            file_name = os.path.basename(path)
            label = str(label)
            print('File->' + file_name + ' : Label->' + label)
            if args.log:
                f.write(file_name + '\t' + label + '\n')

    if args.move_files:
        move_files(predicted_results, os.path.join(args.test_path, 'nolabel'), args.test_path)



def test(model, test_loader, device):
    model.eval()
    predicted_results = [[], []]
    with torch.no_grad():
        for images, targets, paths in tqdm(test_loader):
            images = images.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            predicted_results[0].extend(paths)
            predicted_results[1].extend(predicted.tolist())
    return predicted_results


if __name__ == '__main__':
    main()
