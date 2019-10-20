import os
import time
import argparse
import torch
import torch.nn as nn

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test-path', default=os.path.join('dataset', 'test'), metavar='DIR')
    parser.add_argument('model-path', default='', type=str, metavar='PATH')
    parser.add_argument('--move-files', action='store_true')
    parser.add_argument('--disp', action='store_true')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Using: ' + device)

    print('==> Preparing dataset..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_ds = datasets.ImageFolder(
        args.test_path,
        transforms.Compose([
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
    model = torch.load(model_path)

    print('==> Inference')
    predicted_labels = test(model, test_loader, device, time.time())

    test_ds[1] = predicted_labels

    for i, (file_name, label) in test_ds:
        print('File: ' + file_name + 'Label: ' + label)

    if(args.move_files)
        move_files(test_ds, os.path.join(args.test_path, 'nolabel'), args.test_path)



def test(model, val_loader, device, start):
    model.eval()
    predicted_labels = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            predicted_labels.append(predicted)
            now = time.time()
    return predicted_labels


if __name__ == '__main__':
    main()
