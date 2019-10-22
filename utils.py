import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets


def init_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.constant(m.bias, 0)


def move_files(dataset, in_dir, out_dir):
    n_all_files = len(dataset)
    n_moved_file = 0

    for file_name, label in dataset:
        save_path = os.path.join(out_dir, label)
        os.makedirs(save_path, exist_ok=True)
        shutil.move(os.path.join(in_dir, file_name), save_path)

        n_moved_file += 1
        sys.stdout.write('\rMoved: %d/%d' % (n_moved_file, n_all_files))


class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPath, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
 