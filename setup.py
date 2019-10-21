import os
import sys
import random
import shutil
import argparse

import utils


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--dataset-dir', default='dataset', type=str, metavar='DIR')
    parser.add_argument('--nolabel-dir', default='nolabel', type=str, metavar='DIR')
    parser.add_argument('--label-master', default='label_master.tsv', type=str, metavar='FILE')
    parser.add_argument('--train-master', default='train_master.tsv', type=str, metavar='FILE')
    parser.add_argument('--validate_ratio', default=0.2, type=float, help='validate (default==>2/10)')
    parser.parse_args()
    args = parser.parse_args()

    nolabel_dir = os.path.join(args.dataset_dir, args.nolabel_dir)
    train_dir = os.path.join(args.dataset_dir, 'train')
    validate_dir = os.path.join(args.dataset_dir, 'validate')
    
    print('==> Getting all labels..')
    all_labels = get_all_labels(os.path.join(args.dataset_dir, args.label_master))
    label_div = dict()
    for label in all_labels:
        label_div[label] = list()
    
    print('==> Getting dataset information..')
    ds_info = get_dataset_info(os.path.join(args.dataset_dir, args.train_master))


    for file_name, label in ds_info:
        label_div[label].append([file_name, label])

    print('==> Deviding a dataset..')
    train_ds = list()
    validate_ds = list()
    for _, dataset in label_div.items():
        n_ds = len(dataset)
        div_i = n_ds - int(n_ds * args.validate_ratio)
        if args.shuffle:
            random.shuffle(dataset)
        train_ds.extend(dataset[:div_i])
        validate_ds.extend(dataset[div_i:])

    print('==> Moving a dataset..')
    move_files(train_ds, nolabel_dir, train_dir)
    move_files(validate_ds, nolabel_dir, validate_dir)


def get_all_labels(master_file, start_line=1):
    labels = []
    with open(master_file) as f:
        lines = f.read().split('\n')
        lines = lines[start_line:]

        for line in lines:
            if line:
                labels.append(line.split()[0])

    return labels


def get_dataset_info(master_file, start_line=1):
    ds_info = []
    with open(master_file) as f:
        lines = f.read().split('\n')
        lines = lines[start_line:]
        
        for line in lines:
            if line:
                ds_info.append(line.split())

    return ds_info


if __name__ == '__main__':
    setup()
