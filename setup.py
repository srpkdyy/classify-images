import os
import shutil
import argparse

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', default='dataset', type=str, metavar='DIR')
    parser.add_argument('-t', '--train-dir', default='train', type=str, metavar='DIR')
    parser.add_argument('-i', '--images-dir', default='unknown', type=str, metavar='DIR')
    parser.add_argument('-m', '--master-file', default='train_master.tsv', type=str)
    parser.parse_args()
    args = parser.parse_args()

    train_dir = os.path.join(args.dataset_dir, args.train_dir)
    images_dir = os.path.join(train_dir, args.images_dir)
    
    ds_info = get_dataset_info(os.path.join(args.dataset_dir, args.master_file))

    for file_name, label in ds_info:
        dest_dir = os.path.join(train_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(os.path.join(images_dir, file_name), dest_dir)


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
