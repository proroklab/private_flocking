import os
import random, shutil
import argparse

parser = argparse.ArgumentParser('replay')
parser.add_argument('--file_dir', type=str, default='../data')
parser.add_argument('--train_dir', type=str, default='../data/train')
parser.add_argument('--test_dir', type=str, default='../data/test')
parser.add_argument('--rate', type=float, default=0.3, help='test proportion of the dataset')


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def randomiseDataset(file_dir, train_dir, tar_dir, rate):
    makedirs(file_dir)
    path_dir = os.listdir(file_dir)
    file_number = len(path_dir)
    pick_number = int(file_number * rate)
    sample = random.sample(path_dir, pick_number)
    makedirs(tar_dir)
    for name in sample:
        if '.mat' in name:
            old_dir = file_dir + '/' + name
            new_dir = tar_dir + '/' + name
            shutil.move(old_dir, new_dir)

    new_path_dir = os.listdir(file_dir)
    makedirs(train_dir)
    for name in new_path_dir:
        if '.mat' in name:
            old_dir = file_dir + '/' + name
            new_dir = train_dir + '/' + name
            shutil.move(old_dir, new_dir)


def main():

    args = parser.parse_args()
    randomiseDataset(args.file_dir, args.train_dir, args.test_dir, args.rate)


if __name__ == '__main__':
    main()