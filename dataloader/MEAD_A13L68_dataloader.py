import os
from os.path import join
import random
import json
import argparse
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", type=str, default='/root/TalkingHead/dataset')
parser.add_argument("--num_thread", type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_epoches', type=int, default=500)
parser.add_argument("--learning_rate", type=int, default=1e-4)

class MEAD_A13L68_Dataset(Dataset):
    def __init__(self, path):
        self.data_path = []
        with open(path, 'r') as f:
            self.data_path = json.load(f)

    def __getitem__(self, index):
        parts = self.data_path[index].split('|')
        mfcc_path, lm_path = parts[0], parts[1]
        
 
    def __len__(self):
        return len(self.data_path)

if __name__ == '__main__':
    args = parser.parse_args()
    train_path = join(args.dataset_folder, 'train_MEAD_A13L68.json')
    train_dataset = MEAD_A13L68_Dataset(train_path)

