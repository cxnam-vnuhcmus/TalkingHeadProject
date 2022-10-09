import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from glob import glob
from os.path import join

class Dataset_MEAD_A13L68(Dataset):
    def __init__(self, path):
        self.data_path = []
        with open(path, 'r') as f:
            self.data_path = json.load(f)

    def __getitem__(self, index):
        parts = self.data_path[index].split('|')
        mfcc_path, lm_path = parts[0], parts[1]

        mfcc_data_list, lm_data_list = self.read_data_from_path(mfcc_path, lm_path)

        return torch.from_numpy(mfcc_data_list), torch.from_numpy(lm_data_list)
    
    def read_data_from_path(self, mfcc_path, lm_path):
        mfcc_data_list = []
        mfcc_list = sorted(glob(join(mfcc_path, '*.npy')))
        for mfcc_file in mfcc_list:
            mfcc_data = np.load(mfcc_file)
            mfcc_data_list.append(mfcc_data)
        mfcc_data_list = np.vstack(mfcc_data_list)

        lm_data_list = []
        lm_list = sorted(glob(join(lm_path, '*.json')))
        for lm_file in lm_list:
            with open(lm_file, 'r') as f:
                lm_data = json.load(f)
            lm_data_list.append([lm_data])
        lm_data_list = np.vstack(lm_data_list).astype(np.float32)
        return mfcc_data_list, lm_data_list

    def __len__(self):
        return len(self.data_path)

class DataLoader_MEAD_A13L68(DataLoader):
    def __init__(self, config, is_train=True):
        self.config = config
        dataset_path = config['test_dataset_path']
        if is_train:
            dataset_path = config['train_dataset_path']
        import sys
        print(sys.path)
        self.dataset = Dataset_MEAD_A13L68(dataset_path)
        super(DataLoader_MEAD_A13L68, self).__init__(self.dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'],
                                shuffle=True,
                                drop_last=True,
                                collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) 

    def getDataset(self):
        return self.dataset

    def getStepPerEpoch(self):
        return np.ceil(len(self.dataset) / self.config['batch_size']).astype(np.int32)