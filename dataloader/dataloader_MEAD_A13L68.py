import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class Dataset_MEAD_A13L68(Dataset):
    def __init__(self, path):
        self.data_path = []
        with open(path, 'r') as f:
            self.data_path = json.load(f)

    def __getitem__(self, index):
        parts = self.data_path[index].split('|')
        mfcc_path, lm_path = parts[0], parts[1]

        mfcc_data = np.load(mfcc_path)

        with open(lm_path, 'r') as f:
            lm_data = json.load(f)

        return torch.from_numpy(mfcc_data), torch.tensor(lm_data, dtype=torch.float)
        
    def __len__(self):
        return len(self.data_path)

class DataLoader_MEAD_A13L68(DataLoader):
    def __init__(self, config):
        self.config = config
        self.dataset = Dataset_MEAD_A13L68(config['dataset_path'])
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