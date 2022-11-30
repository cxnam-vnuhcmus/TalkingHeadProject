import json
from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob
from os.path import join

class MeadDataset(Dataset):
    def __init__(self, path=None):
        self.data_path = []
        if path is not None:
            with open(path, 'r') as f:
                self.data_path = json.load(f)

    def __getitem__(self, index):
        parts = self.data_path[index].split('|')
        mfcc_path, lm_path = parts[0], parts[1]

        mfcc_data_list, lm_data_list, _ = self.read_data_from_path(mfcc_path, lm_path)

        return torch.from_numpy(mfcc_data_list), torch.from_numpy(lm_data_list)
    
    def read_data_from_path(self, mfcc_path, lm_path):
        
        mfcc_data_list = []
        mfcc_list = sorted(glob(join(mfcc_path, '*.npy')))
        for mfcc_file in mfcc_list:
            mfcc_data = np.load(mfcc_file)
            mfcc_data = mfcc_data.reshape(-1)
            mfcc_data_list.append(mfcc_data)
        mfcc_data_list = np.vstack(mfcc_data_list).astype(np.float32)
        
        lm_data_list = []
        bb_list = []
        lm_list = sorted(glob(join(lm_path, '*.json')))
        for lm_file in lm_list:
            with open(lm_file, 'r') as f:
                data = json.load(f)
                # lm_data = data['lm68']
                lm_data = data['lm68'] * np.asarray(256/(data['bb'][2] - data['bb'][0]))
                bb_list.append(data['bb'])
            lm_data_list.append([lm_data])
        lm_data_list = np.vstack(lm_data_list).astype(np.float32)
        lm_data_list = lm_data_list.reshape(*lm_data_list.shape[0:-2], -1)
        return mfcc_data_list, lm_data_list, bb_list

    def __len__(self):
        return len(self.data_path)
    
class MeadKeypointDataset(Dataset):
    def __init__(self, path=None):
        self.data_path = []
        if path is not None:
            with open(path, 'r') as f:
                self.data_path = json.load(f)

    def __getitem__(self, index):
        parts = self.data_path[index].split('|')
        mfcc_path, lm_path = parts[0], parts[1]

        mfcc_data_list, lm_data_list, _ = self.read_data_from_path(mfcc_path, lm_path)

        return torch.from_numpy(mfcc_data_list), torch.from_numpy(lm_data_list)
    
    def read_data_from_path(self, mfcc_path, lm_path):
        
        mfcc_data_list = []
        mfcc_list = sorted(glob(join(mfcc_path, '*.npy')))
        for mfcc_file in mfcc_list:
            mfcc_data = np.load(mfcc_file)
            mfcc_data = mfcc_data.reshape(-1)
            mfcc_data_list.append(mfcc_data)
        mfcc_data_list = np.vstack(mfcc_data_list).astype(np.float32)
        
        lm_data_list = []
        bb_list = []
        lm_list = sorted(glob(join(lm_path, '*.json')))
        for lm_file in lm_list:
            with open(lm_file, 'r') as f:
                data = json.load(f)
                # lm_data = data['lm68']
                lm_data = data['lm68'] * np.asarray(256/(data['bb'][2] - data['bb'][0]))
                kp_data = data['kp6'] * np.asarray(256/(data['bb'][2] - data['bb'][0]))
                lm_data = np.concatenate((lm_data, kp_data))
                bb_list.append(data['bb'])
            lm_data_list.append([lm_data])
        lm_data_list = np.vstack(lm_data_list).astype(np.float32)
        lm_data_list = lm_data_list.reshape(*lm_data_list.shape[0:-2], -1)
        return mfcc_data_list, lm_data_list, bb_list

    def __len__(self):
        return len(self.data_path)