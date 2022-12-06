import json
from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob
from os.path import join
import imageio
import os

class MeadDataset(Dataset):
    def __init__(self, path=None):
        self.data_path = []
        if path is not None:
            with open(path, 'r') as f:
                self.data_path = json.load(f)

    def __getitem__(self, index):
        parts = self.data_path[index].split('|')
        
        data = self.read_data_from_path(*parts)

        return  {
                    'audio': data['mfcc_data_list'], 
                    # 'landmark': data['lm_data_list']
                }, data['face_data_list']
                
    
    def read_data_from_path(self, mfcc_path=None, lm_path=None, face_path=None, start=None, end=None):
        mfcc_data_list = None
        lm_data_list = None
        face_data_list = None
        
        if mfcc_path is not None:
            mfcc_data_list = []
            mfcc_list = sorted(glob(join(mfcc_path, '*.npy')))
            if start is None and end is None:
                start = 0
                end = len(mfcc_list)
            else:
                start, end = int(start), int(end)
            for index in range(start,end):
                mfcc_file = mfcc_list[index]
                mfcc_data = np.load(mfcc_file)
                mfcc_data = np.expand_dims(mfcc_data, axis=0)
                mfcc_data_list.append(mfcc_data)
            mfcc_data_list = np.vstack(mfcc_data_list).astype(np.float32)
        
        # if lm_path is not None:
        #     lm_data_list = []
        #     bb_list = []
        #     lm_list = sorted(glob(join(lm_path, '*.json')))
        #     if start is None and end is None:
        #         start = 0
        #         end = len(lm_list)
        #     else:
        #         start, end = int(start), int(end)
        #     for index in range(start,end):
                # try:
                #     lm_file = os.path.join(lm_path, f'{index:05d}.json')
                # except Exception as e:
                #     print(f'{face_path} | {index}')
        #         with open(lm_file, 'r') as f:
        #             data = json.load(f)
        #             # lm_data = data['lm68']
        #             lm_data = data['lm68'] * np.asarray(256/(data['bb'][2] - data['bb'][0]))
        #             bb_list.append(data['bb'])
        #         lm_data_list.append([lm_data])
        #     lm_data_list = np.vstack(lm_data_list).astype(np.float32)
        #     lm_data_list = lm_data_list.reshape(*lm_data_list.shape[0:-2], -1)
            
        if face_path is not None:
            face_data_list = []
            if start is None and end is None:
                start = 0
                end = 1
            else:
                start, end = int(start), int(end)
            for index in range(start,end):
                try:
                    face_file = os.path.join(face_path, f'{index+1:05d}.jpg')
                except Exception as e:
                    print(f'{face_path} | {index}')
                image = imageio.imread(face_file)
                image = image / 255.0
                face_data_list.append([image])
            face_data_list = np.vstack(face_data_list).astype(np.float32)
        return {
            'mfcc_data_list': torch.from_numpy(mfcc_data_list), 
            # 'lm_data_list': torch.from_numpy(lm_data_list), 
            'face_data_list': torch.from_numpy(face_data_list)
        }

    def __len__(self):
        return len(self.data_path)