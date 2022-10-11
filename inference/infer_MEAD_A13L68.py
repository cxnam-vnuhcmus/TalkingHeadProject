import sys
sys.path.append('..')

import torch
import argparse
import yaml
from dataloader.dataloader_import import *
from model.model_import import *
import json
from os.path import join
import os
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='/root/TalkingHead/config/config_MEAD_A13L68.yaml')
args = parser.parse_args()

if __name__ == '__main__': 
    config = yaml.safe_load(open(args.config_file))
    dataloader = globals()[config['dataloader_name']](config, is_train=False)
    if torch.cuda.is_available():
        model = globals()[config['model_name']](config)
    
    #Load pretrain
    model.load(load_optim = True)
    model.eval()

    mfcc_path = '/root/Datasets/Features/M003/mfcc/neutral/level_1/00001'
    lm_path = mfcc_path.replace('mfcc','landmarks')
    mfcc_data_list, lm_data_list = dataloader.getDataset().read_data_from_path(mfcc_path, lm_path)

    x,y = torch.from_numpy(mfcc_data_list).unsqueeze(0), torch.from_numpy(lm_data_list).unsqueeze(0)
    y = y.reshape(*y.shape[0:2], -1)
    pred = model(x)
    output_data = {"pred": pred.reshape(*pred.shape[0:2], 68, 2).tolist(),
                    "gt": y.reshape(*y.shape[0:2], 68, 2).tolist()
                    }
    
    with open(join(config['save_path'],'lm_pred.json'), 'w') as f:
        json.dump(output_data, f)
    
    #Write driving folder
    driving_image_path = join(config['save_path'], 'driving', 'images')
    driving_lm_path = join(config['save_path'], 'driving', 'landmarks-dlib68')
    os.makedirs(driving_image_path, exist_ok=True)
    os.makedirs(driving_lm_path, exist_ok=True)
    
    img_path = mfcc_path.replace('mfcc','images')
    cmd = f'cp -r {img_path}/*.jpeg {driving_image_path}'
    subprocess.call(cmd, shell=True)
    
    lm_path = mfcc_path.replace('mfcc','landmarks')
    cmd = f'cp -r {lm_path}/*.json {driving_lm_path}'
    subprocess.call(cmd, shell=True)

    # pred = pred[0].to(torch.int16)
    # for index in range(pred.shape[0]):
    #     outputPath = join(driving_lm_path, f'{index+1:05d}.json')
    #     with open(outputPath, 'w') as f:  
    #         point = pred[index].reshape(68,2)
    #         json.dump(point.tolist(), f)