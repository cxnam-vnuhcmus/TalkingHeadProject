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
from evaluation.evaluation_landmark import *

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='/root/TalkingHead/config/config_MEAD_A13L74.yaml')
parser.add_argument('--mfcc_folder', type=str, default='/root/Datasets/Features/M003/mfcc/neutral/level_1/00001')
parser.add_argument('--landmark_folder', type=str, default='/root/Datasets/Features/M003/landmarks74-512/neutral/level_1/00001')
args = parser.parse_args()

if __name__ == '__main__': 
    config = yaml.safe_load(open(args.config_file))
    dataloader = globals()[config['dataloader_name']](config, is_train=False)
    if torch.cuda.is_available():
        model = globals()[config['model_name']](config)
    
    #Load pretrain
    model.load(load_optim = True)
    model.eval()

    mfcc_path = args.mfcc_folder
    lm_path = args.landmark_folder
    mfcc_data_list, lm_data_list, bb_list = dataloader.getDataset().read_data_from_path(mfcc_path, lm_path)

    x,y = torch.from_numpy(mfcc_data_list).unsqueeze(0), torch.from_numpy(lm_data_list).unsqueeze(0)
    y = y.reshape(*y.shape[0:2], -1)
    pred = model(x)
    if config['calculate_lmd']:
        y_np = y.detach().numpy().reshape(*pred.shape[0:-1], 68, 2)
        pred_np = pred.detach().numpy().reshape(*pred.shape[0:-1], 68, 2)
        norm_distance = np.sqrt(np.sum((y_np[:,:,0] - y_np[:,:,16])**2, axis=2))
        lmd = calculate_LMD(pred_np, y_np,norm_distance=norm_distance)
        print(f'LMD: {lmd}')
        
    pred = pred.squeeze(0).to(torch.int16)
    y = y.squeeze(0).to(torch.int16)
    output_data =   {
                        "pred": pred.reshape(*pred.shape[0:-1], 68, 2).tolist(),
                        "gt": y.reshape(*y.shape[0:-1], 68, 2).tolist(),
                        "bb": bb_list
                    }
   
    print(f"Image path: {mfcc_path.replace('mfcc', 'images512')}")
    save_path = join(config['save_path'],'lm_pred.json')
    print(f'Save predict landmark to: {save_path}')
    with open(save_path, 'w') as f:
        json.dump(output_data, f)
    
    if config['export_vid2vid']:
        #Write driving folder
        driving_image_path = join(config['driving_path'], 'images')
        driving_lm_path = join(config['driving_path'], 'landmarks-dlib68')
        driving_gtlm_path = join(config['driving_path'], 'gt-landmarks-dlib68')
        os.makedirs(driving_image_path, exist_ok=True)
        os.makedirs(driving_lm_path, exist_ok=True)
        os.makedirs(driving_gtlm_path, exist_ok=True)
        
        img_path = mfcc_path.replace('mfcc','images512')
        cmd = f'cp -r {img_path}/*.jpg {driving_image_path}'
        subprocess.call(cmd, shell=True)
        
        y = y.squeeze(0).to(torch.int16)
        pred = pred.squeeze(0).to(torch.int16)
        
        for index in range(pred.shape[0]):
            outputGtPath = join(driving_gtlm_path, f'{index+1:05d}.json')
            with open(outputGtPath, 'w') as f:  
                point = y[index].reshape(68,2)
                reverse_point = (point / np.asarray(256/(bb_list[index][2] - bb_list[index][0]))).to(torch.int16)
                final_data = reverse_point + np.asarray([bb_list[index][0:2]])
                json.dump(final_data.tolist(), f)
                
            outputPath = join(driving_lm_path, f'{index+1:05d}.json')
            with open(outputPath, 'w') as f:  
                point = pred[index].reshape(68,2)
                reverse_point = (point / np.asarray(256/(bb_list[index][2] - bb_list[index][0]))).to(torch.int16)
                final_data = reverse_point + np.asarray([bb_list[index][0:2]])
                json.dump(final_data.tolist(), f)
