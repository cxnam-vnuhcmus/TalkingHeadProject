import sys
sys.path.append('/root/TalkingHead/evaluation')

import json
import argparse
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from evaluation_landmark import *

parser = argparse.ArgumentParser()
parser.add_argument('--predict_file', type=str, default='result/predict.json')
parser.add_argument('--output_file', type=str, default='result/predict.png')
args = parser.parse_args()

if __name__ == '__main__': 
    with open(args.predict_file, 'rt') as f:
        data = json.load(f)

    gt = np.asarray([data['gt']])
    pred = np.asarray([data['pred']])
    norm_distance = np.sqrt(np.sum((gt[:,:,0] - pred[:,:,16])**2, axis=2))
    lmd = calculate_LMD(pred, gt, norm_distance=norm_distance)
    print(f'LMD: {lmd*100}')
    
    lmv = calculate_LMV(pred, gt, norm_distance=norm_distance[:,1:])
    print(f'LMV: {lmv*100}')
    
    fig = plt.figure(figsize=(15, 15))
    n_frames = 3
    for i in range(n_frames):
        for j in range(n_frames):
            ax = fig.add_subplot(n_frames, n_frames, i*n_frames+j+1)
            image = np.zeros((256,256,3), dtype=np.uint8)
            pixel_size = 2
            frame_index = (i*n_frames+j)**2
            for p in pred[0,frame_index]:
                image[p[1]-pixel_size:p[1]+pixel_size, p[0]-pixel_size:p[0]+pixel_size,:] = (0, 255, 0)
            for p in gt[0,frame_index]:
                image[p[1]-pixel_size:p[1]+pixel_size, p[0]-pixel_size:p[0]+pixel_size,:] = (255, 0, 0)
            ax.imshow(image)
            
            norm_distance = np.sqrt(np.sum((gt[:,frame_index:frame_index+1,0] - pred[:,frame_index:frame_index+1,16])**2, axis=2))
            lmd_f = calculate_LMD(pred[:,frame_index:frame_index+1], gt[:,frame_index:frame_index+1], norm_distance=norm_distance)
            ax.set_title(f'LMD (#{frame_index}): {lmd_f*100}')
    fig.suptitle(f'LMD: {lmd*100}\nLMV: {lmv*100}', fontsize=16)
    plt.savefig(args.output_file)
    