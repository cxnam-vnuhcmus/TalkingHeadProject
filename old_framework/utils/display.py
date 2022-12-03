import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from evaluation.evaluation_landmark import calculate_LMD, calculate_LMV

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='results/imaginaire/MEAD_A13L74_S256/driving/images')
parser.add_argument('--pred_lm_path', type=str, default='results/imaginaire/MEAD_A13L74_S256/driving/landmarks-dlib68')
parser.add_argument('--gt_lm_path', type=str, default='results/imaginaire/MEAD_A13L74_S256/driving/gt-landmarks-dlib68')
parser.add_argument('--frame_index', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__': 
    index = args.frame_index
    image_path = os.path.join(args.image_path, f'{index+1:05d}.jpg')
    lm_path = os.path.join(args.pred_lm_path, f'{index+1:05d}.json')
    gt_path = os.path.join(args.gt_lm_path, f'{index+1:05d}.json')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image)

    with open(lm_path, 'r') as f:
        pred_data = np.asarray(json.load(f))
    with open(gt_path, 'r') as f:
        gt_data = np.asarray(json.load(f))
    
    norm_distance = np.sqrt(np.sum((gt_data[0] - gt_data[16])**2, axis=0))
    lmd = calculate_LMD(pred_data, gt_data, norm_distance=norm_distance)
    print(f'LMD frame ({index}): {lmd}')
    
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(image)

    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(pred_data [:, 0], -pred_data [:, 1], alpha=0.8)

    ax = fig.add_subplot(1, 3, 3)
    img2 = image.copy()
    pixel_size = 2
    for p in pred_data :
        img2[p[1]-pixel_size:p[1]+pixel_size, p[0]-pixel_size:p[0]+pixel_size, :] = (0, 255, 0)
    for p in gt_data :
        img2[p[1]-pixel_size:p[1]+pixel_size, p[0]-pixel_size:p[0]+pixel_size, :] = (255, 0, 0)
    ax.imshow(img2)
    plt.show()