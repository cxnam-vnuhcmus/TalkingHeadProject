import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from evaluation.evaluation_landmark import calculate_LMD

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='../Datasets/Features/M003/images/neutral/level_1/00001')
parser.add_argument('--pred_lm_path', type=str, default='results/MEAD_A13L68/lm_pred.json')
parser.add_argument('--frame_index', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__': 
    image_path = args.image_path
    lm_path = args.pred_lm_path
    index = args.frame_index

    image = cv2.imread(f'{image_path }/{index+1:05d}.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image)

    with open(lm_path, 'r') as f:
        data = json.load(f)
    
    pred_list = np.asarray([data['pred']])
    gt_list = np.asarray([data['gt']])
    pred_data = pred_list[0][index]
    gt_data = gt_list[0][index]
    
    norm_distance = np.sqrt(np.sum((gt_list[0:1,index:index+1,0] - gt_list[0:1,index:index+1,16])**2, axis=2))
    lmd = calculate_LMD(pred_list[0:1,index:index+1], gt_list[0:1,index:index+1],norm_distance=norm_distance)
    print(f'LMD frame ({index}): {lmd}')
    
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(image)

    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(pred_data [:, 0], -pred_data [:, 1], alpha=0.8)

    ax = fig.add_subplot(1, 3, 3)
    img2 = image.copy()
    for p in pred_data :
        if 'bb' in data:
            p[1] = p[1] + data['bb'][1]  
            p[0] = p[0] + data['bb'][0]
        img2[p[1]-1:p[1]+1, p[0]-1:p[0]+1, :] = (0, 255, 0)
    for p in gt_data :
        if 'bb' in data:
            p[1] = p[1] + data['bb'][1]  
            p[0] = p[0] + data['bb'][0]
        img2[p[1]-1:p[1]+1, p[0]-1:p[0]+1, :] = (255, 0, 0)
    ax.imshow(img2)
    plt.show()