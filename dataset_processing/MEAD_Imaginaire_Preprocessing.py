r"""Convert Images and Landmark68 Folders of MEAD dataset to imaginaire format
Spliting dataset into 80% train - 20% test.
imagaires
    |- datasets
        |- train
            |- images
                |- seq00001
                    |- 00001.jpeg
                    |- ...
            |- landmarks-dlib68
                |- seq00001
                    |- 00001.json
                    |- ...
        |- val 
            |- images
                |- seq00001
                    |- 00001.jpeg
                    |- ...
            |- landmarks-dlib68
                |- seq00001
                    |- 00001.json
                    |- ...
Args:
    person              :   id of character folder
    mead_feature_folder :   path of mead features folder
    imaginaire_folder   :   path of imaginaire folder
Returns:
    
"""  

import os
from os.path import join
import random
import subprocess
import argparse
import json
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--person", type=str, default='M030')
parser.add_argument("--landmark_folder", type=str, default='landmarks')
parser.add_argument("--mead_feature_folder", type=str, default='/root/Datasets/Features')
parser.add_argument("--imaginaire_folder", type=str, default='/root/TalkingHead/imaginaire')

def get_landmark68(folder):
    json_list = sorted(glob(join(folder, '*.json')))
    for json_file in json_list:
        with open(json_file, 'r') as f:
            data = json.load(f)
        with open(json_file, 'w') as f:
            json.dump(data['lm68'], f)
            
if __name__ == '__main__':
    args = parser.parse_args()
    train_image_folder = join(args.imaginaire_folder, 'datasets/train/images')
    train_landmarks_folder = join(args.imaginaire_folder, 'datasets/train/landmarks-dlib68')
    val_image_folder = join(args.imaginaire_folder, 'datasets/val/images')
    val_landmarks_folder = join(args.imaginaire_folder, 'datasets/val/landmarks-dlib68')
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(train_landmarks_folder, exist_ok=True)
    os.makedirs(val_image_folder, exist_ok=True)
    os.makedirs(val_landmarks_folder, exist_ok=True)

    print("Load all landmark folders")
    total_list = []
    lm_folder = join(args.mead_feature_folder, args.person, args.landmark_folder)
    for emo in os.listdir(lm_folder):
        lm_emo_folder = join(lm_folder, emo)
        for lv in os.listdir(lm_emo_folder):
            lm_emo_lv_folder = join(lm_emo_folder, lv)
            for seq in os.listdir(lm_emo_lv_folder):
                lm_emo_lv_seq_folder = join(lm_emo_lv_folder, seq)
                total_list.append(lm_emo_lv_seq_folder)

    random.seed(0)
    random.shuffle(total_list)

    for i in tqdm(range(0,len(total_list)*8//10),"Train data: "):
        lm_emo_lv_seq_folder = total_list[i]
        path_split = lm_emo_lv_seq_folder.split('/')
        new_folder_name = f"{args.person}_{path_split[-3]}_{path_split[-2].replace('level_','')}_{path_split[-1]}"
        new_train_landmarks_folder = join(train_landmarks_folder, new_folder_name)
        cmd = f'cp -r {lm_emo_lv_seq_folder} {new_train_landmarks_folder}'
        subprocess.call(cmd, shell=True)

        img_emo_lv_seq_folder = lm_emo_lv_seq_folder.replace(args.landmark_folder,'images')
        new_train_img_folder = join(train_image_folder, new_folder_name)
        cmd = f'cp -r {img_emo_lv_seq_folder} {new_train_img_folder}'
        subprocess.call(cmd, shell=True)
    
    for i in tqdm(range(len(total_list)*8//10,len(total_list)), "Val data: "):
        lm_emo_lv_seq_folder = total_list[i]
        path_split = lm_emo_lv_seq_folder.split('/')
        new_folder_name = f"{args.person}_{path_split[-3]}_{path_split[-2].replace('level_','')}_{path_split[-1]}"
        new_val_landmarks_folder = join(val_landmarks_folder, new_folder_name)
        cmd = f'cp -r {lm_emo_lv_seq_folder} {new_val_landmarks_folder}'
        subprocess.call(cmd, shell=True)

        img_emo_lv_seq_folder = lm_emo_lv_seq_folder.replace(args.landmark_folder,'images')
        new_val_img_folder = join(val_image_folder, new_folder_name)
        cmd = f'cp -r {img_emo_lv_seq_folder} {new_val_img_folder}'
        subprocess.call(cmd, shell=True)
        
