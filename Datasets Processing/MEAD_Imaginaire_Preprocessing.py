import json
import os
from os.path import join
import random
import numpy as np
from tqdm import tqdm
import cv2
from glob import glob
import subprocess
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--person", type=str, default='M003')
parser.add_argument("--mead_feature_folder", type=str, default='/root/Datasets/Features')
parser.add_argument("--imaginaire_folder", type=str, default='/root/imaginaire')

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

    total_list = []
    lm_folder = join(args.mead_feature_folder, args.person, 'landmarks')
    for emo in os.listdir(lm_folder):
        lm_emo_folder = join(lm_folder, emo)
        for lv in os.listdir(lm_emo_folder):
            lm_emo_lv_folder = join(lm_emo_folder, lv)
            for seq in os.listdir(lm_emo_lv_folder):
                lm_emo_lv_seq_folder = join(lm_emo_lv_folder, seq)
                total_list.append(lm_emo_lv_seq_folder)

    random.seed(0)
    random.shuffle(total_list)

    for i in range(0,len(total_list)*8//10):
        lm_emo_lv_seq_folder = total_list[i]
        path_split = lm_emo_lv_seq_folder.split('/')
        new_folder_name = f"{path_split[-3]}_{path_split[-2].replace('level_','')}_{path_split[-1]}"
        new_train_landmarks_folder = join(train_landmarks_folder, new_folder_name)
        cmd = f'cp -r {lm_emo_lv_seq_folder} {new_train_landmarks_folder}'
        subprocess.call(cmd, shell=True)

        img_emo_lv_seq_folder = lm_emo_lv_seq_folder.replace('landmarks','images')
        new_train_img_folder = join(train_image_folder, new_folder_name)
        cmd = f'cp -r {img_emo_lv_seq_folder} {new_train_img_folder}'
        subprocess.call(cmd, shell=True)
    
    for i in range(len(total_list)*8//10,len(total_list)):
        lm_emo_lv_seq_folder = total_list[i]
        path_split = lm_emo_lv_seq_folder.split('/')
        new_folder_name = f"{path_split[-3]}_{path_split[-2].replace('level_','')}_{path_split[-1]}"
        new_val_landmarks_folder = join(val_landmarks_folder, new_folder_name)
        cmd = f'cp -r {lm_emo_lv_seq_folder} {new_val_landmarks_folder}'
        subprocess.call(cmd, shell=True)

        img_emo_lv_seq_folder = lm_emo_lv_seq_folder.replace('landmarks','images')
        new_val_img_folder = join(val_image_folder, new_folder_name)
        cmd = f'cp -r {img_emo_lv_seq_folder} {new_val_img_folder}'
        subprocess.call(cmd, shell=True)