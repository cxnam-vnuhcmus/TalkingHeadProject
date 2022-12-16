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
import cv2

# img_paths = glob(os.path.join('/root/Datasets/Features/M003/images','**/*.jpg'), recursive=True)
# img_folders = []
# for img_path in img_paths:
#     img_folder = os.path.dirname(img_path)
#     if img_folder not in img_folders:
#         img_folders.append(img_folder)

# random.seed(0)
# random.shuffle(img_folders)
      
# for img_folder in img_folders[:len(img_folders)*80//100]:
#     folder_name = img_folder.replace('/root/Datasets/Features/M003/images/','M003/')
#     folder_name = folder_name.replace('_','')
#     folder_name = folder_name.replace('/','_')
#     des_img_folder = os.path.join('/root/TalkingHead/imaginaire/datasets/train/images', folder_name)
#     os.makedirs(des_img_folder, exist_ok=True)
#     subprocess.call(f'cp -r {img_folder}/* {des_img_folder}', shell=True)
    
# for img_folder in img_folders[len(img_folders)*80//100:]:
#     folder_name = img_folder.replace('/root/Datasets/Features/M003/images/','M003/')
#     folder_name = folder_name.replace('_','')
#     folder_name = folder_name.replace('/','_')
#     des_img_folder = os.path.join('/root/TalkingHead/imaginaire/datasets/test/images', folder_name)
#     os.makedirs(des_img_folder, exist_ok=True)
#     subprocess.call(f'cp -r {img_folder}/* {des_img_folder}', shell=True)

# img_files = glob(os.path.join('/root/TalkingHead/imaginaire/datasets/test/images','**/*.jpg'), recursive=True)
# for img_file in tqdm(img_files):
#     img = cv2.imread(img_file)
#     image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)
#     img = img[:,:,2] * thresh
    
#     des_img_file = img_file.replace('/images/','/seg_maps/')
#     os.makedirs(os.path.dirname(des_img_file), exist_ok=True)
#     cv2.imwrite(des_img_file, img)
    
import numpy as np
img_files = glob(os.path.join('/root/TalkingHead/imaginaire/datasets/test/images','**/*.jpg'), recursive=True)
for img_file in tqdm(img_files):
    img_file_parts = img_file.split('/')
    img_file_name = img_file_parts[-1][:-4]
    img_file_path = img_file_parts[-2]
    img_file_path = img_file_path.replace('M003_','')
    img_file_path = img_file_path.replace('_','/')
    img_file_path = img_file_path.replace('level','level_')
    
    full_path = f'/root/Datasets/Features/M003/landmarks74/{img_file_path}/{img_file_name}.json'
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            data = json.load(f)
            lm_data = np.asarray(data['lm68']) + (data['bb'][0],data['bb'][1])
    
    des_path = img_file.replace('/images/','/landmarks-dlib68/')
    des_path = des_path.replace('.jpg','.json')
    os.makedirs(os.path.dirname(des_path), exist_ok=True)
    with open(des_path, 'w') as f:
        json.dump(lm_data.tolist(), f)
        
    
    
    
    

# parser = argparse.ArgumentParser()
# parser.add_argument("--person", type=str, default='M003')
# parser.add_argument("--landmark_folder", type=str, default='landmarks')
# parser.add_argument("--image_folder", type=str, default='images')
# parser.add_argument("--mead_feature_folder", type=str, default='/root/Datasets/Features')
# parser.add_argument("--imaginaire_folder", type=str, default='/root/TalkingHead/imaginaire/datasets')

# def get_landmark68(folder):
#     json_list = sorted(glob(join(folder, '*.json')))
#     for json_file in json_list:
#         with open(json_file, 'r') as f:
#             data = json.load(f)
#         with open(json_file, 'w') as f:
#             json.dump(data['lm68'], f)
            
# if __name__ == '__main__':
#     args = parser.parse_args()
#     train_image_folder = join(args.imaginaire_folder, 'train/images')
#     train_landmarks_folder = join(args.imaginaire_folder, 'train/landmarks-dlib68')
#     val_image_folder = join(args.imaginaire_folder, 'val/images')
#     val_landmarks_folder = join(args.imaginaire_folder, 'val/landmarks-dlib68')
#     os.makedirs(train_image_folder, exist_ok=True)
#     os.makedirs(train_landmarks_folder, exist_ok=True)
#     os.makedirs(val_image_folder, exist_ok=True)
#     os.makedirs(val_landmarks_folder, exist_ok=True)

#     print("Load all landmark folders")
#     total_list = []
#     lm_folder = join(args.mead_feature_folder, args.person, args.landmark_folder)
#     for emo in os.listdir(lm_folder):
#         lm_emo_folder = join(lm_folder, emo)
#         for lv in os.listdir(lm_emo_folder):
#             lm_emo_lv_folder = join(lm_emo_folder, lv)
#             for seq in os.listdir(lm_emo_lv_folder):
#                 lm_emo_lv_seq_folder = join(lm_emo_lv_folder, seq)
#                 total_list.append(lm_emo_lv_seq_folder)

#     random.seed(0)
#     random.shuffle(total_list)

#     for i in tqdm(range(0,len(total_list)*8//10),"Train data: "):
#         lm_emo_lv_seq_folder = total_list[i]
#         path_split = lm_emo_lv_seq_folder.split('/')
#         new_folder_name = f"{args.person}_{path_split[-3]}_{path_split[-2].replace('level_','')}_{path_split[-1]}"
#         new_train_landmarks_folder = join(train_landmarks_folder, new_folder_name)
#         cmd = f'cp -r {lm_emo_lv_seq_folder} {new_train_landmarks_folder}'
#         subprocess.call(cmd, shell=True)

#         img_emo_lv_seq_folder = lm_emo_lv_seq_folder.replace(args.landmark_folder,args.image_folder)
#         new_train_img_folder = join(train_image_folder, new_folder_name)
#         cmd = f'cp -r {img_emo_lv_seq_folder} {new_train_img_folder}'
#         subprocess.call(cmd, shell=True)
    
#     for i in tqdm(range(len(total_list)*8//10,len(total_list)), "Val data: "):
#         lm_emo_lv_seq_folder = total_list[i]
#         path_split = lm_emo_lv_seq_folder.split('/')
#         new_folder_name = f"{args.person}_{path_split[-3]}_{path_split[-2].replace('level_','')}_{path_split[-1]}"
#         new_val_landmarks_folder = join(val_landmarks_folder, new_folder_name)
#         cmd = f'cp -r {lm_emo_lv_seq_folder} {new_val_landmarks_folder}'
#         subprocess.call(cmd, shell=True)

#         img_emo_lv_seq_folder = lm_emo_lv_seq_folder.replace(args.landmark_folder,args.image_folder)
#         new_val_img_folder = join(val_image_folder, new_folder_name)
#         cmd = f'cp -r {img_emo_lv_seq_folder} {new_val_img_folder}'
#         subprocess.call(cmd, shell=True)
        
