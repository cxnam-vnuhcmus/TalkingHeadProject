import os
import random
import json
import numpy as np
from glob import glob
from tqdm import tqdm

def MeadPreprocessing_v2(train_file_name, val_file_name, **source_data_folder):
    data_folder = os.path.dirname(train_file_name)
    os.makedirs(data_folder, exist_ok=True)

    print('Retrieve all mfcc folders')
    total_list = []
    mfcc_folder = source_data_folder['mfccs_path']
    win_len = source_data_folder['win_len']
    hop_len = source_data_folder['hop_len']
    emo_list = sorted(glob(os.path.join(mfcc_folder, '*')))
    for emo in emo_list:
        lv_list = sorted(glob(os.path.join(emo, '*')))
        for lv in lv_list:
            mfcc_list = sorted(glob(os.path.join(lv, '*')))
            for mfcc_file in mfcc_list:
                total_list.append(mfcc_file)
    
    random.seed(0)
    random.shuffle(total_list)
    train_list = total_list[:int(len(total_list) * 0.8)]
    test_list = total_list[int(len(total_list) * 0.8):]
             
    total_train_list = []
    for mfcc_file in train_list:
        lm_file = mfcc_file.replace('mfccs', 'landmarks74')
        face_file = mfcc_file.replace('mfccs', 'faces')
        if os.path.exists(lm_file) and os.path.exists(face_file):
            lm_parts = sorted(glob(os.path.join(lm_file,'*.json')))
            mfcc_len = len(glob(os.path.join(mfcc_file, '*')))
            lm_parts_list = np.zeros(mfcc_len)
            for part in lm_parts:
                id = int(os.path.basename(part)[:-len('.json')]) - 1
                lm_parts_list[id] = 1
            index = 0
            while(index < mfcc_len - win_len):
                if np.sum(lm_parts_list[index:index+win_len]) == win_len:
                    total_train_list.append(f'{mfcc_file}|{lm_file}|{face_file}|{index}|{index+win_len}')
                    index = index + hop_len
                else:
                    while(lm_parts_list[index] != 0):
                        index = index + 1
                    index = index + 1
    
    print(f'Train file: {len(train_list)}; Train sample: {len(total_train_list)}')
    
    with open(train_file_name, 'w') as f:
        json.dump(total_train_list, f)

    total_test_list = []
    for mfcc_file in test_list:
        lm_file = mfcc_file.replace('mfccs', 'landmarks74')
        face_file = mfcc_file.replace('mfccs', 'faces')
        if os.path.exists(lm_file) and os.path.exists(face_file):
            lm_parts = sorted(glob(os.path.join(lm_file,'*.json')))
            mfcc_len = len(glob(os.path.join(mfcc_file, '*')))
            lm_parts_list = np.zeros(mfcc_len)
            for part in lm_parts:
                id = int(os.path.basename(part)[:-len('.json')]) - 1
                lm_parts_list[id] = 1
            index = 0
            while(index < mfcc_len - win_len):
                if np.sum(lm_parts_list[index:index+win_len]) == win_len:
                    total_test_list.append(f'{mfcc_file}|{lm_file}|{face_file}|{index}|{index+win_len}')
                    index = index + hop_len
                else:
                    while(lm_parts_list[index] != 0):
                        index = index + 1
                    index = index + 1

    print(f'Test file: {len(test_list)}; Test sample: {len(total_test_list)}')
    
    with open(val_file_name, 'w') as f:
        json.dump(total_test_list, f)
        
def MeadPreprocessing(train_file_name, val_file_name, **source_data_folder):
    data_folder = os.path.dirname(train_file_name)
    os.makedirs(data_folder, exist_ok=True)

    print('Retrieve all mfcc folders')
    total_list = []
    mfcc_folder = source_data_folder['mfccs_path']
    win_len = source_data_folder['win_len']
    hop_len = source_data_folder['hop_len']
    emo_list = sorted(glob(os.path.join(mfcc_folder, '*')))
    for emo in emo_list:
        lv_list = sorted(glob(os.path.join(emo, '*')))
        for lv in lv_list:
            mfcc_list = sorted(glob(os.path.join(lv, '*')))
            for mfcc_file in mfcc_list:
                lm_file = mfcc_file.replace('mfccs', 'landmarks74')
                face_file = mfcc_file.replace('mfccs', 'faces')
                if os.path.exists(lm_file) and os.path.exists(face_file):
                    lm_parts = sorted(glob(os.path.join(lm_file,'*.json')))
                    mfcc_len = len(glob(os.path.join(mfcc_file, '*')))
                    lm_parts_list = np.zeros(mfcc_len)
                    for part in lm_parts:
                        id = int(os.path.basename(part)[:-len('.json')]) - 1
                        lm_parts_list[id] = 1
                    index = 0
                    while(index < mfcc_len - win_len):
                        if np.sum(lm_parts_list[index:index+win_len]) == win_len:
                            total_list.append(f'{mfcc_file}|{lm_file}|{face_file}|{index}|{index+win_len}')
                            index = index + hop_len
                        else:
                            while(lm_parts_list[index] != 0):
                                index = index + 1
                            index = index + 1
    print(f'Total: {len(total_list)}')
    random.seed(0)
    random.shuffle(total_list)

    with open(train_file_name, 'w') as f:
        json.dump(total_list[:int(len(total_list) * 0.8)], f)

    with open(val_file_name, 'w') as f:
        json.dump(total_list[int(len(total_list) * 0.8):], f)

def CREMADPreprocessing_v2(train_file_name, val_file_name, **source_data_folder):
    data_folder = os.path.dirname(train_file_name)
    os.makedirs(data_folder, exist_ok=True)

    print('Retrieve all mfcc folders')
    total_list = []
    mfcc_folder = source_data_folder['mfccs_path']
    win_len = source_data_folder['win_len']
    hop_len = source_data_folder['hop_len']
    mfcc_list = sorted(glob(os.path.join(mfcc_folder, '*')))
    for mfcc_file in tqdm(mfcc_list):
        total_list.append(mfcc_file)
    
    random.seed(0)
    random.shuffle(total_list)
    train_list = total_list[:int(len(total_list) * 0.8)]
    test_list = total_list[int(len(total_list) * 0.8):]
    
    total_train_list = []
    for mfcc_file in train_list:
        lm_file = mfcc_file.replace('mfccs', 'landmarks74')
        if os.path.exists(lm_file):
            lm_parts = sorted(glob(os.path.join(lm_file,'*.json')))
            mfcc_len = len(glob(os.path.join(mfcc_file, '*')))
            lm_parts_list = np.zeros(mfcc_len)
            try:
                for part in lm_parts:
                    id = int(os.path.basename(part)[:-len('.json')]) - 1
                    lm_parts_list[id] = 1
            except:
                print(f'{lm_file} - {mfcc_len}')

            index = 0
            while(index < mfcc_len - win_len):
                if np.sum(lm_parts_list[index:index+win_len]) == win_len:
                    total_train_list.append(f'{mfcc_file}|{lm_file}|_|{index}|{index+win_len}')
                    index = index + hop_len
                else:
                    while(lm_parts_list[index] != 0):
                        index = index + 1
                    index = index + 1

    print(f'Train file: {len(train_list)}; Train sample: {len(total_train_list)}')
    
    with open(train_file_name, 'w') as f:
        json.dump(total_train_list, f)


    total_test_list = []
    for mfcc_file in test_list:
        lm_file = mfcc_file.replace('mfccs', 'landmarks74')
        if os.path.exists(lm_file):
            lm_parts = sorted(glob(os.path.join(lm_file,'*.json')))
            mfcc_len = len(glob(os.path.join(mfcc_file, '*')))
            lm_parts_list = np.zeros(mfcc_len)
            try:
                for part in lm_parts:
                    id = int(os.path.basename(part)[:-len('.json')]) - 1
                    lm_parts_list[id] = 1
            except:
                print(f'{lm_file} - {mfcc_len}')

            index = 0
            while(index < mfcc_len - win_len):
                if np.sum(lm_parts_list[index:index+win_len]) == win_len:
                    total_test_list.append(f'{mfcc_file}|{lm_file}|_|{index}|{index+win_len}')
                    index = index + hop_len
                else:
                    while(lm_parts_list[index] != 0):
                        index = index + 1
                    index = index + 1
    
    print(f'Test file: {len(test_list)}; Test sample: {len(total_test_list)}')
                    
    with open(val_file_name, 'w') as f:
        json.dump(total_test_list, f)
        
def CREMADPreprocessing(train_file_name, val_file_name, **source_data_folder):
    data_folder = os.path.dirname(train_file_name)
    os.makedirs(data_folder, exist_ok=True)

    print('Retrieve all mfcc folders')
    total_list = []
    mfcc_folder = source_data_folder['mfccs_path']
    win_len = source_data_folder['win_len']
    hop_len = source_data_folder['hop_len']
    mfcc_list = sorted(glob(os.path.join(mfcc_folder, '*')))
    for mfcc_file in tqdm(mfcc_list):
        lm_file = mfcc_file.replace('mfccs', 'landmarks74')
        if os.path.exists(lm_file):
            lm_parts = sorted(glob(os.path.join(lm_file,'*.json')))
            mfcc_len = len(glob(os.path.join(mfcc_file, '*')))
            lm_parts_list = np.zeros(mfcc_len)
            try:
                for part in lm_parts:
                    id = int(os.path.basename(part)[:-len('.json')]) - 1
                    lm_parts_list[id] = 1
            except:
                print(f'{lm_file} - {mfcc_len}')

            index = 0
            while(index < mfcc_len - win_len):
                if np.sum(lm_parts_list[index:index+win_len]) == win_len:
                    total_list.append(f'{mfcc_file}|{lm_file}|_|{index}|{index+win_len}')
                    index = index + hop_len
                else:
                    while(lm_parts_list[index] != 0):
                        index = index + 1
                    index = index + 1
    print(f'Total: {len(total_list)}')
    random.seed(0)
    random.shuffle(total_list)

    with open(train_file_name, 'w') as f:
        json.dump(total_list[:int(len(total_list) * 0.8)], f)

    with open(val_file_name, 'w') as f:
        json.dump(total_list[int(len(total_list) * 0.8):], f)

def Mead_Emo_Preprocessing(train_file_name, val_file_name, **source_data_folder):
    data_folder = os.path.dirname(train_file_name)
    os.makedirs(data_folder, exist_ok=True)

    print('Retrieve all audio feat files')
    total_list = []
    aufeat_paths = source_data_folder['aufeat_paths']
    for aufeat_path in aufeat_paths:
        aufeat_list = sorted(glob(os.path.join(aufeat_path, '**/*.npy'), recursive=True))
        for feat_path in aufeat_list:
            total_list.append(os.path.dirname(feat_path))
    total_list = list(set(total_list))
    print(f'Total: {len(total_list)}')
    random.seed(0)
    random.shuffle(total_list)

    with open(train_file_name, 'w') as f:
        json.dump(total_list[:int(len(total_list) * 0.8)], f)

    with open(val_file_name, 'w') as f:
        json.dump(total_list[int(len(total_list) * 0.8):], f)
        
if __name__ == '__main__':
    # MeadPreprocessing_v2(train_file_name = './data/train_MEAD_v2.json',
    #                   val_file_name = './data/val_MEAD_v2.json',
    #                   mfccs_path = '/root/Datasets/Features/M003/mfccs',
    #                   win_len = 25,
    #                   hop_len = 3
    #                   )
    
    # CREMADPreprocessing_v2(train_file_name = './data/train_CREMAD_v2.json',
    #                   val_file_name = './data/val_CREMAD_v2.json',
    #                   mfccs_path = '/root/Datasets/CREMA-D/Features/mfccs',
    #                   win_len = 25,
    #                   hop_len = 3
    #                   )
    
    Mead_Emo_Preprocessing(train_file_name = './data/train_emo_MEAD.json',
                      val_file_name = './data/val_emo_MEAD.json',
                      aufeat_paths = ['/root/Datasets/Features/M003/aufeat50',
                                      '/root/Datasets/Features/M030/aufeat50']
                      )
