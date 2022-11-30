import os
import random
import json
from glob import glob

def MeadPreprocessing(train_file_name, val_file_name, **source_data_folder):
    data_folder = os.path.dirname(train_file_name)
    os.makedirs(data_folder, exist_ok=True)

    print('Retrieve all mfcc folders')
    total_list = []
    mfcc_folder = source_data_folder['mfccs_path']
    emo_list = sorted(glob(os.path.join(mfcc_folder, '*')))
    for emo in emo_list:
        lv_list = sorted(glob(os.path.join(emo, '*')))
        for lv in lv_list:
            mfcc_list = sorted(glob(os.path.join(lv, '*')))
            for mfcc_file in mfcc_list:
                lm_file = mfcc_file.replace('mfccs', 'landmarks74')
                if os.path.exists(lm_file):
                    total_list.append(f'{mfcc_file}|{lm_file}')
    print(f'Total: {len(total_list)}')
    random.seed(0)
    random.shuffle(total_list)

    with open(train_file_name, 'w') as f:
        json.dump(total_list[:int(len(total_list) * 0.8)], f)

    with open(val_file_name, 'w') as f:
        json.dump(total_list[int(len(total_list) * 0.8):], f)
        
if __name__ == '__main__':
    MeadPreprocessing(train_file_name = './data/train_MEAD.json',
                      val_file_name = './data/val_MEAD.json',
                      mfccs_path = '/root/Datasets/Features/M003/mfccs'
                      )
