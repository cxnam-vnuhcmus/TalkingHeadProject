import os
from os.path import join
import random
import json
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--person", type=str, default='M030')
parser.add_argument("--mead_feature_folder", type=str, default='/root/Datasets/Features')
parser.add_argument("--output_folder", type=str, default='/root/TalkingHead/dataset')
parser.add_argument("--parse_MEAD_A13L68", type=bool, default=True)
parser.add_argument("--parse_MEAD_A13L74", type=bool, default=True)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.parse_MEAD_A13L68:
        os.makedirs(args.output_folder, exist_ok=True)
        train_file = join(args.output_folder, 'train_MEAD_A13L68.json')
        test_file = join(args.output_folder, 'test_MEAD_A13L68.json')
        
        total_list = []
        lm_folder = join(args.mead_feature_folder, args.person, 'landmarks68-512')
        
        total_list = []
        emo_list = sorted(glob(join(lm_folder, '*')))
        for emo in emo_list:
            lv_list = sorted(glob(join(emo, '*')))
            for lv in lv_list:
                lm_list = sorted(glob(join(lv, '*')))
                for lm_file in lm_list:
                    mfcc_file = lm_file.replace('landmarks68-512', 'mfcc')
                    total_list.append(f'{mfcc_file}|{lm_file}')

        random.seed(0)
        random.shuffle(total_list)

        with open(train_file, 'w') as f:
            json.dump(total_list[:int(len(total_list) * 0.8)], f)

        with open(test_file, 'w') as f:
            json.dump(total_list[int(len(total_list) * 0.8):], f)
            
    if args.parse_MEAD_A13L74:
        os.makedirs(args.output_folder, exist_ok=True)
        train_file = join(args.output_folder, 'train_MEAD_A13L74.json')
        test_file = join(args.output_folder, 'test_MEAD_A13L74.json')
        
        total_list = []
        lm_folder = join(args.mead_feature_folder, args.person, 'landmarks74-512')
        
        total_list = []
        emo_list = sorted(glob(join(lm_folder, '*')))
        for emo in emo_list:
            lv_list = sorted(glob(join(emo, '*')))
            for lv in lv_list:
                lm_list = sorted(glob(join(lv, '*')))
                for lm_file in lm_list:
                    mfcc_file = lm_file.replace('landmarks74-512', 'mfcc')
                    total_list.append(f'{mfcc_file}|{lm_file}')

        random.seed(0)
        random.shuffle(total_list)

        with open(train_file, 'w') as f:
            json.dump(total_list[:int(len(total_list) * 0.8)], f)

        with open(test_file, 'w') as f:
            json.dump(total_list[int(len(total_list) * 0.8):], f)