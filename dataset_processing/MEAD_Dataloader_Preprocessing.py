import os
from os.path import join
import random
import json
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--person", type=str, default='M003')
parser.add_argument("--mead_feature_folder", type=str, default='/root/Datasets/Features')
parser.add_argument("--output_folder", type=str, default='/root/TalkingHead/dataset')
parser.add_argument("--parse_MEAD_A13L68", type=bool, default=True)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.parse_MEAD_A13L68:
        os.makedirs(args.output_folder, exist_ok=True)
        train_file = join(args.output_folder, 'train_MEAD_A13L68.json')
        test_file = join(args.output_folder, 'test_MEAD_A13L68.json')
        
        total_list = []
        lm_folder = join(args.mead_feature_folder, args.person, 'landmarks')
        lm_list = sorted(glob(join(lm_folder, '**/*.json'), recursive=True))

        total_list = []
        for lm_file in lm_list:
            mfcc_file = lm_file.replace('landmarks', 'mfcc')
            mfcc_file = mfcc_file.replace('json', 'npy')
            total_list.append(f'{mfcc_file}|{lm_file}')

        random.seed(0)
        random.shuffle(total_list)

        with open(train_file, 'w') as f:
            json.dump(total_list[:int(len(total_list) * 0.8)], f)

        with open(test_file, 'w') as f:
            json.dump(total_list[int(len(total_list) * 0.8):], f)