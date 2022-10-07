import os
import librosa
import numpy as np
import python_speech_features
import argparse
from os.path import join
from glob import glob
from tqdm import tqdm
import moviepy.editor as mp

parser = argparse.ArgumentParser()
parser.add_argument("--person", type=str, default='M003')
parser.add_argument("--mead_feature_path", type=str, default='/root/Datasets/Features')
parser.add_argument("--extract_mfcc", type=bool, default=False)
parser.add_argument("--extract_mfcc13", type=bool, default=True)

if __name__ == '__main__':
    args = parser.parse_args()
    audio_folder = join(args.mead_feature_path, args.person, "audios")
    filelist = sorted(glob(join(audio_folder, '**/*.wav'), recursive=True))

    if args.extract_mfcc:
        for idx, filename in tqdm(enumerate(filelist)):
            outputPath = filename.replace('audios','mfcc')[:-4]
            os.makedirs(outputPath, exist_ok=True)

            speech, sr = librosa.load(filename, sr=16000)
            print(len(speech))
            speech = np.insert(speech, 0, np.zeros(1920))
            speech = np.append(speech, np.zeros(1920))
            mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)

            time_len = mfcc.shape[0]
            print(mfcc.shape)
            for input_idx in range(int((time_len-28)/4)+1):
                input_feat = mfcc[4*input_idx:4*input_idx+28,:]
                np.save(join(outputPath, f'{input_idx+1:05d}.npy'), input_feat)
            break

    if args.extract_mfcc13:
        for idx, filename in tqdm(enumerate(filelist)):
            outputPath = filename.replace('audios','mfcc')[:-4]
            os.makedirs(outputPath, exist_ok=True)

            imagePath = outputPath.replace('mfcc', 'images')
            num_frames = len(os.listdir(imagePath))

            audio_signal, sr = librosa.load(filename, sr=16000)
            hop_length = int(sr/ 30)

            check = 1
            while(check):
                features = librosa.feature.mfcc(audio_signal, sr, n_mfcc=13, hop_length= hop_length, n_fft = 1200)
                if features.shape[1] == num_frames:
                    check = 0
                    for input_idx in range(num_frames):
                        np.save(join(outputPath, f'{input_idx+1:05d}.npy'), features)
                else:
                    if features.shape[1] > num_frames:
                        hop_length += 1
                    else:
                        hop_length -= 1
