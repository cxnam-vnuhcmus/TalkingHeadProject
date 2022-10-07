r"""Video Preprocessing 
- Extract to Audio
- Extract to Images (frames of each video) 256x256x3
- Extract to Landmarks (dlib68 landmark of each frame)    
"""  

import json
import os
from os.path import join
import numpy as np
from tqdm import tqdm
import cv2
from mlxtend.image import extract_face_landmarks
from glob import glob
import subprocess
import argparse
import moviepy.editor as mp

parser = argparse.ArgumentParser()
parser.add_argument("--person", type=str, default='M003')
parser.add_argument("--extract_audio", type=bool, default=True)
parser.add_argument("--extract_images", type=bool, default=False)
parser.add_argument("--extract_lm68", type=bool, default=False)

if __name__ == '__main__':
    args = parser.parse_args()
    root = '/root/Datasets/'
    os.makedirs(join(root, 'Features'), exist_ok=True)
    if (args.extract_audio):
        inputFolder = join(root, f'MEAD/{args.person}/video/front')
        outputFolder = join(root, f'Features/{args.person}/audios')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder, '**/*.mp4'), recursive=True))

        for idx, filename in tqdm(enumerate(filelist)):
            subPath = filename[len(inputFolder)+1:-len('.mp4')-4]
            num = int(filename[-len('.mp4')-3:-len('.mp4')])
            
            outputPath = join(root, outputFolder, subPath)
            os.makedirs(outputPath, exist_ok=True)
            outputPath = join(outputPath, f'{num:05d}.wav')

            # cmd = 'ffmpeg -i ' + filename + ' -ab 160k -ac 1 -ar 16000 -vn ' + outputPath
            # subprocess.call(cmd, shell=True)
            clip = mp.VideoFileClip(filename)
            num_frames = round(clip.reader.fps*clip.reader.duration)
            clip.audio.write_audiofile(outputPath, fps=16000)
            
    if (args.extract_images):
        inputFolder = join(root, f'MEAD/{args.person}/video/front')
        outputFolder = join(root, f'Features/{args.person}/images')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder,'**/*.mp4'), recursive=True))

        for idx, filename in tqdm(enumerate(filelist)):
            subPath = filename[len(inputFolder)+1:-len('.mp4')-4]
            num = int(filename[-len('.mp4')-3:-len('.mp4')])           
            
            outputPath = join(root, outputFolder, subPath, f'{num:05d}')
            os.makedirs(outputPath, exist_ok=True)
            
            # Create the images
            cmd = 'ffmpeg -i ' + filename + ' -vf scale=-1:256 '+ outputPath + '/$filename%05d' + '.bmp'
            subprocess.call(cmd, shell=True)

            # Cropping
            imglist = sorted(glob( outputPath + '/*.bmp'))

            for i in range(len(imglist)):
                img = cv2.imread(imglist[i])
                x = int(np.floor((img.shape[1]-256)/2))
                crop_img = img[0:256, x:x+256]
                cv2.imwrite( imglist[i][0:-len('.bmp')] + '.jpeg', crop_img)

            subprocess.call('rm -rf '+ outputPath + '/*.bmp', shell=True)
            
    if (args.extract_lm68):
        inputFolder = join(root, f'Features/{args.person}/images')
        outputFolder = join(root, f'Features/{args.person}/landmarks')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder,'**/*.jpeg'), recursive=True))

        for idx, filename in tqdm(enumerate(filelist)):
            subPath = filename[len(inputFolder)+1:-len('.jpeg')-5]
            num = int(filename[-len('.jpeg')-4:-len('.jpeg')])           
            
            outputPath = join(root, outputFolder, subPath)
            os.makedirs(outputPath, exist_ok=True) 
            
            img = cv2.imread(filename)
            landmarks = extract_face_landmarks(img)
            if len(landmarks) == 68:
                with open(join(outputPath, f'{num:05d}.json'), 'w') as f:  
                    json.dump(landmarks.tolist(), f)
            
            
            
        
        