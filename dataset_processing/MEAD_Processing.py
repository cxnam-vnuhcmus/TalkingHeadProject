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
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

parser = argparse.ArgumentParser()
parser.add_argument("--person", type=str, default='M003')
parser.add_argument("--extract_audio", type=bool, default=False)
parser.add_argument("--extract_images", type=bool, default=False)
parser.add_argument("--extract_lm68", type=bool, default=False)
parser.add_argument("--extract_lm74", type=bool, default=True)

if __name__ == '__main__':
    args = parser.parse_args()
    root = '/root/Datasets/'
    os.makedirs(join(root, 'Features'), exist_ok=True)
    if (args.extract_audio):
        inputFolder = join(root, f'MEAD/{args.person}/video/front')
        outputFolder = join(root, f'Features/{args.person}/audios')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder, '**/*.mp4'), recursive=True))

        for filename in tqdm(filelist):
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
        outputFolder = join(root, f'Features/{args.person}/images512')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder,'**/*.mp4'), recursive=True))

        for filename in tqdm(filelist):
            subPath = filename[len(inputFolder)+1:-len('.mp4')-4]
            num = int(filename[-len('.mp4')-3:-len('.mp4')])           
            
            outputPath = join(root, outputFolder, subPath, f'{num:05d}')
            os.makedirs(outputPath, exist_ok=True)
            
            # Create the images
            cmd = 'ffmpeg -i ' + filename + ' -vf scale=-1:512 '+ outputPath + '/$filename%05d' + '.bmp'
            subprocess.call(cmd, shell=True)

            # Cropping
            imglist = sorted(glob( outputPath + '/*.bmp'))

            for i in range(len(imglist)):
                img = cv2.imread(imglist[i])
                x = int(np.floor((img.shape[1]-512)/2))
                crop_img = img[0:512, x:x+512]
                cv2.imwrite( imglist[i][0:-len('.bmp')] + '.jpeg', crop_img)

            subprocess.call('rm -rf '+ outputPath + '/*.bmp', shell=True)
            
    if (args.extract_lm68):
        inputFolder = join(root, f'Features/{args.person}/images512')
        outputFolder = join(root, f'Features/{args.person}/landmarks68-512')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder,'**/*.jpeg'), recursive=True))

        for filename in tqdm(filelist):
            subPath = filename[len(inputFolder)+1:-len('.jpeg')-5]
            num = int(filename[-len('.jpeg')-4:-len('.jpeg')])           
            
            outputPath = join(root, outputFolder, subPath)
            os.makedirs(outputPath, exist_ok=True) 
            
            img = cv2.imread(filename)
            landmarks = extract_face_landmarks(img)
            if len(landmarks) == 68:
                with open(join(outputPath, f'{num:05d}.json'), 'w') as f:  
                    json.dump(landmarks.tolist(), f)

    if (args.extract_lm74):
        inputFolder = join(root, f'Features/{args.person}/images512')
        outputFolder = join(root, f'Features/{args.person}/landmarks74-512')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder,'**/*.jpeg'), recursive=True))

        for filename in tqdm(filelist):
            subPath = filename[len(inputFolder)+1:-len('.jpeg')-5]
            num = int(filename[-len('.jpeg')-4:-len('.jpeg')])           
            
            outputPath = join(root, outputFolder, subPath)
            os.makedirs(outputPath, exist_ok=True) 
            
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8) as face_detection:
                results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.detections:
                    index_detection = 0
                    max_score_detection = results.detections[0].score[0]
                    for index in range(1, len(results.detections)):
                        if results.detections[index].score[0] > max_score_detection:
                            index_detection = index
                            max_score_detection = results.detections[index].score[0]
                    
                    detection = results.detections[index_detection]
                    bb = detection.location_data.relative_bounding_box
                    xmin = int(bb.xmin*img.shape[1]) - 50
                    ymin = int(bb.ymin*img.shape[0]) - 50
                    width, height = int(bb.width*img.shape[1]), int(bb.height*img.shape[0])
                    xmax = int(bb.xmin*img.shape[1]) + width + 50
                    ymax = int(bb.ymin*img.shape[0]) + height + 50
                    face = img[ymin:ymax, xmin:xmax].copy()
                    
                    kp_list = []
                    for kp in detection.location_data.relative_keypoints:
                        x = int(kp.x * img.shape[1])- xmin
                        y = int(kp.y * img.shape[0])- ymin
                        kp_list.append((x,y))
                    
                    landmarks = extract_face_landmarks(face)

                    if landmarks is None:
                        print(filename)

                    elif len(landmarks) == 68:
                        json_data = {
                                    'bb': [xmin, ymin, xmax, ymax],
                                    'lm68': landmarks.tolist(),
                                    'kp6': kp_list
                                    }
                        with open(join(outputPath, f'{num:05d}.json'), 'w') as f:  
                            json.dump(json_data, f)

        