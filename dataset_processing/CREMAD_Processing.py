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
import moviepy.editor as mv
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
import librosa
import python_speech_features

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default='VideoFlash')
parser.add_argument("--extract_audio", type=bool, default=False)
parser.add_argument("--extract_images", type=bool, default=False)
parser.add_argument("--extract_mfcc13", type=bool, default=False)
parser.add_argument("--extract_lm68", type=bool, default=False)
parser.add_argument("--extract_lm74", type=bool, default=True)
parser.add_argument("--extract_faces", type=bool, default=False)

if __name__ == '__main__':
    args = parser.parse_args()
    root = '/root/Datasets/CREMA-D/'
    os.makedirs(join(root, 'Features'), exist_ok=True)
    
    if (args.extract_audio):
        print('==Audio Extraction==')
        inputFolder = join(root, args.video_path)
        outputFolder = join(root, f'Features/audios')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder,'**/*.flv'), recursive=True))

        for filepath in tqdm(filelist):
            filename = os.path.basename(filepath)[:-len('.flv')]
            
            outputPath = join(outputFolder, f'{filename}.wav')

            # cmd = 'ffmpeg -i ' + filename + ' -ab 160k -ac 1 -ar 16000 -vn ' + outputPath
            # subprocess.call(cmd, shell=True)
            clip = mv.VideoFileClip(filepath)
            num_frames = round(clip.reader.fps*clip.reader.duration)
            clip.audio.write_audiofile(outputPath, fps=16000)
            
    if (args.extract_images):
        print('==Frames Extraction==')
        inputFolder = join(root, args.video_path)
        outputFolder = join(root, f'Features/images')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder,'**/*.flv'), recursive=True))

        for filepath in tqdm(filelist):
            filename = os.path.basename(filepath)[:-len('.flv')]
            
            outputPath = join(root, outputFolder, filename)
            os.makedirs(outputPath, exist_ok=True)
            
            clip = mv.VideoFileClip(filepath)
            clip = clip.set_fps(25)
            numframes = int(clip.duration * clip.fps)
            for i,frame in enumerate(clip.iter_frames()):
                if i <= numframes:
                    outputFramePath = join(outputPath, f'{(i+1):05d}.jpg')
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height = frame.shape[0]
                    x = int(np.floor((frame.shape[1]-height)/2))
                    frame = frame[:, x:x+height]
                    frame = cv2.resize(frame, (512,512), interpolation = cv2.INTER_AREA)
                    cv2.imwrite( outputFramePath, frame)
                    
    if args.extract_mfcc13:
        print('==MFCC(28,12) Extraction==')
        inputFolder = join(root, f'Features/audios')
        outputFolder = join(root, f'Features/mfccs')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder, '**/*.wav'), recursive=True))
        
        for idx, filepath in tqdm(enumerate(filelist)):
            filename = os.path.basename(filepath)[:-len('.wav')]            
            outputPath = join(root, outputFolder, filename)
            os.makedirs(outputPath, exist_ok=True)
            
            imagePath = outputPath.replace('mfccs', 'images')
            num_frames = len(os.listdir(imagePath))

            audio_signal, sr = librosa.load(filepath, sr=16000)
            win_length = sr//25
            for i in range(num_frames):
                try:
                    mfcc = python_speech_features.mfcc(audio_signal[i*win_length:i*win_length+win_length], \
                                                   sr,winlen=0.010,winstep=0.001)
                    np.save(join(outputPath, f'{i+1:05d}.npy'), mfcc[1:-2,1:])
                except:
                    print(filepath)
                    continue
    
    if (args.extract_lm74):
        print('==Landmark74 Extraction==')
        inputFolder = join(root, f'Features/images')
        outputFolder = join(root, f'Features/landmarks74')
        os.makedirs(outputFolder, exist_ok=True)
        filelist = sorted(glob(join(inputFolder,'**/*.jpg'), recursive=True))

        for filename in tqdm(filelist):
            subPath = filename[len(inputFolder)+1:-len('.jpg')-5]
            num = int(filename[-len('.jpg')-4:-len('.jpg')])           
            
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