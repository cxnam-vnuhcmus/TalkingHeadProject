from tqdm import tqdm
from scipy.spatial import ConvexHull
import torch
from torch import nn
import numpy as np
import imageio
import os
import numpy as np
import torch
from glob import glob
from os.path import join
import json
import matplotlib.pyplot as plt

class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
      s = value
    else:
      s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s

class OneEuroFilter:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    
    cutoff = self.mincutoff + self.beta * np.abs(edx)
    return self.x_filter.process(x, self.compute_alpha(cutoff))

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def make_animation_smooth(source_image, driving_video, transformed_video, deco_out, kp_loss, generator, kp_detector, kp_detector_a, emo_detector, add_emo, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []

        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

        if not cpu:
            source = source.cuda()

        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        transformed_driving = torch.tensor(np.array(transformed_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector_a(deco_out[:,0])

        emo_driving_all = []
        features = []
        kp_driving_all = []
        for frame_idx in tqdm(range(len(deco_out[0]))):

            driving_frame = driving[:, :, frame_idx]
            transformed_frame = transformed_driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
                transformed_frame = transformed_frame.cuda()
            kp_driving = kp_detector_a(deco_out[:,frame_idx])
            kp_driving_all.append(kp_driving)
            if add_emo:
                value = kp_driving['value']
                jacobian = kp_driving['jacobian']
                emo_driving,_ = emo_detector(transformed_frame,value,jacobian)
                features.append(emo_detector.feature(transformed_frame).data.cpu().numpy())
            
                emo_driving_all.append(emo_driving)
        features = np.array(features)
        if add_emo:        
            one_euro_filter_v = OneEuroFilter(mincutoff=1, beta=0.2, dcutoff=1.0, freq=100)#1 0.4
            one_euro_filter_j = OneEuroFilter(mincutoff=1, beta=0.2, dcutoff=1.0, freq=100)#1 0.4

            for j in range(len(emo_driving_all)):
                emo_driving_all[j]['value']=one_euro_filter_v.process(emo_driving_all[j]['value'].cpu()*100)/100
                emo_driving_all[j]['value'] = emo_driving_all[j]['value'].cuda()
                emo_driving_all[j]['jacobian']=one_euro_filter_j.process(emo_driving_all[j]['jacobian'].cpu()*100)/100
                emo_driving_all[j]['jacobian'] = emo_driving_all[j]['jacobian'].cuda()


        one_euro_filter_v = OneEuroFilter(mincutoff=0.05, beta=8, dcutoff=1.0, freq=100)
        one_euro_filter_j = OneEuroFilter(mincutoff=0.05, beta=8, dcutoff=1.0, freq=100)

        for j in range(len(kp_driving_all)):
            kp_driving_all[j]['value']=one_euro_filter_v.process(kp_driving_all[j]['value'].cpu()*10)/10
            kp_driving_all[j]['value'] = kp_driving_all[j]['value'].cuda()
            kp_driving_all[j]['jacobian']=one_euro_filter_j.process(kp_driving_all[j]['jacobian'].cpu()*10)/10
            kp_driving_all[j]['jacobian'] = kp_driving_all[j]['jacobian'].cuda()


        for frame_idx in tqdm(range(len(deco_out[0]))):
            
            kp_driving = kp_driving_all[frame_idx]

       #     kp_driving_real = kp_detector(driving_frame)

       #     kp_driving['value'] = (1-opt.weight)*kp_driving['value'] + opt.weight*kp_driving_real['value']
       #     kp_driving['jacobian'] = (1-opt.weight)*kp_driving['jacobian'] + opt.weight*kp_driving_real['jacobian']

            if add_emo:
                emo_driving = emo_driving_all[frame_idx]
                kp_driving['value'][:,1] = kp_driving['value'][:,1] + emo_driving['value'][:,0]*0.2
                kp_driving['jacobian'][:,1] = kp_driving['jacobian'][:,1] + emo_driving['jacobian'][:,0]*0.2
                kp_driving['value'][:,4] = kp_driving['value'][:,4] + emo_driving['value'][:,1]
                kp_driving['jacobian'][:,4] = kp_driving['jacobian'][:,4] + emo_driving['jacobian'][:,1]
                kp_driving['value'][:,6] = kp_driving['value'][:,6] + emo_driving['value'][:,2]
                kp_driving['jacobian'][:,6] = kp_driving['jacobian'][:,6] + emo_driving['jacobian'][:,2]
                # kp_driving['value'][:,8] = kp_driving['value'][:,8] + emo_driving['value'][:,3]
                # kp_driving['jacobian'][:,8] = kp_driving['jacobian'][:,8] + emo_driving['jacobian'][:,3]
               
         
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions, features

import numbers
import random
import numpy as np
import PIL
import cv2
from skimage.transform import resize, rotate
from numpy import pad
import torchvision

import warnings

from skimage import img_as_ubyte, img_as_float


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
            ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def pad_clip(clip, h, w):
    im_h, im_w = clip[0].shape[:2]
    pad_h = (0, 0) if h < im_h else ((h - im_h) // 2, (h - im_h + 1) // 2)
    pad_w = (0, 0) if w < im_w else ((w - im_w) // 2, (w - im_w + 1) // 2)

    return pad(clip, ((0, 0), pad_h, pad_w, (0, 0)), mode='edge')


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]

        scaled = [
            resize(img, size, order=1 if interpolation == 'bilinear' else 0, preserve_range=True,
                   mode='constant', anti_aliasing=True) for img in clip
            ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


class RandomFlip(object):
    def __init__(self, time_flip=False, horizontal_flip=False):
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip

    def __call__(self, clip):
        if random.random() < 0.5 and self.time_flip:
            return clip[::-1]
        if random.random() < 0.5 and self.horizontal_flip:
            return [np.fliplr(img) for img in clip]

        return clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = resize_clip(
            clip, new_size, interpolation=self.interpolation)

        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of videos
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        clip = pad_clip(clip, h, w)
        im_h, im_w = clip.shape[1:3]
        x1 = 0 if h == im_h else random.randint(0, im_w - w)
        y1 = 0 if w == im_w else random.randint(0, im_h - h)
        cropped = crop_clip(clip, y1, x1, h, w)

        return cropped


class MouthCrop(object):
    """Extract random crop at the same location for a list of videos
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, center_x, center_y, mask_width, mask_height):
        

        self.center_x = center_x
        self.center_y = center_y
        self.mask_width = mask_width
        self.mask_height = mask_height

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        start_x = self.center_x - int(self.mask_width/2)
        start_y = self.center_y - int(self.mask_height/2) 
        end_x = start_x + self.mask_width
        end_y = start_y + self.mask_height
        # mask is all white
        # mask = 255*np.ones((mask_height, mask_width, 3), dtype=np.uint8)
        # mask is uniform noise
        cropped = []
        for i in range(len(clip)):
            mask = np.random.rand(self.mask_height, self.mask_width, 3)
            img = clip[i].copy()
            img[start_y:end_y, start_x:end_x, :] = mask
        
            cropped.append(img)
        cropped = np.array(cropped)
        return cropped

class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [rotate(image=img, angle=angle, preserve_range=True) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated

class RandomPerspective(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, pers_num, enlarge_num):
        self.pers_num = pers_num
        self.enlarge_num = enlarge_num

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        out = clip
        for i in range(len(clip)):
            self.pers_size = np.random.randint(20, self.pers_num) * pow(-1, np.random.randint(2))
            self.enlarge_size = np.random.randint(20, self.enlarge_num) * pow(-1, np.random.randint(2))
            h, w, c = clip[i].shape
            crop_size=256
            dst = np.array([
                [-self.enlarge_size, -self.enlarge_size],
                [-self.enlarge_size + self.pers_size, w + self.enlarge_size],
                [h + self.enlarge_size, -self.enlarge_size],
                [h + self.enlarge_size - self.pers_size, w + self.enlarge_size],], dtype=np.float32)
            src = np.array([[-self.enlarge_size, -self.enlarge_size], [-self.enlarge_size, w + self.enlarge_size],
                        [h + self.enlarge_size, -self.enlarge_size], [h + self.enlarge_size, w + self.enlarge_size]]).astype(np.float32())
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(clip[i], M, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
            out[i] = warped

        return out


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + img_transforms + [np.array,
                                                                                                     img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_clip = []
                for img in clip:
                    jittered_img = img
                    for func in img_transforms:
                        jittered_img = func(jittered_img)
                    jittered_clip.append(jittered_img.astype('float32'))
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all videos
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip


class AllAugmentationTransform:
    def __init__(self, crop_mouth_param = None, resize_param=None, rotation_param=None, perspective_param=None, flip_param=None, crop_param=None, jitter_param=None):
        self.transforms = []
        if crop_mouth_param is not None:
            self.transforms.append(MouthCrop(**crop_mouth_param))
        
        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))

        if rotation_param is not None:
            self.transforms.append(RandomRotation(**rotation_param))
        
        if perspective_param is not None:
            self.transforms.append(RandomPerspective(**perspective_param))

        if resize_param is not None:
            self.transforms.append(RandomResize(**resize_param))

        if crop_param is not None:
            self.transforms.append(RandomCrop(**crop_param))

        if jitter_param is not None:
            self.transforms.append(ColorJitter(**jitter_param))
      
    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip
    
import os
import imageio

def create_video(video_frames, output_path, fps=25):
    imageio.mimsave(output_path, video_frames, fps=fps)

def add_audio(video_path, audio_path, output_path):
    command = 'ffmpeg -i ' + video_path  + ' -i ' + audio_path + ' -vcodec copy  -acodec copy -y  ' + output_path
    print (command)
    os.system(command)
    
def read_data_from_path(mfcc_path=None, lm_path=None, face_path=None, start=None, end=None):
    mfcc_data_list = None
    lm_data_list = None
    face_data_list = None
    
    if mfcc_path is not None:
        mfcc_data_list = []
        mfcc_list = sorted(glob(join(mfcc_path, '*.npy')))
        if start is None and end is None:
            start = 0
            end = 1
        else:
            start, end = int(start), int(end)
        for index in range(start,end):
            mfcc_file = mfcc_list[index]
            mfcc_data = np.load(mfcc_file)
            mfcc_data = np.expand_dims(mfcc_data, axis=0)
            mfcc_data_list.append(mfcc_data)
        mfcc_data_list = np.vstack(mfcc_data_list).astype(np.float32)
    
    if lm_path is not None:
        lm_data_list = []
        bb_list = []
        lm_list = sorted(glob(join(lm_path, '*.json')))
        if start is None and end is None:
            start = 0
            end = 1
        else:
            start, end = int(start), int(end)
        for index in range(start,end):
            try:
                lm_file = os.path.join(lm_path, f'{index+1:05d}.json')
            except Exception as e:
                print(f'{lm_path} | {index}')
            with open(lm_file, 'r') as f:
                data = json.load(f)
                # lm_data = data['lm68']
                lm_data = data['lm68'] * np.asarray(256/(data['bb'][2] - data['bb'][0]))
                bb_list.append(data['bb'])
            lm_data_list.append([lm_data])
        lm_data_list = np.vstack(lm_data_list).astype(np.float32)
        lm_data_list = lm_data_list.reshape(*lm_data_list.shape[0:-2], -1)
        
    if face_path is not None:
        face_data_list = []
        if start is None and end is None:
            start = 0
            end = 1
        else:
            start, end = int(start), int(end)
        for index in range(start,end):
            try:
                face_file = os.path.join(face_path, f'{index+1:05d}.jpg')
            except Exception as e:
                print(f'{face_path} | {index}')
            image = imageio.imread(face_file)
            image = image / 255.0
            face_data_list.append([image])
        face_data_list = np.vstack(face_data_list).astype(np.float32)
    return {
        'mfcc_data_list': mfcc_data_list, 
        'lm_data_list': lm_data_list, 
        'face_data_list': face_data_list
    }
    
def save_model(model, epoch, optimizer=None, save_file='.'):
    dir_name = os.path.dirname(save_file)
    os.makedirs(dir_name, exist_ok=True)
    if optimizer is not None:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, str(save_file))
    else:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict()
        }, str(save_file))
        
def load_model(model, optimizer=None, save_file='.'):
    if next(model.parameters()).is_cuda and torch.cuda.is_available():
        checkpoint = torch.load(save_file, map_location=f'cuda:{torch.cuda.current_device()}')
    else:
        checkpoint = torch.load(save_file, map_location='cpu')
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch = checkpoint["epoch"]
    print(f"Load pretrained model at Epoch: {epoch}")
    return epoch

def save_plots(train_acc, val_acc, train_loss, val_loss, save_path='.'):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        val_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path,'accuracy.png'))
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        val_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path,'loss.png'))
    
def compute_FeatureMatching_loss(self, pred_fake, pred_real):
    # GAN feature matching loss
    # feature: 25, 512, 1, 1
    loss_FM = torch.zeros(1)
    feat_weights = 4.0 / (self.opt.n_layers_D + 1)
    D_weights = 1.0 / self.opt.num_D
    for i in range(min(len(pred_fake), self.opt.num_D)):
        for j in range(len(pred_fake[i])):
            loss_FM += D_weights * feat_weights * \
                nn.L1Loss()(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
    
    return loss_FM

from torchvision import models
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return 
    
class VGGLoss(nn.Module):
    def __init__(self, model=None):
        super(VGGLoss, self).__init__()
        if model is None:
            self.vgg = Vgg19()
        else:
            self.vgg = model

        self.vgg.cuda()
        # self.vgg.eval()
        self.criterion = nn.L1Loss()
        # self.style_criterion = StyleLoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        # self.style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        # self.weights = [5.0, 1.0, 0.5, 0.4, 0.8]
        # self.style_weights = [10e4, 1000, 50, 15, 50]

    def forward(self, x, y, style=False):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if style:
            # return both perceptual loss and style loss.
            style_loss = 0
            for i in range(len(x_vgg)):
                this_loss = (self.weights[i] *
                             self.criterion(x_vgg[i], y_vgg[i].detach()))
                # this_style_loss = (self.style_weights[i] *
                #                    self.style_criterion(x_vgg[i], y_vgg[i].detach()))
                loss += this_loss
                # style_loss += this_style_loss
            # return loss, style_loss
            return loss

        for i in range(len(x_vgg)):
            this_loss = (self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()))
            loss += this_loss
        return loss