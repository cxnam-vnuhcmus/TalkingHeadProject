from skimage.metrics import structural_similarity as ssim
from evaluation.fid import FidScore
import cpbd
from math import log10, sqrt
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2
import os

def calculate_ssim(image1, image2, channel_axis=None):
    r"""Compute Structural Similarity Index Measure
    Args:
        image1          :   numpy.ndarray([H, W, C], dtype=np.float32)
        image2          :   numpy.ndarray([H, W, C], dtype=np.float32)
        channel_axis    :   (int) axis of color channel
    Returns:
        SSIM            :   float [-1, 1] (↑)
    """
    return ssim(image1, image2, channel_axis=channel_axis)

def calculate_fid(image1, image2, dims=2048):
    r"""Compute Frechet Inception Distance
    Args:
        image1          :   numpy.ndarray([H, W, C], dtype=np.float32)
        image2          :   numpy.ndarray([H, W, C], dtype=np.float32)
        dims            :   (int) one of [64, 192, 768, 2048]
    Returns:
        FID             :   float (↓)
    """
    fid = FidScore(dims=dims)
    return fid.calculate_fid_score_one_shot(image1, image2)

def calculate_psnr(image1, image2):
    r"""Compute Peak Signal-to-Noise Ratio
    Args:
        image1          :   numpy.ndarray([H, W, C], dtype=np.float32)
        image2          :   numpy.ndarray([H, W, C], dtype=np.float32)
    Returns:
        PSNR            :   float (↑)
    """
    mse = np.mean((image1 - image2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def calculate_cpbd(image_grayscale):  
    r"""Compute Cumulative Probability of Blur Detection
    Args:
        image_grayscale :   numpy.ndarray([H, W], dtype=np.float32)
    Returns:
        CPBD            :   float (↑)
    """  
    return cpbd.compute(image_grayscale)

def calculate_folder_image(folderA, folderB, eval_ssim=False, eval_fid=False, eval_psnr=False):
    total_ssim = 0
    total_fid = 0
    total_psnr = 0
    folderA_size = len(glob(os.path.join(folderA, '*.jpg'), recursive=True))
    for imageA_path in tqdm(glob(os.path.join(folderA, '*.jpg'), recursive=True)):
        imageB_path = os.path.join(folderB, imageA_path.split('/')[-1])
        imageA = cv2.imread(imageA_path)
        imageB = cv2.imread(imageB_path)
        
        if eval_ssim:
            ssim = calculate_ssim(imageA, imageB, channel_axis=2)
            total_ssim = total_ssim + ssim
        if eval_fid:
            fid = calculate_fid(imageA, imageB, dims=192)
            total_fid = total_fid + fid
        if eval_psnr:
            psnr = calculate_psnr(imageA, imageB)
            total_psnr = total_psnr + psnr
    
    return {"ssim": total_ssim/folderA_size,
            "fid": total_fid/folderA_size,
            "psnr": total_psnr/folderA_size,}