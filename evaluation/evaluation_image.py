from skimage.metrics import structural_similarity as ssim
from evaluation.fid import FidScore
import cpbd
from math import log10, sqrt
import numpy as np

def calculate_ssim(image1, image2, channel_axis=None):
    '''
    channel_axis: (int) axis of color channel
    '''
    return ssim(image1, image2, channel_axis=channel_axis)

def calculate_fid(image1, image2, dims=2048):
    '''
    dims: (int) one of [64, 192, 768, 2048]
    '''
    fid = FidScore(dims=dims)
    return fid.calculate_fid_score_one_shot(image1, image2)

def calculate_cpbd(image):    
    return cpbd.compute(image)

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr