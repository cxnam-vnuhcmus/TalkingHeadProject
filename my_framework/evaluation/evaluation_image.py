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
    list_ssim = []
    list_fid = []
    list_psnr = []
    # folderA_size = len(glob(os.path.join(folderA, '*.jpg'), recursive=True))
    for imageA_path in tqdm(glob(os.path.join(folderA, '*.jpg'), recursive=True)):
        imageB_path = os.path.join(folderB, imageA_path.split('/')[-1])
        if os.path.exists(imageB_path):
            imageA = cv2.imread(imageA_path)
            imageB = cv2.imread(imageB_path)
            
            if eval_ssim:
                ssim = calculate_ssim(imageA, imageB, channel_axis=2)
                list_ssim.append(ssim)
            if eval_fid:
                fid = calculate_fid(imageA, imageB, dims=192)
                list_fid.append(fid)
            if eval_psnr:
                psnr = calculate_psnr(imageA, imageB)
                list_psnr.append(psnr)
    
    return {"ssim": sum(list_ssim)/len(list_ssim),
            "fid": sum(list_fid)/len(list_fid),
            "psnr": sum(list_psnr)/len(list_psnr),}
    
"""
Created on Thu Dec  3 00:28:15 2020
@author: Yunpeng Li, Tianjin University
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()