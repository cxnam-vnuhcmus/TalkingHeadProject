import os
import pathlib

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
import cv2

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from .inception import InceptionV3


class FidScore(object):
    def __init__(self, paths=None, device=torch.device('cpu'), batch_size=50, dims=2048):
        self.paths, self.batch_size, self.device, self.dims = paths, batch_size, device, dims
        if paths is not None:
            if len(self.paths) != 2:
                raise ValueError("paths should be a list of source image folder and target image folder")

    def _imread(self, filename):
        """
        Loads an image file into a (height, width, 3) uint8 ndarray.
        """
        return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]

    def _get_activations(self, files, model):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                         Make sure that the number of samples is a multiple of
                         the batch size, otherwise some samples are ignored. This
                         behavior is retained to match the original FID score
                         implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the number
                         of calculated batches is reported.
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        model.eval()

        if self.batch_size > len(files):
            print('Warning: batch size is bigger than the data size. Setting batch size to data size')
            self.batch_size = len(files)

        pred_arr = np.empty((len(files), self.dims))

        for i in tqdm(range(0, len(files), self.batch_size)):
            print(f'\rPresent batch {i+1}/{self.batch_size}', end='', flush=True)
            start = i
            end = i + self.batch_size

            images = np.array([self._imread(str(f)).astype(np.float32) for f in files[start:end]])

            # Reshape to (n_images, 3, height, width)
            images = images.transpose((0, 3, 1, 2))
            images /= 255

            batch = torch.from_numpy(images).type(torch.FloatTensor)
            batch = batch.to(self.device)

            pred = model(batch)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

            print('done')
        return pred_arr

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def _calculate_activation_statistics(self, files, model):
        act = self._get_activations(files, model)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def _compute_statistics_of_path(self, path, model):
        if path.endswith('.npz'):
            f = np.load(path)
            m, s = f['mu'][:], f['sigma'][:]
            f.close()
        else:
            path = pathlib.Path(path)
            files = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.jpeg'))
            m, s = self._calculate_activation_statistics(files, model)
        return m, s

    def _calculate_fid_given_paths(self, paths):
        """Calculates the FID of two paths"""
        for p in paths:
            if not os.path.exists(p):
                raise RuntimeError('Invalid path: {p}')

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]

        model = InceptionV3([block_idx])
        model.to(self.device)

        m1, s1 = self._compute_statistics_of_path(paths[0], model)
        m2, s2 = self._compute_statistics_of_path(paths[1], model)
        fid_value = self._calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value

    def calculate_fid_score(self):
        return self._calculate_fid_given_paths(self.paths)

    def _calculate_activation_statistics_one_shot(self, image, model):
        image = cv2.resize(image, (self.dims, self.dims))
        image = np.expand_dims(image, 0)
        
        # Reshape to (n_images, 3, height, width)
        image = image.transpose((0, 3, 1, 2))
        image = image / 255.

        batch = torch.from_numpy(image).type(torch.FloatTensor)
        batch = batch.to(self.device)

        pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.cpu().data.numpy().reshape(pred.size(0), -1)

        m1 = np.mean(pred, axis=0)
        s1 = np.cov(pred, rowvar=False)
        return (m1, s1)

    def calculate_fid_score_one_shot(self, image1, image2):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]

        model = InceptionV3([block_idx])
        model.to(self.device)

        m1, s1 = self._calculate_activation_statistics_one_shot(image1, model)
        m2, s2 = self._calculate_activation_statistics_one_shot(image2, model)
        fid_value = self._calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

'''
How to use
import cv2
import numpy as np
import torch

image = cv2.imread('/root/Datasets/Features/M003/images/angry/level_1/00001/00001.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)

image2 = cv2.imread('/root/Datasets/Features/M003/images/happy/level_1/00002/00001.jpeg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
print(image2.shape)

from evaluation.fid import FidScore
fid = FidScore(dims=192)
score = fid.calculate_fid_score_one_shot(image, image2)
print(score)
'''

'''
from fid_score.fid_score import FiDScore
fid = FidScore(paths, device, batch_size)
score = fid.calculate_fid_score()

Arguments
fid = FidScore(paths, device, batch_size)
paths = ['path of source image dir', 'path of target image dir']
device = torch.device('cuda:0') or default: torch.device('cpu')
batch_size = batch size 
'''