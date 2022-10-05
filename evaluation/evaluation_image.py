from skimage.metrics import structural_similarity as ssim
from evaluation.fid import FidScore

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
