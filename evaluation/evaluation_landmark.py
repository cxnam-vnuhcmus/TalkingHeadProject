import numpy as np

def calculate_LMD(pred_landmark, gt_landmark, norm_distance=1.0):
    r"""Compute Landmark Distance by Normalize Mean Square Error

    Args:
        pred_landmark   :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        gt_landmark     :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        norm_distance   :   float
    Returns:
        LMD            :   float (↓)
    """
    euclidean_distance = np.sqrt(np.sum((pred_landmark - gt_landmark)**2, axis=3))
    norm_per_frame = np.mean(euclidean_distance, axis=2)
    norm_per_frame_norm = np.divide(norm_per_frame, norm_distance)
    norm_per_batch = np.mean(norm_per_frame_norm, axis=1)
    norm_all_batch = np.mean(norm_per_batch, axis=0)
    return norm_all_batch

def calculate_LMV(pred_landmark, gt_landmark, norm_distance=1.0):
    r"""Compute Landmark Velocity by Normalize Mean Square Error

    Args:
        pred_landmark   :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        gt_landmark     :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        norm_distance   :   float
    Returns:
        LMV            :   float (↓)
    """
    velocity_pred_landmark = pred_landmark[:, 1:, :, :] - pred_landmark[:, 0:-1, :, :]
    velocity_gt_landmark = gt_landmark[:, 1:, :, :] - gt_landmark[:, 0:-1, :, :]
    euclidean_distance = np.sqrt(np.sum((velocity_pred_landmark - velocity_gt_landmark)**2, axis=3))
    norm_per_frame = np.mean(euclidean_distance, axis=2)
    norm_per_frame_norm = np.divide(norm_per_frame, norm_distance)
    norm_per_batch = np.mean(norm_per_frame_norm, axis=1)
    norm_all_batch = np.mean(norm_per_batch, axis=0)
    return norm_all_batch