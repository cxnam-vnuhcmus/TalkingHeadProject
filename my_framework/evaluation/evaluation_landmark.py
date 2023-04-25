import numpy as np
import json
from glob import glob
from tqdm import tqdm
import os
import torch

def calculate_LMD(pred_landmark, gt_landmark, norm_distance=1.0):
    r"""Compute Landmark Distance by Normalize Mean Square Error

    Args:
        pred_landmark   :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        gt_landmark     :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        norm_distance   :   float
    Returns:
        LMD            :   float (↓)
    """
    euclidean_distance = np.sqrt(np.sum((pred_landmark - gt_landmark)**2, axis=(pred_landmark.ndim - 1)))
    norm_per_frame = np.mean(euclidean_distance, axis=(pred_landmark.ndim - 2))
    lmd = np.divide(norm_per_frame, norm_distance)
    if (pred_landmark.ndim - 3) >= 0:
        lmd = np.mean(lmd, axis=(pred_landmark.ndim - 3))
    if (pred_landmark.ndim - 4) >= 0:
        lmd = np.mean(lmd, axis=(pred_landmark.ndim - 4))
    return lmd

def calculate_LMD_torch(pred_landmark, gt_landmark, norm_distance=1.0):
    r"""Compute Landmark Distance by Normalize Mean Square Error

    Args:
        pred_landmark   :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        gt_landmark     :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        norm_distance   :   float
    Returns:
        LMD            :   float (↓)
    """
    euclidean_distance = torch.sqrt(torch.sum((pred_landmark - gt_landmark)**2, dim=(pred_landmark.ndim - 1)))
    norm_per_frame = torch.mean(euclidean_distance, dim=(pred_landmark.ndim - 2))
    lmd = torch.divide(norm_per_frame, norm_distance)
    if (pred_landmark.ndim - 3) >= 0:
        lmd = torch.mean(lmd, dim=(pred_landmark.ndim - 3))
    if (pred_landmark.ndim - 4) >= 0:
        lmd = torch.mean(lmd, dim=(pred_landmark.ndim - 4))
    return lmd.item()

def calculate_rmse_torch(pred_landmark, gt_landmark):
    """Calculate Root Mean Square Error (RMSE) loss between predicted and gt landmarks"""
    # Calculate the squared errors
    squared_errors = torch.sum((pred_landmark - gt_landmark) ** 2, dim=3)

    # Calculate the mean of squared errors
    mse = torch.mean(squared_errors)

    # Calculate the RMSE
    rmse = torch.sqrt(mse)
    return rmse

def calculate_mae_torch(pred_landmark, gt_landmark):
    """Calculate Mean Average Error (MAE) loss between predicted and gt landmarks"""
    mae = torch.mean(torch.abs(pred_landmark - gt_landmark))
    return mae

def calculate_LMV(pred_landmark, gt_landmark, norm_distance=1.0):
    r"""Compute Landmark Velocity by Normalize Mean Square Error

    Args:
        pred_landmark   :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        gt_landmark     :   numpy.ndarray([BS, N_frame, N_point, feature_dim], dtype=np.float32)
        norm_distance   :   float
    Returns:
        LMV            :   float (↓)
    """
    if gt_landmark.ndim == 4:
        velocity_pred_landmark = pred_landmark[:, 1:, :, :] - pred_landmark[:, 0:-1, :, :]
        velocity_gt_landmark = gt_landmark[:, 1:, :, :] - gt_landmark[:, 0:-1, :, :]
    elif gt_landmark.ndim == 3:
        velocity_pred_landmark = pred_landmark[1:, :, :] - pred_landmark[0:-1, :, :]
        velocity_gt_landmark = gt_landmark[1:, :, :] - gt_landmark[0:-1, :, :]
        
    euclidean_distance = np.sqrt(np.sum((velocity_pred_landmark - velocity_gt_landmark)**2, axis=(pred_landmark.ndim - 1)))
    norm_per_frame = np.mean(euclidean_distance, axis=(pred_landmark.ndim - 2))
    lmv = np.divide(norm_per_frame, norm_distance)
    if (pred_landmark.ndim - 3) >= 0:
        lmv = np.mean(lmv, axis=(pred_landmark.ndim - 3))
    if (pred_landmark.ndim - 4) >= 0:
        lmv = np.mean(lmv, axis=(pred_landmark.ndim - 4))
    return lmv

def calculate_LMV_torch(pred_landmark, gt_landmark, norm_distance=1.0):
    if gt_landmark.ndim == 4:
        velocity_pred_landmark = pred_landmark[:, 1:, :, :] - pred_landmark[:, 0:-1, :, :]
        velocity_gt_landmark = gt_landmark[:, 1:, :, :] - gt_landmark[:, 0:-1, :, :]
    elif gt_landmark.ndim == 3:
        velocity_pred_landmark = pred_landmark[1:, :, :] - pred_landmark[0:-1, :, :]
        velocity_gt_landmark = gt_landmark[1:, :, :] - gt_landmark[0:-1, :, :]
            
    euclidean_distance = torch.sqrt(torch.sum((velocity_pred_landmark - velocity_gt_landmark)**2, dim=(pred_landmark.ndim - 1)))
    norm_per_frame = torch.mean(euclidean_distance, dim=(pred_landmark.ndim - 2))
    lmv = torch.div(norm_per_frame, norm_distance)
    if (pred_landmark.ndim - 3) >= 0:
        lmv = torch.mean(lmv, dim=(pred_landmark.ndim - 3))
    if (pred_landmark.ndim - 4) >= 0:
        lmv = torch.mean(lmv, dim=(pred_landmark.ndim - 4))
    return lmv

def calculate_folder_landmark(folderA, folderB, eval_lmd=True, eval_lmv=True):
    total_lmd = 0
    total_lmv = 0
    if eval_lmd:
        lmA_list = []
        lmB_list = []
        folderA_data = glob(os.path.join(folderA, '**/*.json'), recursive=True)
        for lmA_path in tqdm(folderA_data):
            lmB_path = lmA_path.replace(folderA, folderB)
            if os.path.exists(lmB_path):
                with open(lmA_path, 'r') as f:
                    lmA = json.load(f)
                    lmA_list.append(lmA)
                with open(lmB_path, 'r') as f:
                    lmB = json.load(f)
                    lmB_list.append(lmB)     
        lmA_list = np.asarray(lmA_list)
        lmB_list = np.asarray(lmB_list) 
        norm_distance = np.sqrt(np.sum((lmB_list[:,0] - lmB_list[:,16])**2, axis=1))
        total_lmd = calculate_LMD(lmA_list, lmB_list,norm_distance=norm_distance)
    if eval_lmv:
        list_lmv = []
        folderA_data = glob(os.path.join(folderA, '**/*.json'), recursive=True)
        folderA_data_new = []
        for data in folderA_data:
            data_new = data[:-11]
            if data_new not in folderA_data_new:
                folderA_data_new.append(data[:-11])
        folderA_data_new = sorted(folderA_data_new)
        for lmA_path in tqdm(folderA_data_new):
            lmB_path = lmA_path.replace(folderA, folderB)
            if os.path.exists(lmB_path):
                lmA = []
                lmB = []
                for lmA_subfolder in glob(os.path.join(lmA_path, '*.json')):
                    with open(lmA_subfolder, 'r') as f:
                        lmA_data = json.load(f)
                    lmB_subfolder = lmA_subfolder.replace(folderA, folderB)
                    if os.path.exists(lmB_subfolder):
                        lmA.append(lmA_data)
                        with open(lmB_subfolder, 'r') as f:
                            lmB.append(json.load(f))
                lmA = np.asarray(lmA)
                lmB = np.asarray(lmB)
                
                norm_distance = np.sqrt(np.sum((lmB[:,0] - lmB[:,16])**2, axis=0))[:-1]
                lmv = calculate_LMV(lmB, lmA, norm_distance=norm_distance)
                list_lmv.append(lmv)
        total_lmv = sum(list_lmv)/len(list_lmv)
    return {"lmd": total_lmd,
            "lmv": total_lmv}