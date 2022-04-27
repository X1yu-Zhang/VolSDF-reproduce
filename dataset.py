from tkinter import W
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

from utils import get_rays_rgb

def load_dataset(path, test = 0):
    camera_path = os.path.join(path, 'cameras.npz')
    images_path = os.path.join(path, 'image')

    cameras = np.load(camera_path)
    images_list = os.listdir(images_path)
    images_list.sort()
    rays_rgb = []
    for image_name in images_list:
        idx = int(image_name.split('.')[0])
        scale_mat = cameras['scale_mat_{}'.format(idx)]
        world_mat = cameras['world_mat_{}'.format(idx)]

        P = world_mat @ scale_mat
        K, R, t = cv2.decomposeProjectionMatrix(P[:3])[:3] 
        K = K / K[2,2]
        t = t[:3] / t[3]
        # t = - R.T @ t
        pose = np.concatenate([R.T,t], axis=-1)
        img = cv2.cvtColor(cv2.imread(os.path.join(images_path, image_name)), cv2.COLOR_BGR2RGB)

        rays_rgb.append(get_rays_rgb(pose, img, K))
        if test:    
            break

    rays_rgb = np.concatenate(rays_rgb, axis=0, dtype=np.float32)
    if test:
        rays_rgb = rays_rgb[:test]
    return rays_rgb

class RaysDataset(Dataset):
    def __init__(self, rays):
        self.rays = rays
        self.len = rays.shape[0]
        pass    

    def __getitem__(self, index):
        return self.rays[index]

    def __len__(self):
        return self.len

