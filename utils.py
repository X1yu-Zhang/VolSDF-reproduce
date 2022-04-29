import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cal_density = lambda d, beta:1/beta*(0.5+torch.sign(-d)*0.5*(1-laplace(-d, beta)))

laplace = lambda s, beta: torch.exp(-torch.abs(s)/beta)

get_sample_pts = lambda rays_o, rays_d, t: rays_o[:, None, :] + t[..., None] * rays_d[:, None, :]

to8b = lambda x:(255*np.clip(x, 0, 1)).astype(np.uint8)

import numpy as np
from model import VolSDF, GeometryNetwork, RadienceFieldNetwork, NeRF
import configargparse

def get_rays_rgb(pose, image, K):
    H, W = image.shape[:-1]
    d = get_xyz(H, W, K)
    d = np.broadcast_to(d, image.shape) # [H, W, 3]
    d = (pose[None, None, :3, :3] @ d[...,None]).squeeze(axis = -1) # [H, W, 3]
    d = d / np.linalg.norm(d, axis = -1, keepdims=True)
    origins = pose[..., -1] # [3]
    origins = origins[None, None, :]
    origins = np.broadcast_to(origins, d.shape)
    rays_rgb = np.concatenate([origins, d, image], axis = -1).reshape([-1,9])
    return rays_rgb

def get_rays_with_pose(H, W, K, pose):
    xyz = get_xyz(H, W, K)
    pose = pose[:3,:4]
    # print(pose.shape, xyz.shape)
    d = (pose[:3,:3] @ xyz[...,None]).squeeze(axis = -1)
    o = pose[None, None, :, 3]
    o = np.broadcast_to(o, d.shape)
    d = d / np.linalg.norm(d, axis = -1, keepdims=True)
    rays = np.concatenate([o,d], axis = -1).reshape([-1,6])
    return rays

def get_xyz(H, W, K):
    j, i = np.mgrid[0:H, 0:W].astype(np.int32)
    uv = np.stack([i,j,np.ones_like(i)], axis = -1)
    uv = np.linalg.inv(K) @ uv[..., None]
    uv = uv.squeeze(-1)
    return uv

def create_model(**config):
    r = config['radius']
    beta = config['beta']
    lr = config['lr']
    device = config['device']
    posit_network = GeometryNetwork(**config['geometry_config']).to(device)
    render_network = RadienceFieldNetwork(**config['radiance_config']).to(device)

    if config['bg_render']:
        NeRF = NeRF(**config['nerf_config']).to(device)
    else:
        NeRF = None

    start = 0
    if config['pretrained_model'] is not None:
        print("loading pretrained model ...")
        state = torch.load(config['pretrained_model'])
        posit_network.load_state_dict(state['geometry_network'])   
        render_network.load_state_dict(state['rendering_network'])
        beta = state['beta']
        start = state['step']
        pass

    model = VolSDF(posit_network, render_network, NeRF, r, float(beta))
    grad_var = list(posit_network.parameters()) + list(render_network.parameters()) + [model.beta]
    optimizer = optim.Adam(params=grad_var, lr=lr, betas=(0.9, 0.999))

    if config['pretrained_model'] is not None: 
        optimizer.load_state_dict(state['optimizer'])

    print("Done!")
    return optimizer, model, start


def config():
    args = configargparse.ArgumentParser()
    args.add_argument("--config", is_config_file=True) 
    args.add_argument("--device", type=str, default='cuda' )
    
    ## dataset config
    args.add_argument("--datadir", type=str, default='./data')
    args.add_argument("--datatype", choices=['DTU', 'BlendedMVS'], default="DTU")
    args.add_argument("--scan_id", type=int, default=65)
    args.add_argument("--white_bkgd", action="store_true")
    
    ## sampling algorithm parameters
    args.add_argument("--N_init", type=int, default=128)
    args.add_argument("--N_sample", type=int, default=64)
    args.add_argument("--N_final", type=int, default=64)
    args.add_argument("--N_sample_extra", type=int, default=32)
    args.add_argument("--epsilon", type=float, default=0.1)
    args.add_argument("--max_iter", type=int, default=5)
    args.add_argument("--search_iter", type=int, default=10)
    args.add_argument("--near", type=float, default=0)

    ## network parameters
    args.add_argument("--o_freq", type=int, default=6)
    args.add_argument("--d_freq", type=int, default=4)
    args.add_argument("--feature_dim", type=int, default=256)
    args.add_argument("--radius", type=float, default=3.)
    args.add_argument("--beta", type=float, default=0.1)
    args.add_argument("--scale",type=float, default=1)

    args.add_argument("--geo_D", type=int, default=8)
    args.add_argument("--geo_W", type=int, default=256)
    args.add_argument("--rad_D", type=int, default=4)
    args.add_argument("--rad_W", type=int, default=256)
    args.add_argument("--geo_skip", type=list, default=[4])

    args.add_argument("--bg_render", action="store_true")
    
    ## training parameters
    args.add_argument("--lr", type=float, default=5e-4)
    args.add_argument("--lr_decay", type=float, default=250)
    args.add_argument("--training_iters", type=int, default=20000)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--lambda", type=float, default=0.1)

    args.add_argument("--i_save", type=int, default=500)
    args.add_argument("--ckpt", type=str, default='./ckpt')
    args.add_argument("--i_show_loss", type=int, default=10)

    args.add_argument("--test", type=int, default= 0)
    args.add_argument("--render_only", action="store_true")

    args.add_argument("--pretrained_model", type=str, default=None)
    args.add_argument("--output", type=str, default='./output')

    args = args.parse_args()
    
    return split_config(vars(args))

    
def split_config(config):

    training_config = {
        "lr": config['lr'],
        "lr_decay": config['lr_decay'],
        "l": config['lambda'],
        "N_iters" : config['training_iters'],
        "batch_size" : config['batch_size'],
        "i_save": config['i_save'],
        "ckpt": config['ckpt'],
        "i_show_loss": config['i_show_loss']
    }
    
    dataset_config = {
        "path" : config['datadir'],
        "datatype" : config['datatype'],
        "scan_id": config['scan_id'],
        "test" : config['test']
    }

    sampling_config = {
        "near": config['near'],
        "radius" : config['radius'],
        "epsilon": config['epsilon'],
        "N_init" : config['N_init'],
        "N_sample": config['N_sample'],
        "N_sample_extra": config['N_sample_extra'],
        "N_final": config['N_final'],
        "max_iter" : config['max_iter'],
        "search_iter": config['search_iter'],
        "bg_render": config['bg_render'],
        "render_only": config['render_only'],
        "device": config['device']
    }

    rendering_config = {
        "white_bkgd": config['white_bkgd'],
        "sampling_config": sampling_config,
        "bg_render": config['bg_render'],
        "device": config['device'], 
        "render_only": config['render_only'],
    }

    geometry_config = {
        "input_dim":3,
        "embed_length": config['o_freq'],
        "output_dim": config['feature_dim'],
        "D": config['geo_D'],
        "W": config['geo_W'],
        "skip_connect": config['geo_skip'],
        "device": config['device']
    }

    radiance_config = {
        "input_dim":3,
        "embed_length": config['d_freq'],
        "feature_length": config['feature_dim'],
        "D": config['rad_D'],
        "W": config['rad_W'],
        "device": config['device']
    }

    nerf_config = {
        "input_ch": 4,
        "input_ch_view": 3,
        "o_freq": 10,
        "d_freq": 4,
    }

    model_config = {
        "geometry_config": geometry_config,
        "radiance_config": radiance_config,
        "nerf_config": nerf_config,
        "lr": config['lr'],
        "radius": config['radius'],
        "beta": config['beta'],
        "bg_render": config['bg_render'],
        "pretrained_model": config['pretrained_model'],
        "device": config['device']
    }
    
    output_config = {
        "model_config": model_config,
        "dataset_config": dataset_config,
        "rendering_config": rendering_config,           
        "device": config['device'],
        "render_only": config['render_only'],
        "output": config['output'],
        "scan_id": config['scan_id']
    }
    
    output_config.update(training_config)

    return output_config

