import torch
from sample import sampling_algorithm
from utils import * 

def output2rgb(t, density, output):
    # t = torch.cat([t, torch.Tensor([1e10]).expand([t.shape[0], 1])], dim = -1)
    delta = t[1:] - t[:-1]
    p = -delta * density[...,:-1] # 1 ... m-1
    T = torch.cat([torch.zeros([p.shape[0], 1]), torch.cumprod(p[..., :-1], dim = -1)], dim = -1) # 1 ... m
    tau = torch.cat([(1 - p) * T[..., : -1] , T[..., -1]], dim = -1)
    rgb = torch.sum(tau * output, dim = -1)
    return rgb

def volume_rendering(rays_o, rays_d, near, far, **config):
    model = config['model']

    t, t_loss = sampling_algorithm(near, far, rays_o, rays_d)
    pts = get_sample_pts(rays_o, rays_d, t)
    pts_loss = get_sample_pts(rays_o, rays_d, t_loss)

    pts_shape = pts.shape
    density, output = model(pts.reshape([-1,3]))
    density = density.reshape(pts_shape[:-1])
    output = output.reshape(pts_shape[:-1])

    rgb = output2rgb(t, density, output)
    gradient = model.gradient(pts_loss)

    return rgb, gradient

def train():

    pass

def main():
    pass

if __name__ == "__main__":
    pass