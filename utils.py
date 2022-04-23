import torch
import torch.nn as nn
import torch.nn.functional as F

get_gradient = lambda x, y:torch.autograd.grad(
    outputs= y,
    inputs=x,
    grad_outputs=torch.ones_like(y, requires_grad=False, device=y.device),
    retain_graph=True,
    create_graph=True,
    only_inputs=True,
)[0]


cal_density = lambda d, beta:1/beta*(0.5+0.5*torch.sign(d)*(1-torch.exp(-torch.abs(d)/beta)))

laplace = lambda s, beta: torch.exp(-torch.aba(s)/beta)

get_sample_pts = lambda rays_o, rays_d, t: rays_o[:, None, :] + t[None, ...] * rays_d[:, None, :]

