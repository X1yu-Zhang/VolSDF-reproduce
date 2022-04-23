import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def uniform_sampling(near, far, num):
    t = near + (far - near) * torch.linspace(0,1, steps=num) 
    return t

def inverse_CDF_sampling(cdf, u, bins):
    u = u.contiguous()
    idx_x = torch.searchsorted(cdf, u, right = True)
    left = torch.max(torch.zeros_like(idx_x), idx_x - 1)
    right = torch.min(torch.ones_like(idx_x) * (cdf.shape[-1]-1), idx_x)
    idx = torch.stack([left, right], dim = -1)

    expand_shape = list(idx.shape[:-1]) + [cdf.shape[-1]]
    sample_interval = torch.gather(bins[:, None, :].expand(expand_shape), 2, idx)
    cdf_interval = torch.gather(cdf[:, None, :].expand(expand_shape), 2, idx)

    proportion_interval = cdf_interval[..., 1] - cdf[..., 0]
    proportion_interval[proportion_interval < 1e-5] = 1
    proportion_interval = (u - cdf_interval[..., 0]) / proportion_interval

    samples = sample_interval[..., 0] + proportion_interval * (sample_interval[..., 1] - sample_interval[..., 0])

    return samples

def get_d_start(delta, d):

    d_ip1 = torch.abs(d[1:])
    d_i   = torch.abs(d[:-1])

    s = 0.5*(delta + d_ip1 + d_i)
    h = torch.sqrt(s*(s-d_ip1)*(s-d_i)*(s-delta)) * 0.5 / delta
    d_min = torch.minimum(d_ip1, d_i)
    mask = 1 - (d_ip1 + d_i <= delta )
    d_star = mask * torch.where(torch.abs(d_i**2-d_ip1**2) >= delta **2, d_min, h)
    
    d_star = (d[:,1:].sign() * d[:,:-1].sign() == 1) * d_star
    return d_star

def get_error_bound(delta, beta, d_star, density):
    dE = delta ** 2 * laplace(d_star, beta) / ( 4 * beta **2 )
    dR = delta * density[...,:-1]
    E = torch.cumsum(dE, dim = -1)[:-1]
    R = torch.cat(torch.zeros_like(dR[...,0]), torch.cumsum(dR, dim = -1)[...,:-1],dim = -1)
    B = torch.exp(-R) * (torch.clamp(torch.exp(E), max=1.e6) - 1)
    return B


def sampling_algorithm(near, far, rays_o, rays_d, model, epsilon, N_init, N_sample, N_sample_extra, N_final, max_iter, search_iter, render_only):
    samples = uniform_sampling(near, far, rays_o, rays_d, N_init)
    beta = torch.abs(model.beta).detach()

    T = samples
    dist = T[1:] - T[:-1]
    beta_p = torch.sqrt(1/(1+torch.log(1+epsilon))*(torch.sum(dist**2, dim = -1)))
    beta_p = beta_p.expand([rays_o.shape[0], 1])
    
    iter = 0
    converge = False
    t = samples.expand([rays_o.shape[0], N_init])
    t_idx = None
    sampels = None

    while iter < max_iter and not converge:

        if samples is None:
            rays_sample = get_sample_pts(rays_o, rays_d, t)
        else:
            rays_sample = get_sample_pts(rays_o, rays_d, samples)

        sample_shape = rays_sample.shape
        rays_sample = rays_sample.reshape([-1,3])
        
        with torch.no_grad():
            sample_sdf = model.get_sdf(rays_sample).reshape(sample_shape[:-1])

        if t_idx is not None:
            sdf = torch.cat([sdf, sample_sdf], dim = -1)                        
            sdf = torch.gather(sdf, 1, t_idx)
        else:
            sdf = sample_sdf

        delta = t[1:] - t[:-1]
        d_star = get_d_start(delta, sdf)
        density = cal_density(sdf, beta)
        B = get_error_bound(delta, sdf, beta, d_star, density)
        B_T_beta_p = B.max(-1)[0]

        beta_p[B_T_beta_p <= epsilon] = beta
        beta_l, beta_r = beta.expand(beta_p.shape), beta_p
        for i in range(search_iter):
            beta_mid = (beta_l + beta_r) / 2
            B_beta_star = torch.max(get_error_bound(delta, beta_mid, d_star), dim = -1)[0]
            beta_l[B_beta_star <= epsilon] = beta_mid[B_beta_star <= epsilon]
            beta_r[B_beta_star > epsilon] = beta_mid[B_beta_star > epsilon]
        
        beta_p = beta_r
        
        iter += 1
        converge = (beta < beta_p).all()
        
        if not converge and iter < max_iter:
            pdf = B / (torch.sum(B, dim = -1, keepdim=True) + 1e-6)
            cdf = torch.cat([torch.zeros_like(pdf[...,0]), torch.cumsum(pdf, -1)], dim = -1)
            N = N_sample
        else:
            dR = delta * density[...,:-1]
            p = 1 - torch.exp(-dR)
            R = torch.cat([torch.zeros_like(dR[..., 0]), torch.cumsum(dR, dim = -1)[...,:-1]], dim = -1)
            T = torch.exp(-R)
            weight = p * T

            pdf = weight / (torch.sum(weight, dim=-1, keepdim=True) + 1e-5)
            cdf = torch.cat([torch.zeros_like(pdf[..., 0]), torch.cumsum(pdf, -1)], dim = -1)
            N = N_final
            
        if (not converge and iter < max_iter) or (not render_only):
            u = torch.linspace(0, 1, N)
        else:
            u = torch.rand([cdf.shape[0], N])

        samples = inverse_CDF_sampling(cdf, u, t)    

        if not converge and iter < max_iter:
            t, t_idx = torch.sort(torch.cat([t, samples], dim = -1), dim = -1)
        

    t = samples
    t_extra = torch.Tensor([near, far]).expand([rays_d.shape[0], 2])
    if N_sample_extra > 0:
        if render_only:
            idx = torch.randperm(t.shape[1])[:N_sample_extra]
        else:
            idx = torch.range(t.shape[1])[:N_sample_extra]
        t_extra = torch.cat([t, t_extra])
        
    t , _ = torch.sort(torch.cat([t, t_extra], dim = -1), dim=-1)
    
    idx = torch.randint(t.shape[-1], t.shape[0])
    t_loss = torch.gather(t, 1, idx[..., None])

    return t, t_loss