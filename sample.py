import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def uniform_sampling(near, radius, num, rays_o, rays_d, take_sphere_intersection, render_only, device):
    num_rays = rays_o.shape[0]
    near = torch.ones([num_rays, 1], device=device) * near

    # if take_sphere_intersection:
    #     far = None
    # else:
    far = torch.ones([num_rays, 1], device=device) * radius * 2

    t = near + (far - near) * torch.linspace(0, 1, steps=num, device= device)
    if not render_only:
        mids = (t[..., 1:] + t[..., :-1] ) /2
        right = torch.cat([mids, t[..., -1, None]], dim=-1)
        left = torch.cat([t[..., 0, None], mids], dim = -1)
        t = left + (right - left) * torch.rand(t.shape, device=device)

    return t


def inverse_CDF_sampling(cdf, u, bins, device):
    u = u.expand([cdf.shape[0], u.shape[0]])
    u = u.contiguous()
    idx_x = torch.searchsorted(cdf, u, right=True)
    left = torch.max(torch.zeros_like(idx_x, device=device), idx_x - 1)
    right = torch.min(torch.ones_like(idx_x, device=device) * (cdf.shape[-1] - 1), idx_x)
    idx = torch.stack([left, right], dim=-1)

    expand_shape = list(idx.shape[:-1]) + [cdf.shape[-1]]
    sample_interval = torch.gather(bins[:, None, :].expand(expand_shape), 2, idx)
    cdf_interval = torch.gather(cdf[:, None, :].expand(expand_shape), 2, idx)

    proportion_interval = cdf_interval[..., 1] - cdf_interval[..., 0]
    proportion_interval = torch.where(proportion_interval < 1e-5, torch.ones_like(proportion_interval), proportion_interval)
    proportion_interval = (u - cdf_interval[..., 0]) / proportion_interval
    samples = sample_interval[..., 0] + proportion_interval * (sample_interval[..., 1] - sample_interval[..., 0])
    return samples


def get_d_start(delta, d):
    d_ip1 = torch.abs(d[..., 1:])
    d_i = torch.abs(d[..., :-1])
    s = 0.5 * (delta + d_ip1 + d_i)
    under_sqrt = s * (s - d_ip1) * (s - d_i) * (s - delta)
    d_min = torch.minimum(d_ip1, d_i)
    mask1 = d_ip1 + d_i <= delta
    mask2 = torch.abs(d_i**2 - d_ip1**2) >= delta ** 2

    d_star = torch.zeros_like(delta)

    d_star[mask2] = d_min[mask2]
    mask = ~(mask1 | mask2) & (delta < d_i + d_ip1)
    d_star[mask] = 2 * torch.sqrt(under_sqrt[mask]) / (delta[mask] + 1e-6)

    d_star = (torch.sign(d[..., 1:]) * torch.sign(d[..., :-1]) == 1) * d_star

    return d_star


def get_error_bound(delta, sdf, beta, d_star):
    density = cal_density(sdf, beta)
    dE = delta ** 2 * laplace(d_star, beta) / (4 * beta ** 2)
    dR = delta * density[..., :-1]
    E = torch.cumsum(dE, dim=-1)
    R = torch.cat([torch.zeros_like(dR[..., 0, None]), torch.cumsum(dR, dim=-1)[..., :-1]], dim=-1)
    B = torch.exp(-R) * (torch.clamp(torch.exp(E), max=1.e6) - 1)
    return B

def get_max_error_bound(delta, sdf, beta, d_star):
    return torch.max(get_error_bound(delta, sdf, beta, d_star), dim = -1)[0]

def sampling_algorithm(rays_o, rays_d, model, near, radius, epsilon, N_init, N_sample, N_sample_extra, N_final, max_iter,
                       search_iter, render_only, device, take_sphere_intersection = False, **config):

    samples = uniform_sampling(near, radius, N_init, rays_o, rays_d, take_sphere_intersection, render_only, device)
    beta = torch.abs(model.beta).detach() + 1e-3
    epsilon = torch.Tensor([epsilon]).to(device)

    T = samples
    dist = T[..., 1:] - T[..., :-1]
    beta_p = torch.sqrt(1 / (1 + torch.log(1 + epsilon)) * (torch.sum(dist ** 2, dim=-1)))
    beta_p = beta_p[..., None]
    beta = beta.repeat([rays_o.shape[0], 1])
    iter = 0
    converge = False
    t = samples
    t_idx = None
    samples = None

    while iter < max_iter and not converge:

        if samples is None:
            rays_sample = get_sample_pts(rays_o, rays_d, t)
        else:
            rays_sample = get_sample_pts(rays_o, rays_d, samples)

        sample_shape = rays_sample.shape
        rays_sample = rays_sample.reshape([-1, 3])
        with torch.no_grad():
            sample_sdf = model.get_sdf(rays_sample).reshape(sample_shape[:-1])
        # break
        if t_idx is not None:
            sdf = torch.cat([sdf, sample_sdf], dim=-1)
            sdf = torch.gather(sdf, 1, t_idx)
        else:
            sdf = sample_sdf

        delta = t[..., 1:] - t[..., :-1]
        d_star = get_d_start(delta, sdf)
        B_beta = get_max_error_bound(delta, sdf, beta, d_star)

        beta_p[B_beta <= epsilon] = beta[B_beta <= epsilon]
        beta_l, beta_r = beta.expand(beta_p.shape), beta_p
        for i in range(search_iter):
            beta_mid = (beta_l + beta_r) / 2
            B_beta_star = get_max_error_bound(delta, sdf, beta_mid, d_star)
            beta_l[B_beta_star <= epsilon] = beta_mid[B_beta_star <= epsilon]
            beta_r[B_beta_star > epsilon] = beta_mid[B_beta_star > epsilon]

        beta_p = beta_r
        iter += 1
        converge = (beta_p <= beta).all()

        if not converge and iter < max_iter:
            B = get_error_bound(delta, sdf, beta_p, d_star)
            pdf = B / (torch.sum(B, dim=-1, keepdim=True) + 1e-6)
            cdf = torch.cat([torch.zeros_like(pdf[..., 0, None]), torch.cumsum(pdf, -1)], dim=-1)
            N = N_sample
        else:
            density = cal_density(sdf, beta_p)
            dR = delta * density[..., :-1]
            p = 1 - torch.exp(-dR)
            R = torch.cat([torch.zeros_like(dR[..., 0, None]), torch.cumsum(dR, dim=-1)[..., :-1]], dim=-1)
            T = torch.exp(-R)
            weight = p * T

            pdf = weight / (torch.sum(weight, dim=-1, keepdim=True) + 1e-5)
            cdf = torch.cat([torch.zeros_like(pdf[..., 0,None]), torch.cumsum(pdf, -1)], dim=-1)
            N = N_final

        if (not converge and iter < max_iter) or (not render_only):
            u = torch.linspace(0, 1, N, device=device)
        else:
            u = torch.rand([cdf.shape[0], N], device=device)
        samples = inverse_CDF_sampling(cdf, u, t, device)

        if not converge and iter < max_iter:
            t, t_idx = torch.sort(torch.cat([t, samples], dim=-1), dim=-1)

    t = samples
    t_extra = torch.Tensor([near, 2*radius]).expand([rays_d.shape[0], 2]).to(device)
    if N_sample_extra > 0:
        if render_only:
            idx = torch.randperm(t.shape[1], device=device)[:N_sample_extra]
        else:
            idx = torch.arange(0, t.shape[1], device=device)[:N_sample_extra]
        t_extra = torch.cat([t[..., idx.long()], t_extra], dim = -1)

    t, _ = torch.sort(torch.cat([t, t_extra], dim=-1), dim=-1)
    idx = torch.randint(t.shape[1], size=(t.shape[0], 1), device=device)
    t_loss = torch.gather(t, 1, idx)

    return t, t_loss

# def sample_bg(N_sample_bg, )