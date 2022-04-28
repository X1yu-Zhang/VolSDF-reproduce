import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

get_grad = lambda x, y:torch.autograd.grad(
    outputs= y,
    inputs=x,
    grad_outputs=torch.ones_like(y, requires_grad=False, device=y.device),
    retain_graph=True,
    create_graph=True,
    only_inputs=True,
)[0]

class Embedding:
    def __init__(self, input_dim, length, include_input = True, log_sampling = True, device='cuda'):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.length = length
        self.functs = [torch.sin, torch.cos]
        self.output_dim = 2 * length * input_dim + include_input * self.input_dim
        if log_sampling:
            self.freq = torch.pow(2, torch.linspace(0, length - 1, steps = length, device=device) )
        else:
            self.freq = torch.linspace(1, 2**(length-1), steps = length, device=device)

    def embed(self, x):
        ##  x: [N_rays, 3]
        ## self.freq: [length]
        # print(x.shape)
        embed_vec = x[..., None] * self.freq # [N_rays, 3, length]
        embed_vec = torch.stack([func(embed_vec) for func in self.functs], dim = -1) # [N_rays, 3, length, 2]
        # print(embed_vec.shape)
        embed_vec = embed_vec.permute([0,2,3,1]).reshape([embed_vec.shape[0], -1])  # [N_rays, length, 2, 3] [N_rays, 3 * 2 * length]
        x = torch.cat([x, embed_vec], dim = -1)
        return x
    

class GeometryNetwork(nn.Module):
    def __init__(self, input_dim, embed_length, output_dim = 256, D = 8, W = 256, skip_connect = [4], r = 3, bound_scale = 1, bias = 0.6, device='cuda'):
        super(GeometryNetwork, self).__init__()
        
        self.skip_connect = skip_connect
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = Embedding(input_dim, embed_length, device=device)
        self.r = r
        self.bound_scale = bound_scale
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.embedding.output_dim, W)] + [nn.Linear(W,W-self.embedding.output_dim) if i+1 in self.skip_connect else nn.Linear(W,W) for i in range(1, D)]
        )

        for idx, layer in enumerate(self.pts_linears):
            if idx == D - 1:
                nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(W), std=0.0001)
                nn.init.constant_(layer.bias, -bias)
            elif idx == 0 and embed_length > 0:
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.weight[:, 3:], 0)
                nn.init.normal_(layer.weight[:, 3], 0, np.sqrt(2) / np.sqrt(W))
            elif idx in self.skip_connect and embed_length > 0:
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, 0, np.sqrt(2)/np.sqrt(W))
                nn.init.constant_(layer.weight[:, -(self.embedding.output_dim-3): ], 0)
            elif idx + 1 in self.skip_connect:
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, 0, np.sqrt(2)/np.sqrt(W-self.embedding.output_dim))
            else: 
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, 0, np.sqrt(2)/np.sqrt(W))

            layer = nn.utils.weight_norm(layer)
        self.feature_linear = nn.Linear(W, output_dim + 1)
        nn.init.constant_(self.feature_linear.bias, 0)
        nn.init.normal_(self.feature_linear.weight, 0, np.sqrt(2)/np.sqrt(W+1))
        self.feature_linear = nn.utils.weight_norm(self.feature_linear)
        self.softplus = nn.Softplus(beta = 100)

    def output(self, x):
        x = self.embedding.embed(x)
        h = x
        for i, model in enumerate(self.pts_linears):
            h = self.softplus(model(h))
            if i+1 in self.skip_connect:
                h = torch.cat([x,h], dim = -1) / np.sqrt(2)
        
        h = self.feature_linear(h)
        return h[..., :1], h[..., 1:]
                
    def gradient_for_loss(self, x):
        x.requires_grad_(True)
        d, _ = self.output(x)
        gradient = get_grad(x,d)
        return gradient

    def forward(self, x):
        x.requires_grad_(True)
        d, feature = self.output(x)
        bound = self.bound_scale * (self.r - torch.norm(x,p=2,dim=-1, keepdim=True))
        d = torch.minimum(d, bound)
        gradient = get_grad(x, d)
        return d, feature, gradient

        
class RadienceFieldNetwork(nn.Module):
    def __init__(self, input_dim, embed_length, feature_length, D = 4, W = 256, device='cuda'):
        super(RadienceFieldNetwork, self).__init__()
        self.output_dim = 3

        self.embedding = Embedding(input_dim, embed_length, device=device)

        self.input_dim = 3 + self.embedding.output_dim + 3 + feature_length
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_dim, W)]+[nn.Linear(W, W) for i in range(D-1)] + [nn.Linear(W, self.output_dim)]
        )

    def forward(self, points, view, normals, feature):
        view = self.embedding.embed(view)
        x = torch.cat([points, view, normals, feature], dim = -1)
        for i, model in enumerate(self.pts_linears):
            if i == 0:
                x = model(x)
            else:
                x = model(F.relu(x))
        
        return torch.sigmoid(x)

class VolSDF(nn.Module):
    def __init__(self, position_network, rendering_network, NeRF = None, r = 3, beta = 0.1, device='cuda'):
        super(VolSDF, self).__init__()

        self.position_network = position_network
        self.radience_field_network = rendering_network
        self.NeRF = NeRF

        self.beta = torch.Tensor([beta]).to(device)
        self.r = r

    def forward(self, x, view):
        beta = torch.abs(self.beta) + 1e-4
        sdf, feature, gradient = self.position_network(x)
        raw_color = self.radience_field_network(x, view, gradient, feature)
        density = cal_density(sdf, beta)
        # density = (0.5+0.5*torch.sign(d)*(1-torch.exp(-torch.abs(d)/beta))) * 1 / beta
        return density, raw_color

    def gradient(self, x):
        return self.position_network.gradient_for_loss(x)

    def get_sdf(self, x):
        d, _ = self.position_network.output(x)
        d = torch.minimum(d, self.r - torch.norm(x, dim = -1, p=2, keepdim=True))
        return d

    def density_from_sdf(self, sdf):  
        density = cal_density(sdf, torch.abs(self.beta) + 1e-4)
        return density

    def density(self, x):
        sdf, _ = self.position_network.output(x)
        sdf = torch.minimum(sdf, self.r - torch.norm(x, dim = -1, p=2, keepdim=True)) 
        density = self.density_from_sdf(d)
        return density

class NeRF(nn.Module):
    def __init__(self, D = 8, W = 256, input_ch = 3, input_ch_view = 3, output_ch = 4, skip_connect = [4], o_freq = 10, d_freq = 4, log_sampling = True, device='cuda'):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.device = device
        self.o_embedding = Embedding(input_ch, o_freq, log_sampling=log_sampling)
        self.d_embedding = Embedding(input_ch_view, d_freq, log_sampling=log_sampling)
        self.input_ch = self.o_embedding.output_dim
        self.input_ch_view = self.d_embedding.output_dim
        self.skip = skip_connect
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] + \
            [nn.Linear(self.W, self.W) if i not in skip_connect else nn.Linear(self.W+self.input_ch, W) for i in range(D-1)]
        )
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        # self.rgb_linears = nn.ModuleList(
        #     [nn.Linear(W+self.input_ch_view, W//2), nn.Linear(W//2, 3)]
        # )
        self.rgb_linear = nn.Linear(W//2,3)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view+W, W//2)])
        
    def forward(self, rays):  
        o, d = torch.split(rays, [3,3], dim = 1)
        o = self.o_embedding.embed(o)
        d = self.d_embedding.embed(d)
        h = o
        for i, model in enumerate(self.pts_linears):
            h = F.relu(model(h))
            if i in self.skip:
                h = torch.cat([o, h], -1)        
        alpha = self.alpha_linear(h)

        feature = self.feature_linear(h)
        h = torch.cat([feature, d], -1)
    
        for i, model in enumerate(self.views_linears):
            h = F.relu(model(h))

        rgb = self.rgb_linear(h)
        return rgb, alpha
      

