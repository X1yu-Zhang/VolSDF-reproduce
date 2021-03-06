import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import load_dataset, RaysDataset, load_test_data
from tqdm import tqdm, trange
from sample import sampling_algorithm
from utils import * 
from plot import *
import os
import logging

def output2weight(t, density, white_bkgd, device):
    # t = torch.cat([t, torch.Tensor([1e10]).expand([t.shape[0], 1])], dim = -1)
    delta = t[..., 1:] - t[..., :-1]
    delta = torch.cat([delta, torch.tensor([1e10], device=device).expand([t.shape[0],1])], dim = -1)
    p = torch.exp(-delta * density) # 1 ... m-1
    T = torch.cat([torch.ones([p.shape[0], 1], device=device), torch.cumprod(p[..., :-1], dim = -1)], dim = -1) # 1 ... m
    # tau = torch.cat([(1 - p) * T[..., : -1] , T[..., -1, None]], dim = -1)
    weight = (1-p) * T

    return weight

def volume_rendering(rays_o, rays_d, model, device, **rendering_config):

    white_bkgd = rendering_config['white_bkgd']
    # with torch.no_grad():
    t, t_loss = sampling_algorithm(rays_o, rays_d, model, **rendering_config['sampling_config'])
    pts = get_sample_pts(rays_o, rays_d, t)
    pts_loss = get_sample_pts(rays_o, rays_d, t_loss)

    pts_shape = pts.shape
    rays_d = rays_d[:,None,:].expand(pts_shape).reshape([-1,3])
    pts = pts.reshape([-1,3])

    density, output, gradient = model(pts, rays_d)
    density = density.reshape(pts_shape[:-1])
    output = output.reshape(pts_shape)
    gradient = gradient.reshape(pts_shape)

    weight = output2weight(t, density, white_bkgd, device)

    rgb = torch.sum(weight[..., None]*output, dim=1)
    if white_bkgd:
        rgb = rgb+(1-torch.sum(weight, dim=-1)[..., None])
    
    if not rendering_config['render_only']:
        pts_near = pts_loss.reshape([-1,3])
        pts_far = torch.empty(rays_d.shape[0], 3).uniform_(-model.r, model.r).to(device)
        pts_loss = torch.cat([pts_near, pts_far], dim = 0)
        
        gradient = model.gradient(pts_loss)
    else:
        gradient = gradient.detach()
        gradient = gradient / gradient.norm(2, -1, keepdim=True)
        gradient = torch.sum(weight[..., None]*gradient, dim = 1)
    
    return rgb, gradient

def save_model(ckpt, model, step, optimizer, scan_id, name):
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)
    path = os.path.join(ckpt, "scan{}_{}.ckpt".format(scan_id, name))
    torch.save({
        'geometry_network': model.position_network.state_dict(), 
        'rendering_network': model.radience_field_network.state_dict(),
        'beta': model.beta,
        'step': step,
        'optimizer': optimizer.state_dict(),
        }, path)


def train(lr, lr_decay, N_iters, batch_size, l, i_save, ckpt, device,i_show_loss, **others):
    print("creating model...")
    optimizer, model, start = create_model(**others['model_config'])
    print("loading data ...")
    all_rays_rgb = load_dataset(**others['dataset_config'])
    print("loading finished")

    print("creating dataloader")
    train_dataset = RaysDataset(all_rays_rgb)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    print("done!")

    global_step = start
    scan_id = others['dataset_config']['scan_id']
    loss_avg = 0.
    rgb_avg = 0.
    eik_avg = 0.
    cnt_avg = 0.

    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./logs/training.log',
                    filemode='w')
    
    with tqdm(total=N_iters - start) as t:
        while global_step < N_iters:
            for idx, data in enumerate(train_loader):
                rays_o, rays_d, target = torch.split(data.to(device), [3,3,3], dim = 1)
                rgb, gradient = volume_rendering(rays_o, rays_d, model, **others['rendering_config'])

                rgb_loss = F.l1_loss(rgb, target)
                eik_loss = torch.mean((torch.norm(gradient, p=2, dim = -1) - 1)**2)

                loss = rgb_loss + l * eik_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logging.info("step: %-6d, rgb_loss: %9.5f, eik_loss: %9.5f, loss: %9.5f, beta: %9.5f"%(global_step, float(rgb_loss), float(eik_loss), float(loss), float(model.beta)))
                global_step += 1

                decay_rate = 0.1
                decay_steps = lr_decay * 1000
                new_lrate = lr * (decay_rate ** (global_step / decay_steps))

                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                if global_step % i_save == 0:
                    save_model(ckpt, model, global_step, optimizer, scan_id, global_step)

                rgb_avg += float(rgb_loss)
                eik_avg += float(eik_loss)
                loss_avg += float(loss)
                cnt_avg += 1.

                if global_step % i_show_loss == 0:
                    t.set_postfix({"rgb": rgb_avg / cnt_avg,"eik": eik_avg/cnt_avg, "loss": loss_avg / cnt_avg})
                    eik_avg = 0
                    rgb_avg = 0
                    loss_avg = 0
                    cnt_avg = 0
                t.update()
                if global_step > N_iters:
                    break
        
    save_model(ckpt, model, global_step, optimizer, 'final')

def test(batch_size, device, output, **config):
    _, model, _ = create_model(**config['model_config'])
    K, pose, img = load_test_data(**config['dataset_config'])
    H, W = img.shape[:-1]
    rays = get_rays_with_pose(H, W, K, pose)

    rgbs = []
    normals = []
    for i in trange(0, rays.shape[0], batch_size):
        data_slice = rays[i:i+batch_size]
        rays_o, rays_d = torch.split(torch.from_numpy(data_slice).float(), [3,3], dim = -1)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)

        rgb, normal = volume_rendering(rays_o, rays_d, model, **config['rendering_config'])

        rgbs.append(rgb.detach().cpu())
        normals.append(normal.detach().cpu())

    rgbs = torch.cat(rgbs, dim = 0).reshape([H, W, 3]).numpy()
    normals = torch.cat(normals, dim = 0).reshape([H, W, 3]).numpy()

    plot(output, model, rgbs=rgbs, normal_map=normals, **config['dataset_config'])

    loss = np.mean(np.linalg.norm(rgbs - img, ord=1, axis = -1))
    print("render_loss: ", loss)
    pass
def main():
    args = config()
    if args['render_only']: 
        test(**args)
    else:
        train(**args)
    pass

if __name__ == "__main__":
    main()
    pass
