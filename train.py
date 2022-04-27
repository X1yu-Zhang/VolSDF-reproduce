import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import load_dataset, RaysDataset, load_test_data
from tqdm import tqdm, trange
from sample import sampling_algorithm
from utils import * 
import os
import imageio
from torch.utils.tensorboard import SummaryWriter

def output2rgb(t, density, output, white_bkgd, device):
    # t = torch.cat([t, torch.Tensor([1e10]).expand([t.shape[0], 1])], dim = -1)
    delta = t[..., 1:] - t[..., :-1]
    p = -delta * density[...,:-1] # 1 ... m-1
    T = torch.cat([torch.zeros([p.shape[0], 1], device=device), torch.cumprod(p, dim = -1)], dim = -1) # 1 ... m
    tau = torch.cat([(1 - p) * T[..., : -1] , T[..., -1, None]], dim = -1)
    rgb = torch.sum(tau[..., None] * output, dim = 1)

    if white_bkgd:
        rgb = rgb + (1-torch.sum(tau, dim = -1))[..., None]
    return rgb

def volume_rendering(rays_o, rays_d, model, device, **rendering_config):

    white_bkgd = rendering_config['white_bkgd']
    with torch.no_grad():
        t, t_loss = sampling_algorithm(rays_o, rays_d, model, **rendering_config['sampling_config'])
    pts = get_sample_pts(rays_o, rays_d, t)
    pts_loss = get_sample_pts(rays_o, rays_d, t_loss)

    pts_shape = pts.shape
    rays_d = rays_d[:,None,:].expand(pts_shape).reshape([-1,3])
    pts = pts.reshape([-1,3])

    density, output = model(pts, rays_d)
    density = density.reshape(pts_shape[:-1])
    output = output.reshape(pts_shape)

    rgb = output2rgb(t, density, output, white_bkgd, device)
    gradient = None

    if not rendering_config['render_only']:
        pts_near = pts_loss.reshape([-1,3])
        pts_far = torch.empty(rays_d.shape[0], 3).uniform_(-model.r, model.r).to(device)
        pts_loss = torch.cat([pts_near, pts_far], dim = 0)
        
        gradient = model.gradient(pts_loss)
    
    return rgb, gradient

def save_model(ckpt, model, step, optimizer, name):
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)
    path = os.path.join(ckpt, "{}.ckpt".format(name))
    torch.save({
        'geometry_network': model.position_network.state_dict(), 
        'rendering_network': model.radience_field_network.state_dict(),
        'beta': model.beta,
        'step': step,
        'optimizer': optimizer.state_dict(),
        }, path)

def save_img(output, datatype, scan_id, img, **config):
    if not os.path.exists(output):
        os.mkdir(output)
    path = os.path.join(output, datatype+"_"+"scan{}".format(scan_id)+'.png')
    imageio.imwrite(path, img)
    return 
    
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
    loss_avg = 0.
    cnt_avg = 0.
    writer = SummaryWriter()
    with tqdm(total=N_iters - start, postfix={"loss": 0}) as t:
        while global_step < N_iters:
            for idx, data in enumerate(train_loader):
                rays_o, rays_d, target = torch.split(data.to(device), [3,3,3], dim = 1)
                rgb, gradient = volume_rendering(rays_o, rays_d, model, **others['rendering_config'])

                loss = F.l1_loss(rgb, target) + l * (F.mse_loss(gradient, torch.zeros_like(gradient))-1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1

                decay_rate = 0.1
                decay_steps = lr_decay * 1000
                new_lrate = lr * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                if global_step % i_save == 0:
                    save_model(ckpt, model, global_step, optimizer, global_step)

                loss_avg += float(loss)
                cnt_avg += 1.

                if global_step % i_show_loss == 0:
                    writer.add_scalar('Loss', loss_avg / cnt_avg, global_step)
                    t.set_postfix({"loss": loss_avg / cnt_avg})
                    loss_avg = 0
                    cnt_avg = 0
                t.update()
                if global_step > N_iters:
                    break
        
    save_model(ckpt, model, global_step, optimizer, 'final')
    writer.close()

def test(batch_size, device, output, **config):
    _, model, _ = create_model(**config['model_config'])
    K, pose, img = load_test_data(**config['dataset_config'])
    H, W = img.shape[:-1]
    rays = get_rays_with_pose(H, W, K, pose)

    rgbs = []
    for i in trange(0, rays.shape[0], batch_size):
        data_slice = rays[i:i+batch_size]
        rays_o, rays_d = torch.split(torch.from_numpy(data_slice).float(), [3,3], dim = -1)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)

        rgb, _ = volume_rendering(rays_o, rays_d, model, **config['rendering_config'])

        rgbs.append(rgb.detach().cpu())

    rgbs = torch.cat(rgbs, dim = 0).reshape([H, W, 3]).numpy()
    img_render = to8b(rgbs)

    save_img(output, **config['dataset_config'], img=img_render)
    
    loss = np.linalg.norm(rgbs - img, ord=1, axis = -1)
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