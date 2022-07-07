import numpy as np
from PIL import Image
import wandb
from configs import global_config
import torch
import matplotlib.pyplot as plt
from typing import List

def log_image_from_w(w, G, name, text='current inversion', extra_text = ''):
    img = get_image_from_w(w, G)
    pillow_image = Image.fromarray(img)
    if extra_text != '':
        extra_text = extra_text + ' '
    wandb.log(
        {f"{extra_text}{name}": [
            wandb.Image(pillow_image, caption=f"{text} {name}")], "lstep":global_config.local_step,"batch": global_config.training_step},
        step=global_config.training_step)

def print_config(config, force_print = False):
    l = {}
    for kk in config.__dir__():
        ww = getattr(config, kk)
        if isinstance(ww, str) or isinstance(ww, int) or isinstance(ww, list) or isinstance(ww, float) or isinstance(ww, bool):
            if force_print:
                print(f"{kk}:{ww}")
            l[kk] = ww 
    return l

def log_image(img, name, use_pil = False):
    if not use_pil:
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().squeeze(0).cpu().numpy()
        pillow_img = Image.fromarray(img)
    else:
        pillow_img = img
    wandb.log({f"{name}": [
        wandb.Image(pillow_img, caption=f"{name} original, current; edits: rotation, smile and age")
    ],
    "lstep":global_config.local_step, 
    "batch": global_config.img_index},
    step=global_config.training_step)

def log_image_after_edit(image, w, G, latent_editor, name:str, skip_old_generator:bool=True, edit_range:List[int]=[-2, 2]):
    latents_after_edit = latent_editor.get_single_interface_gan_edits(w, edit_range)
    
    latents = []
    for direction, factor_and_edit in latents_after_edit.items():
        for latent in factor_and_edit.values():
            latents.append(latent)
            #log_image_from_w(latent, G, name, direction, direction)
    latents.insert(0, w)
    imgs = [get_image_from_w(lat, G) for lat in latents]        
    #import pdb; pdb.set_trace()
    imgs.insert(0, (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().squeeze(0).cpu().numpy())
    img = create_alongside_images(imgs)
    log_image(img, name, True)

def log_images_from_w(ws, G, names):
    for name, w in zip(names, ws):
        w = w.to(global_config.device)
        log_image_from_w(w, G, name)


def plot_image_from_w(w, G):
    img = get_image_from_w(w, G)
    pillow_image = Image.fromarray(img)
    plt.imshow(pillow_image)
    plt.show()


def plot_image(img):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    pillow_image = Image.fromarray(img[0])
    plt.imshow(pillow_image)
    plt.show()


def save_image(name, method_type, results_dir, image, run_id):
    image.save(f'{results_dir}/{method_type}_{name}_{run_id}.jpg')


def save_w(w, G, name, method_type, results_dir):
    im = get_image_from_w(w, G)
    im = Image.fromarray(im, mode='RGB')
    save_image(name, method_type, results_dir, im)


def save_concat_image(base_dir, image_latents, new_inv_image_latent, new_G,
                      old_G,
                      file_name,
                      extra_image=None):
    images_to_save = []
    if extra_image is not None:
        images_to_save.append(extra_image)
    for latent in image_latents:
        images_to_save.append(get_image_from_w(latent, old_G))
    images_to_save.append(get_image_from_w(new_inv_image_latent, new_G))
    result_image = create_alongside_images(images_to_save)
    result_image.save(f'{base_dir}/{file_name}.jpg')


def save_single_image(base_dir, image_latent, G, file_name):
    image_to_save = get_image_from_w(image_latent, G)
    image_to_save = Image.fromarray(image_to_save, mode='RGB')
    image_to_save.save(f'{base_dir}/{file_name}.jpg')


def create_alongside_images(images):
    res = np.concatenate([np.array(image) for image in images], axis=1)
    return Image.fromarray(res, mode='RGB')


def get_image_from_w(w, G):
    if len(w.size()) <= 2:
        w = w.unsqueeze(0)
    with torch.no_grad():
        img = G.synthesis(w, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    return img[0]


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    from matplotlib.lines import Line2D
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.clone().cpu().detach().abs().mean())
            max_grads.append(p.grad.clone().cpu().detach().abs().max())
    fig = plt.figure(figsize =(10, 7))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    #wandb.log({"grad_flow": fig}, step=global_config.training_step) #wandb.Image(figure=fig)})
    wandb.log({"grad_flow": wandb.Image(fig)}, step=global_config.training_step) #wandb.Image(figure=fig)})