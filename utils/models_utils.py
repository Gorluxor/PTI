import pickle
import functools
import torch
from configs import paths_config, global_config


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type):
    new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt'
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G():
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G


def initialize_weights(model):
    from torch import nn
    from torch.nn import init
    initialize_method = getattr(init, global_config.init_method)
    initialize_method_bias = getattr(init, global_config.init_bias_method)
    with torch.no_grad():
        for name, m in model.named_modules():
            #print(f"{name} passed 1")
            if not any(layer in name for layer in global_config.init_layers):
               print(f"{name} skipped reinitialization with {global_config.init_layers}")
               continue # not selected, change in global_config
            # else:
            #     print(f"{type(m)}:{name}")
            if isinstance(m, nn.Conv2d) or m._get_name() == 'SynthesisLayer':
                if global_config.init_reset_generator:
                        print(f"Using:{m.reset_parameters} method")
                        m.reset_parameters()
                else:    
                    print(f"{name} layer: reinitilized with {global_config.init_method}")
                    initialize_method(m.weight, gain=torch.nn.init.calculate_gain("relu"))
                    #nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        initialize_method_bias(m.bias, 0)
                        #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear) or m._get_name() == 'FullyConnectedLayer' or m._get_name() == 'ToRGBLayer':
                # if global_config.init_reset_generator:
                #     print(f"{name} layer: reset")
                #     m.reset_parameters()
                # else: 
                print(f"{name} layer: reinitilized with {global_config.init_method}")
                #nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                initialize_method(m.weight, 1)
                #initialize_method(m.weight, gain=torch.nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                print(f"{type(m)}:{name} not initialized")

# def toggle_model_activations(model):
#     from torch import nn
#     with torch.no_grad():
#         for name, m in model.named_modules():
#             #if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             m.eval()

def expand_model(model, also_return:bool = False):
    base_class = model.synthesis.b1024.__class__ # SynthesisBlock
    # TODO Parametize more?
    info_2048 = {'w_dim': 512,
        'resolution': 2048,
        'img_channels': 3,
        'is_last': False,
        'use_fp16': True,
        'conv_clamp': 256,
        'architecture': 'skip',
        'resample_filter': [1, 3, 3, 1],
        'use_noise': True,
        'activation': 'lrelu',
    }
    args = (32, 16) # in channels and out channels 
    b2048 = base_class(*args, **info_2048).cuda()
    model.synthesis.num_ws += b2048.num_conv
    setattr(model.synthesis, f'b2048', b2048)
    model.synthesis.block_resolutions.append(2048)
    return also_return
