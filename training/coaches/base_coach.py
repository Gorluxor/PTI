import abc
import os
import pickle
from argparse import Namespace
import wandb
import os.path
from criteria.localitly_regulizer import Space_Regulizer
import torch
from torchvision import transforms
from lpips import LPIPS
from scripts.latent_editor_wrapper import LatentEditorWrapper
from training.projectors import w_projector
from configs import global_config, paths_config, hyperparameters
from criteria import l2_loss
from models.e4e.psp import pSp
from utils.log_utils import log_image_from_w
from utils.models_utils import toogle_grad, load_old_G


from torchvision.transforms import Resize
from metric.psnr import psnr
class BaseCoach:
    def __init__(self, data_loader, use_wandb):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0

        if hyperparameters.first_inv_type == 'w+':
            self.initilize_e4e()

        self.e4e_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if global_config.use_edits_on_run:
            self.initilize_edits()
        if hyperparameters.pt_MS_GMSD_lambda > 0:
            from piq import MultiScaleGMSDLoss
            self.gmsd_loss = MultiScaleGMSDLoss()


    def initilize_edits(self):
        if hyperparameters.first_inv_type == 'w+':
            print('Using w+ space for StyleCLIP Edits')
            pass # TODO make StyleCLIP edits
        print('Using w space for inferGAN Edits')
        self.latent_editor = LatentEditorWrapper()
        #self.latents_after_edit = self.latent_editor.get_single_ganspace_edits

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G()
        toogle_grad(self.G, True)

        
            #torch.nn.init.xavier_uniform_(self.G.synthesis.b1024.weight, gain=torch.nn.init.calculate_gain('relu'))
            #nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            #if self.G.synthesis.b1024.bias is not None:
            #torch.nn.init.constant_(self.G.synthesis.b1024.bias, 0)
                #nn.init.constant_(m.bias, 0)

            # self.G.synthesis.b1024.eval().requires_grad_(False)
            # self.G.synthesis.b512.eval().requires_grad_(False)

        # if global_config.init_reinit_generator:
        #     ## Remove the original weights of the trainable network
        #     #for name, param in self.G.named_parameters():
        #     if False:
        #         print('a')
                # if any(layer in name for layer in global_config.init_layers): #name.contains('fc7'):
                #     layer_type = 'linear'
                    
                #     initialize_method = getattr(torch.nn, global_config.init_method)
                #     initialize_method(param, gain=torch.nn.init.calculate_gain('relu'))


        self.original_G = load_old_G()

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')  # TODO change number 0

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        if hyperparameters.first_inv_type == 'w+':
            w_potential_path = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}/0.pt'  # TODO, change number
        else:
            w_potential_path = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}/0.pt'
        if not os.path.isfile(w_potential_path):
            return None
        w = torch.load(w_potential_path).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image, image_name):

        if hyperparameters.first_inv_type == 'w+':
            w = self.get_e4e_inversion(image)

        else:
            id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
            if global_config.add_extra_layer:
                # downsamples the image to 1024 by 1024
                from torchvision.transforms import Resize
                transform_size = 1024 # TODO: variable?
                id_image = Resize((transform_size, transform_size))(id_image)
                #id_image = id_image.resize((transform_size, transform_size), PIL.Image.ANTIALIAS)

            w = w_projector.project(self.G, id_image, device=torch.device(global_config.device), w_avg_samples=600,
                                    num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                                    use_wandb=self.use_wandb)

        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer

    # @staticmethod
    # def downsample_2d(X, sz):
    #     """
    #     Downsamples a stack of square images.
        
    #     Args:
    #         X: a stack of images (batch, channels, ny, ny).
    #         sz: the desired size of images.
            
    #     Returns:
    #         The downsampled images, a tensor of shape (batch, channel, sz, sz)
    #     """
    #     import torch.nn.functional as F
    #     kernel = torch.tensor([[.25, .5, .25], 
    #                         [.5, 1, .5], 
    #                         [.25, .5, .25]], device=X.device).reshape(1, 1, 3, 3)
    #     kernel = kernel.repeat((X.shape[1], 1, 1, 1))
    #     while sz < X.shape[-1] / 2:
    #         # Downsample by a factor 2 with smoothing
    #         mask = torch.ones(1, *X.shape[1:])
    #         mask = F.conv2d(mask, kernel, groups=X.shape[1], stride=2, padding=1)
    #         X = F.conv2d(X, kernel, groups=X.shape[1], stride=2, padding=1)
            
    #         # Normalize the edges and corners.
    #         X = X = X / mask
        
    #     return F.interpolate(X, size=sz, mode='bilinear')
    

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        import torch.nn.functional as F
        loss = 0.0
        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu(), "lstep":global_config.local_step, "batch": global_config.img_index}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            if global_config.downsample_lpips: #True
                #target_images = F.interpolate(generated_images, size=(256, 256), mode='area')
                #resize_transform = Resize((256, 256), antialias = global_config.antialias)
                loss_lpips = self.lpips_loss(F.interpolate(generated_images, size=(256, 256), mode='area', antialias = global_config.antialias), 
                                            F.interpolate(real_images, size=(256, 256), mode='area', antialias = global_config.antialias)) # Downsample to 256 x 256
            else:
                loss_lpips = self.lpips_loss(generated_images, real_images) # keep original 1024 x 1024
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu(), "lstep":global_config.local_step, "batch":global_config.img_index}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda
             
        psnr_val = psnr(torch.clamp(generated_images.clone().detach().add_(1).div_(2), 0, 1), torch.clamp(real_images.clone().detach().add_(1).div_(2), 0, 1), reduction='none')
        if self.use_wandb:
            wandb.log({f'PSNR_val_{log_name}': psnr_val.detach().cpu(), "lstep":global_config.local_step, "batch":global_config.img_index}, step=global_config.training_step)
        if hyperparameters.pt_MS_GMSD_lambda > 0:
            
            gmsd_loss = self.gmsd_loss(torch.clamp(generated_images.clone().detach().add_(1).div_(2), 0, 1), torch.clamp(real_images.clone().detach().add_(1).div_(2), 0, 1))
            #self.gmsd_loss(generated_images, real_images)
            loss += gmsd_loss * hyperparameters.pt_MS_GMSD_lambda

            if self.use_wandb:
                wandb.log({f'GMSD_loss_val_{log_name}': gmsd_loss.detach().cpu(), "lstep":global_config.local_step, "batch":global_config.img_index}, step=global_config.training_step)

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def forward(self, w):
        generated_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)

        return generated_images

    def initilize_e4e(self):
        ckpt = torch.load(paths_config.e4e, map_location='cpu')
        opts = ckpt['opts']
        opts['batch_size'] = hyperparameters.train_batch_size
        opts['checkpoint_path'] = paths_config.e4e
        opts = Namespace(**opts)
        self.e4e_inversion_net = pSp(opts)
        self.e4e_inversion_net.eval()
        self.e4e_inversion_net = self.e4e_inversion_net.to(global_config.device)
        toogle_grad(self.e4e_inversion_net, False)

    def get_e4e_inversion(self, image):
        image = (image + 1) / 2
        new_image = self.e4e_image_transform(image[0]).to(global_config.device)
        _, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
                                      input_code=False)
        if self.use_wandb:
            log_image_from_w(w, self.G, 'First e4e inversion')
        return w
