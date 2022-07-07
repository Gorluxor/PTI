import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w, log_image, log_image_after_edit, print_config, plot_grad_flow

class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def reinit(self):
        from utils.models_utils import initialize_weights
        if global_config.init_reinit_generator or global_config.init_reset_generator:
            print("Whole generator is being reinitilized")
            initialize_weights(self.G)

        if global_config.zero_out_generator:
            print("B1024 generator is being reinitilized")
            initialize_weights(self.G.synthesis.b1024)

    # Freeze layers in global_config.freeze_layers list
    def freeze_layers(self, model):
      for name, m in model.named_parameters():
            if any(layer in name for layer in global_config.freeze_layers):
               print(f"Freezing layer {name}")
               m.requires_grad = False

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True
        global_config.img_index = 0
        if global_config.downsample_lpips:
            print("WARNING: Peihao change - using downsampled lpips")
        if self.use_wandb:
            # set wandb to watch for gradients of the self.G.synthesis model
            import wandb
            print('Wandb is being used to visualize gradient')
            wandb.watch(self.G, log='gradients', log_freq=100, idx=None)
        for fname, image in tqdm(self.data_loader):
            global_config.local_step = 0
            image_name = fname[0]

            #if self.use_wandb:
                #import pdb; pdb.set_trace()
               # log_image(image, image_name)
            #import wandb
            #global_config.run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=f"{global_config.run_name}_{image_name}", group=f"{global_config.run_name}")
            #wandb.config.update(print_config(hyperparameters)) # Added for more info in wandb
            #wandb.config.update(print_config(global_config))

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(w_path_dir, image_name)

            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot = self.calc_inversions(image, image_name)
            self.reinit()
            
            if global_config.add_extra_layer:
                from utils.models_utils import expand_model
                print("WARNING: Added extra layer, use 2k preprocessed images")
                expand_model(self.G)
            
            if len(global_config.freeze_layers) > 0: # if we have configured freeze layers, we need to freeze them
                self.freeze_layers(self.G)
            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            
            if global_config.add_extra_layer:
                # Have to add 2 extra latent codes because of the increase of the size
                # this is accomplished for now by adding a new layer to the generator and copying the latent code
                # from the previous ones (for w+ and all are the same for w optimization)
                w_pivot = torch.cat([w_pivot.clone().detach(), w_pivot.clone().detach()[:,1:3,:]], dim=1).to(global_config.device)
            else:
                w_pivot = w_pivot.to(global_config.device)

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            global_config.img_index += 1
            for i in tqdm(range(hyperparameters.max_pti_steps)):

                



                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                # if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                #     break

                loss.backward()
                #plot_grad_flow(self.G.synthesis.named_parameters())
                
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    
                    if global_config.use_edits_on_run and self.latent_editor is not None:
                        log_image_after_edit(image, w_pivot, self.G, self.latent_editor, image_name)
                    else:
                        log_images_from_w([w_pivot], self.G, [image_name]) # Original log
                global_config.training_step += 1
                log_images_counter += 1
                global_config.local_step += 1

            self.image_counter += 1

            torch.save(self.G,
                       f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
