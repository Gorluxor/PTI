## Device
cuda_visible_devices = '0'
device = 'cuda:0'

## Logs
training_step = 1
image_rec_result_log_snapshot = 100
pivotal_training_steps = 0
model_snapshot_interval = 400

## Run name to be updated during PTI
run_name = ''

# Lpips
downsample_lpips = False
antialias = False

# Randomize old generator

# Pre processing use of other methods
lanczos = True # Used in preprocessing 

#from torch.nn import init
init_reset_generator = False
init_reinit_generator = False
init_method = "xavier_uniform_"
init_bias_method = "constant_"
init_layers = ['b2048']
freeze_layers = []

use_edits_on_run = False
img_index = 0

zero_out_generator = False

add_extra_layer = True

plot_grad = False