import warnings

from tlib import tlearn
from tjects import main_config


# You might want to change the parameters in the following box:

################################################################## below
# define dataset
dataset = 'anthroprotect'  # anthroprotect

# parameters
batch_size = 32
max_image_size = 2048
num_workers = main_config.num_workers
device = main_config.device

random_seed = 0
channels = None  # None means take all channels
cutmix = 0.8
################################################################## above


# data type
data_type = 'sentinel_2'


# define number of in_channels
if channels is not None:
    in_channels = len(channels)
elif data_type == 'sentinel_2':
    in_channels = 10


# define normalization and clip range
if data_type == 'sentinel_2':
    normalization = (0, 10000)

clip_range = (0, 1)


# set random seed
tlearn.utils.set_random_seed(random_seed=0)


# folders
working_dir = main_config.working_dir
data_folder = main_config.data_folder
log_path = main_config.log_path  # the folder 'version_x' in 'ttorch_logs' must be renamed and moved according to given log_path

data_folder_raw = main_config.anthroprotect_data_folder_raw
data_folder_tiles = main_config.anthroprotect_data_folder_tiles
data_folder_investigative = main_config.anthroprotect_data_folder_investigative
file_infos_path_raw = main_config.anthroprotect_file_infos_path_raw

# server
server_url = main_config.server_url
