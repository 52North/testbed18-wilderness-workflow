import os
from torch.cuda import is_available

# ------------------------------------------------------------------------------------------------------------
# Please check the following configurations:

# working directory:
# Logging files, model checkpoints and figures will be saved here and will be loaded from here.
# If you have already a trained model, put the 'logs' folder into this working_dir.
working_dir = os.environ.get('WORKING_DIR', '/home/martin/Projekte/OGC_Testbed-18/software/asos_work_dir')

# anthroprotect data folder:
# Define the path to the anthroprotect dataset folder.
data_folder = os.environ.get('DATA_FOLDER', '/home/martin/Projekte/OGC_Testbed-18/data/anthroprotect')

# server url
# url to external server (API Coverages, ...)
server_url = os.environ.get('SERVER_URL', 'https://18.testbed.dev.52north.org/geodatacube/')

# number of workers:
# Define the number of workers to load data while training the model and running model predictions.
num_workers = int(os.environ.get('NUM_WORKERS', 8))

# device:
# Define which device is available and should be used
device = os.environ.get('DEVICE', 'cpu')  # 'cuda', 'cuda:<cuda_id>' or 'cpu'
if device.startswith('cuda') and not is_available():
    raise Exception("'cuda' is selected as device but not available on your machine")

# You might want to run the following lines on your system to avoid abortion while training the model.
# Read more at: https://pytorch.org/docs/stable/multiprocessing.html#file-descriptor-file-descriptor
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# ------------------------------------------------------------------------------------------------------------


# The following configurations must be only changed in specific cases.

# If you changed the structure within the AnthroProtect data folder, you should change the following variables:
anthroprotect_data_folder_tiles = os.path.join(data_folder, 'tiles/s2')
anthroprotect_data_folder_investigative = os.path.join(data_folder, 'investigative')
anthroprotect_file_infos_path_raw = os.path.join(data_folder, 'infos.csv')

# If you download the AnthroProtect dataset by yourself, you need to specify the location of the raw data at some point:
anthroprotect_data_folder_raw = '/media/timo/My Book/data/anthroprotect/raw'

# If you want to change the name of the 'logs' folder in which logging files etc. are stored, you need to change this path:
log_path = os.path.join(working_dir, 'logs')
