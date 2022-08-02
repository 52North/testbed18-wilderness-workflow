# Testbed-18: Task "Identifiers for Reproducible Science" - wilderness workflow

Use case: "Exploring Wilderness Using Explainable Machine Learning in Satellite Imagery"

Based on: https://gitlab.jsc.fz-juelich.de/kiste/asos

## Configuration

Set the following environment variables (see main_config.py):

* WORKING_DIR=
  * *Logging files, model checkpoints and figures will be saved here and will be loaded from here. If you have already a trained model, put the 'logs' folder into this working_dir.*
* DATA_FOLDER=
  *  *Define the path to the anthroprotect dataset folder.*
* SRC_DIR=
  * *Define the base directory where your source code is located.*
* NUM_WORKERS=
  * *Define the number of workers to load data while training the model and running model predictions.*
* DEVICE=
  * *Define which device is available and should be used ('cuda', 'cuda:<cuda_id>' or 'cpu').*

## Additional notes

* The asos folder (with only \_\_init\_\_.py and modules.py) needs to be in place to unpickle the model checkpoints.
