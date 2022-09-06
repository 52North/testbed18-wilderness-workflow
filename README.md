# Testbed-18: Task "Identifiers for Reproducible Science" - wilderness workflow

Use case: "Exploring Wilderness Using Explainable Machine Learning in Satellite Imagery"

Based on: https://gitlab.jsc.fz-juelich.de/kiste/asos

## Installation/prerequisites

Install `tlib` (note: do not install via pip because there is different library with this name):
```
pip install git+https://gitlab.jsc.fz-juelich.de/kiste/asos@main
```

## Configuration

Set the following environment variables (see main_config.py):

* WORKING_DIR=
  * *Logging files, model checkpoints and figures will be saved here and will be loaded from here. If you have already a trained model, put the 'logs' folder into this working_dir.*
  * Default:
* DATA_FOLDER=
  *  *Define the path to the anthroprotect dataset folder.*
  * Default:
* SRC_DIR=
  * *Define the base directory where your source code is located.*
  * Default:
* NUM_WORKERS=
  * *Define the number of workers to load data while training the model and running model predictions.*
  * Default: 8
* DEVICE=
  * *Define which device is available and should be used ('cuda', 'cuda:<cuda_id>' or 'cpu').*
  * Default: cpu
* SERVER_URL=
  * *Define the url of a server providing data via API Coverages*
  * Default: https://18.testbed.dev.52north.org/geodatacube/

## Additional notes

* The asos folder (with only \_\_init\_\_.py and modules.py) needs to be in place to unpickle the model checkpoints.
