# Testbed-18: Task "Identifiers for Reproducible Science"

Use case: "Exploring Wilderness Using Explainable Machine Learning in Satellite Imagery"

Based on: https://gitlab.jsc.fz-juelich.de/kiste/asos

## Configuration

Set the following environment variables (see main_config.py):

* WORKING_DIR=
** Logging files, model checkpoints and figures will be saved here and will be loaded from here. If you have already a trained model, put the 'logs' folder into this working_dir.*


DATA_FOLDER=

SRC_DIR=

NUM_WORKERS=

DEVICE=

## Additional notes

* The asos folder needs to be in place to unpickle the model checkpoints.