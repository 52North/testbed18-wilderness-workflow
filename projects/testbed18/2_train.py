import os

import torch.nn as nn

from projects.asos import config, modules, utils
from tlib import ttorch, tlearn
import projects.utils


# test run
test_run = False


# model kwargs

if config.dataset in ['anthroprotect', 'places']:

    model_kwargs = {
        'in_channels': config.in_channels,
        'n_unet_maps': 3,
        'n_classes': 1,

        'unet_base_channels': 32,  # standard UNet has 64
        'double_conv': False,  # standard UNet has True, we use False
        'batch_norm': True,  # standard UNet has False, we use True
        'unet_mode': 'bilinear',  # standard UNet has None, we use 'bilinear'
        'unet_activation': nn.Tanh(),

        'final_activation': nn.Sigmoid(),  # nn.Sigmoid() or nn.Softmax(dim=1)
    }

# trainer params

if config.dataset in ['anthroprotect', 'places']:

    criterion = nn.MSELoss()  # e.g. nn.MSELoss() or nn.BCEWithLogitsLoss()
    optimizer = None  # defined by lr etc.

    lr = 1e-2
    weight_decay = 1e-4
    epochs = 5 if config.dataset == 'anthroprotect' else 20

if test_run:
    epochs = 1


def run_training(lr=lr, weight_decay=weight_decay, epochs=epochs, trainer=None):

    if trainer is None:  # create trainer

        model = modules.Model(**model_kwargs)
        datamodule = utils.get_new_datamodule()

        trainer = ttorch.train.ClassTrainer(
            model=model, datamodule=datamodule, log_dir=config.working_dir, criterion=criterion, optimizer=optimizer,
            lr=lr, weight_decay=weight_decay, one_cycle_lr_epochs=epochs, test_run=test_run,)

    # save file infos before training (and later also after training) in log_dir
    fi = utils.load_file_infos(raw=True)
    fi.save(os.path.join(trainer.log_dir, 'file_infos.csv'))
    
    # run training
    trainer(epochs)
    #if not test_run:
    evaluate(trainer)

    projects.utils.print_log_path_warning()

    return trainer


def evaluate(trainer):

    # run trainer evalation
    trainer.evaluate()

    # store predictions to file_infos.csv
    print('\nstore predictions:')
    fi = utils.load_file_infos(raw=True)
    dataset = trainer.datamodule.get_dataset(files=fi.df.index, labels=fi.df.label, prepend_folder=True)
        # dataset is not loaded with trainer because trainer might have changed due to training
    preds = trainer.predict_dataset(dataset)

    fi.df['pred'] = tlearn.utils.preds_to_pred_labels(preds)
    fi.df['score'] = tlearn.utils.preds_to_pred_scores(preds)
    fi.df['correct'] = fi.df['label'] == fi.df['pred']
    fi.save(os.path.join(trainer.log_dir, 'file_infos.csv'))

    # plot false predicted samples on map
    if 'lon' in fi.df.columns:  # if coordinates (lon and lat) are given
        datasets = ['train', 'val', 'test']
        fi.plot_column(
            column='correct',
            df=fi.df[fi.df['datasplit'].isin(datasets)],
            output_dir=os.path.join(trainer.log_dir, 'map_correct_preds.html'),
        )


def tune_hyperparams(
    lrs=[1e-2, 1e-3],
    weight_decays=[0, 1e-4, 1e-3, 1e-2, 1e-1],
    max_epochss=[3, 5, 10],
):
    """
    Runs training with all possible combinations of given parameters.

    :param lrs: list of learning rates
    :param weight_decays: list of weight decays
    :param max_epochss: list of maximum number of epochs
    :return:
    """

    n_runs = len(lrs) * len(weight_decays) * len(max_epochss)
    count = 0
    for lr in lrs:
        for weight_decay in weight_decays:
            for max_epochs in max_epochss:
                count += 1
                print(f'\n\n-----------------\n Run {count} out of {n_runs}.\n-----------------\n\n')

                run_training(lr=lr, weight_decay=weight_decay, max_epochs=max_epochs)


if __name__ == '__main__':

    # choose one of the following:

    # train once with standard parameters
    run_training()

    # just evaluate
    #evaluate(trainer=utils.load_trainer())

    # continue training
    #trainer = utils.load_trainer()
    #left_epochs = epochs - trainer._current_epoch
    #run_training(trainer=trainer, epochs=left_epochs)

    # train multiple times (hyperparameter tuning)
    #tune_hyperparams()
