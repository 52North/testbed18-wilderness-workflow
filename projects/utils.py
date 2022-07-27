import warnings

from projects import main_config


def print_log_path_warning():
    warnings.warn(
        f'\n\n!!!'
        f'Folder \'version_x\' in \'ttorch_logs\' must be moved to working directory: {main_config.working_dir}\n'
        f'and be renamed to \'{main_config.log_path.split("/")[-1]}\' to continue with the workflow.\n'
    )
