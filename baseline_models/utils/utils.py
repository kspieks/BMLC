import logging
import os

import joblib
import yaml

def create_logger(name, log_dir):
    """
    Creates a logger with a stream handler and file handler.
    
    Args:
        name (str): The name of the logger.
        log_dir (str): The directory in which to save the logs.
    
    Returns:
        logger: the logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger


def read_yaml_file(path):
    """
    Read an input YAML file and return the parameters as python variables.

    Args:
        path (str): The input YAML file path to read.

    Returns: dict
        The content read from the file.
    """
    if not isinstance(path, str):
        raise ValueError(f'yaml file path must be a string, got {path} which is a {type(path)}')
    
    if not os.path.isfile(path):
        raise ValueError(f'Could not find the YAML file {path}')
    
    with open(path, 'r') as f:
        content = yaml.safe_load(f)

    return content


def save_models(models, path):
    """
    Save a list of fitted models with joblib, compressed with zlib level 3.

    Compression shrinks the file because the models hold large numpy arrays
    (e.g., 5 random forests: 96.4 MiB uncompressed, 28.4 MiB compressed).
    Compression level 3 is joblib's suggested trade-off between file size and
    speed.

    Args:
        models (list): The fitted models, one per split.
        path (str): The file path to write, e.g. 'RF_MACCS_best_models.pkl'.
    """
    joblib.dump(models, path, compress=('zlib', 3))


def load_models(path):
    """
    Load a list of fitted models from a file written by `save_models` or by
    `pickle.dump`.

    joblib.load reads the first bytes of the file to detect whether it is
    compressed (zlib, gzip, bz2, xz/lzma) or not compressed, and handles each
    case. Older uncompressed .pkl files therefore load without any change,
    e.g., examples/RDB7/training_results/XGB_MACCS_best_models.pkl.
    The reverse does not hold: pickle.load cannot read a compressed file and
    raises an UnpicklingError.

    Like pickle, joblib can execute arbitrary code when loading a file, so only
    load files from a trusted source. Loading also requires the same package
    versions (scikit-learn, xgboost, numpy) as the environment that saved them.

    Args:
        path (str): The file path to read.

    Returns:
        list: The fitted models, one per split.
    """
    return joblib.load(path)
