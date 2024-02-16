import logging
import os
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
