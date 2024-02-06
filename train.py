#!/usr/bin/env python3
# encoding: utf-8

"""
Baseline ML models for cheminformatics
"""
from pprint import pprint

from baseline_models.main import BaselineML
from baseline_models.utils.parsing import parse_training_command_line_arguments
from baseline_models.utils.utils import create_logger, read_yaml_file


def main():
    """
    The main executable function.
    """

    args, config_dict = parse_training_command_line_arguments()

    logger = create_logger(args.log_name, args.save_dir)
    logger.info('Using arguments...')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')
    logger.info('')

    baseline_configs = config_dict['baseline_configs']
    baseline_configs['logger'] = logger
    baseline_configs['save_dir'] = args.save_dir

    # replace filepath with dictionary of featurizer settings
    baseline_configs['featurizer_settings'] = read_yaml_file(args.featurizer_yaml_path)
    logger.info('Featurizer settings:')
    logger.info(pprint(baseline_configs['featurizer_settings']))
    logger.info('')
    
    baseline_ML = BaselineML(**baseline_configs)
    baseline_ML.execute()


if __name__ == '__main__':
    main()
