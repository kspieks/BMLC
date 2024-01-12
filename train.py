#!/usr/bin/env python3
# encoding: utf-8

"""
Baseline ML models for cheminformatics
"""
from baseline_models.main import BaselineML
from utils.parsing import parse_training_command_line_arguments
from utils.utils import create_logger


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
    baseline_ML = BaselineML(**baseline_configs)
    baseline_ML.execute()


if __name__ == '__main__':
    main()
