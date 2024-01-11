import os
from argparse import ArgumentParser


def parse_training_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """
    parser = ArgumentParser(description='Baseline models for cheminformatics')

    parser.add_argument('--log_dir', type=str,
                        help='Directory to store the log file.')
    parser.add_argument('--log_name', type=str, default='train',
                        help='Filename for the training log.')

    
    baseline_configs = parser.add_argument_group('baseline_configs')
    baseline_configs.add_argument('--data_path', type=str,
                        help='Path to the csv file containing SMILES and prediction target for regression.')
    baseline_configs.add_argument('--target', type=str,
                        help='Name of column to use as regression target.')
    baseline_configs.add_argument('--split_path', type=str,
                        help='Path to .pkl file with a list of train, val, and test indices.')
    baseline_configs.add_argument('--rxn_mode', action='store_true', default=False,
                        help='Boolean indicating whether the smiles column contains reaction SMILES \
                        whose features will be concatenated as r + (p-r).')
    
    baseline_configs.add_argument('--featurizers', nargs='+',
                        choices=['morgan_counts', 'morgan_binary',
                                 'rdkit_2d', 'rdkit_2d_normalized',
                                 'AtomPair', 'Avalon', 'MACCS', 'MQN'],
                        help='Fingerprint featurizers to use.')

    baseline_configs.add_argument('--models', nargs='+',
                        choices=['LinearSVR', 'RF', 'XGB'],
                        help='Sklearn models to train.')
    
    baseline_configs.add_argument('--random_state', type=int,
                                  default=42,
                                  help='Random state used to initialize both sklearn models and Optuna.')

    # Optuna arguments
    baseline_configs.add_argument('--n_jobs', type=int, default=4,
                        help='Number of CPUs to use in parallel for exploring hyperparameters.')
    baseline_configs.add_argument('--n_trials', type=int, default=32,
                        help='Number of trials to do with Optuna.')

    args = parser.parse_args(command_line_args)

    if args.log_dir is None:
        args.log_dir = os.getcwd()

    config_dict = dict({})
    group_list = ['baseline_configs']
    for group in parser._action_groups:
        if group.title in group_list:
            config_dict[group.title] = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}

    return args, config_dict
