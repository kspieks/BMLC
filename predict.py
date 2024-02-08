#!/usr/bin/env python3
# encoding: utf-8

"""
Script to make predictions using a pretrained model (or ensemble of trained models).
"""

import os
import pickle as pkl
import yaml

import numpy as np
import pandas as pd
from pandarallel import pandarallel

from baseline_models.features.featurizers import _FP_FEATURIZERS
from baseline_models.features.utils import get_rxn_fp
from baseline_models.utils.parsing import parse_prediction_command_line_arguments
from baseline_models.utils.utils import create_logger, read_yaml_file


def main():
    """
    The main executable function.
    """

    args = parse_prediction_command_line_arguments()

    logger = create_logger(args.log_name, args.save_dir)
    logger.info('Using arguments...')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')
    logger.info('')

    # replace filepath with dictionary of featurizer settings
    featurizer_settings = read_yaml_file(args.featurizer_yaml_path)
    logger.info('Featurizer settings:')
    logger.info(yaml.dump(featurizer_settings, default_flow_style=False))
    logger.info('')

    # read in data
    df = pd.read_csv(args.data_path)
    logger.info(f'{args.data_path} has {len(df):,} rows')

    # generate features in parallel
    pandarallel.initialize(nb_workers=args.n_cpus_featurize, progress_bar=True)
    fp_arrays = []
    for featurizer, parameter_dict in featurizer_settings.items():
        params = parameter_dict['parameters']
        logger.info(f'Calculating {featurizer} features...')

        featurizers = featurizer.split('+')
        if len(featurizers) != len(parameter_dict['parameters']):
            msg = "Dimension mismatch!\n"
            msg += f"There are {len(featurizers)} featurizers specified and "
            msg += f"{len(parameter_dict['parameters'])} parameter dictionaries specified in the input json file.\n"
            msg += "These values should be identical."
            raise ValueError(msg)

        fp_arrays = []
        for f, params in zip(featurizers, parameter_dict['parameters']):
            if len(featurizers) > 1:
                # if there are only one set of features to calculate, don't print the same line twice
                logger.info(f'Calculating {f} features...')
            logger.info(f'Specified settings include\n{params}')

            # reaction mode
            if args.rxn_mode:
                params['featurizer'] = _FP_FEATURIZERS[f]
                fps = df[args.smiles_column].parallel_apply(get_rxn_fp, **params)

            # molecule mode
            else:
                fps = df[args.smiles_column].parallel_apply(_FP_FEATURIZERS[f], **params)
        
            fp_array = np.stack(fps.values)
            logger.info(f'Fingerprint array has shape {fp_array.shape}\n')
            fp_arrays.append(fp_array)

    X = np.hstack(fp_arrays)
    logger.info(f'X.shape: {X.shape}')

    # load models and scalers
    with open(args.model_path, 'rb') as f:
        models = pkl.load(f)
    with open(args.scaler_path, 'rb') as f:
        scalers = pkl.load(f)
    
    if len(models) != len(scalers):
        msg = "Number of models does not equal number of scalers!\n"
        msg += f"Found {len(models)} models and {len(scalers)} scalers.\n"
        msg += "These values should be identical..."
        raise ValueError(msg)
    
    y_preds = np.zeros((len(X), len(scalers)))
    for i, (model, scaler) in enumerate(zip(models, scalers)):
        y_pred = scaler.inverse_transform(model.predict(X).reshape(-1, 1)).flatten()
        y_preds[:, i] = y_pred

    if len(models) > 1:
        df_preds = pd.DataFrame(y_preds, columns=[f'model{i}_pred' for i in range(len(models))])
        df_preds.insert(0, 'mean_pred', y_preds.mean(axis=1))
        df_preds.insert(1, 'ensemble_std', np.sqrt(y_preds.var(axis=1)))
    else:
        df_preds = pd.DataFrame(y_preds, columns=['pred'])
    
    # add smiles
    df_preds.insert(0, 'smiles', df[args.smiles_column].values)
    
    # save predictions
    df_preds.to_csv(os.path.join(args.save_dir, args.pred_name), index=False)

if __name__ == '__main__':
    main()
