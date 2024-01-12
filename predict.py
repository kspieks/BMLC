#!/usr/bin/env python3
# encoding: utf-8

"""
Script to make predictions using a pretrained model (or ensemble of trained models).
"""

import os
import pickle as pkl

import numpy as np
import pandas as pd

from baseline_models.featurizers import _FP_FEATURIZERS, get_rxn_fp
from utils.parsing import parse_prediction_command_line_arguments
from utils.utils import create_logger


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

    # read in data
    df = pd.read_csv(args.data_path)
    logger.info(f'{args.data_path} has {len(df):,} rows')

    # generate features
    featurizer = args.featurizer
    # reaction mode
    if args.rxn_mode:
        df[featurizer] = df.smiles.apply(get_rxn_fp, featurizer=_FP_FEATURIZERS[featurizer])

    # molecule mode
    else:
        df[featurizer] = df.smiles.apply(_FP_FEATURIZERS[featurizer])
    
    X = np.stack(df[featurizer].values)
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
    df_preds.insert(0, 'smiles', df.smiles.values)
    
    # save predictions
    df_preds.to_csv(os.path.join(args.save_dir, args.pred_name), index=False)

if __name__ == '__main__':
    main()
