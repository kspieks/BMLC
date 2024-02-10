import os
import pickle as pkl
import time

import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import yaml

from .features.create_features import create_features
from .training import Objective, callback, naive_baseline, train_model


class BaselineML(object):
    """
    Class to train several baseline models.
    """

    def __init__(
        self,
        data_path,
        smiles_column,
        target,
        split_path,
        featurizer_settings,
        models,
        rxn_mode,
        n_cpus_featurize,
        n_cpus_optuna,
        n_trials,
        logger,
        save_dir,
        random_state=42,
    ):
        
        self.data_path = data_path
        self.smiles_column = smiles_column
        self.target = target
        self.split_path = split_path

        self.featurizer_settings = featurizer_settings
        self.models = models
        self.rxn_mode = rxn_mode

        self.n_cpus_featurize = n_cpus_featurize
        self.n_cpus_optuna = n_cpus_optuna
        self.n_trials = n_trials

        self.logger = logger
        self.save_dir = save_dir

        self.random_state = random_state

        # read in data
        self.df = pd.read_csv(self.data_path)

        # read in splits
        with open(self.split_path, "rb") as f:
            self.splits = pkl.load(f)

    def train_naive_baseline(self, y):
        # use naive baseline model that predicts the mean value from the training set for all test molecules
        self.logger.info("*" * 88)
        self.logger.info("Naive baseline: mean predictor")
        df_summary = naive_baseline(y, self.splits, self.logger)
        df_summary.to_csv(os.path.join(self.save_dir, f'naive_baseline_summary.csv'), index=False)

    def execute(self):
        # assign regression target y
        y = self.df[self.target].values

        # train the naive baseline model
        self.train_naive_baseline(y)

        # calculate input features X (i.e., fingerprint vectors)
        for featurizer, parameter_dict in self.featurizer_settings.items():
            self.logger.info("*" * 88)
            self.logger.info(f"Calculating {featurizer} features...")
            self.logger.info(f"Specified settings include\n{parameter_dict['parameters']}")

            X = create_features(df=self.df,
                                smiles_column=self.smiles_column,
                                rxn_mode=self.rxn_mode,
                                featurizer=featurizer,
                                parameters=parameter_dict['parameters'],
                                n_cpus=self.n_cpus_featurize
                                )
            
            # ensure the dimensions match before proceeding
            self.logger.info(f'X.shape is: {X.shape}')
            if X.shape[0] != len(y):
                msg = "Dimension mismatch!\n"
                msg += f"There are {X.shape[0]} SMILES and {len(y)} target values.\n"
                msg += "These values should be identical..."
                raise ValueError(msg)
            
            # save the featurization settings to be used during prediction
            save_file_name = parameter_dict['save_file_name']
            with open(f"{save_file_name}_settings.yaml", 'w') as f:
                yaml_string = yaml.dump({featurizer: 
                                         {'parameters': parameter_dict['parameters']}
                                         })
                f.write(yaml_string)

            for model_type in self.models:
                self.logger.info("*" * 44)
                self.logger.info(f"Training {model_type} model with {featurizer} features")
                self.logger.info(f"X.shape: {X.shape}")
                self.logger.info(f"y.shape: {y.shape}")

                start = time.time()
                # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
                study = optuna.create_study(
                    sampler=TPESampler(seed=self.random_state), direction="minimize"
                )
                study.optimize(
                    Objective(
                        X=X,
                        y=y,
                        splits=self.splits,
                        model_type=model_type,
                        random_state=self.random_state,
                    ),
                    n_trials=self.n_trials,
                    n_jobs=self.n_cpus_optuna,
                    callbacks=[callback],
                )
                self.logger.info(f"Elapsed time: {time.time() - start:.2f}s")

                # NaN value for columns/parameters which do not apply to that algorithm
                # sort to put best validation performance on top
                study.trials_dataframe().sort_values(by="value").to_csv(
                    os.path.join(self.save_dir, f"{model_type}_{save_file_name}_optuna_results.csv"), 
                    index=False
                )
                self.logger.info(study.trials_dataframe().sort_values(by="value"))

                self.logger.info('')
                self.logger.info(f"The best trial is : \n{study.best_trial}")
                self.logger.info(f"The best parameters are : \n{study.best_params}")

                best_model = study.user_attrs["best_model"]

                # refit the best model
                df_summary, df_predictions, models, y_scalers = train_model(
                    best_model,
                    X=X,
                    y=y,
                    splits=self.splits,
                )
              
                with open(os.path.join(self.save_dir, f"{model_type}_{save_file_name}_y_scalers.pkl"), "wb") as f:
                    pkl.dump(y_scalers, f)
                with open(os.path.join(self.save_dir, f"{model_type}_{save_file_name}_best_models.pkl"), "wb") as f:
                    pkl.dump(models, f)

                df_summary.to_csv(os.path.join(self.save_dir, f"{model_type}_{save_file_name}_summary.csv"), index=False)
                df_predictions.to_csv(os.path.join(self.save_dir, f"{model_type}_{save_file_name}_predictions.csv"), index=False)

                df_tmp = df_summary.query("set == 'test'")
                self.logger.info('')
                self.logger.info(f"Test MAE (mean +- 1 std): {df_tmp.MAE.mean():.4f} +- "
                                 f"{df_tmp.MAE.std():.4f}")
                self.logger.info(f"Test RMSE (mean +- 1 std): {df_tmp.RMSE.mean():.4f} +- "
                                 f"{df_tmp.RMSE.std():.4f}")
                self.logger.info(f"Test R2 (mean +- 1 std): {df_tmp.R2.mean():.4f} +- "
                                 f"{df_tmp.R2.std():.4f}")
                self.logger.info(f"Test Kendall Tau (mean +- 1 std): {df_tmp.kendall_tau_statistic.mean():.4f} +- "
                                 f"{df_tmp.kendall_tau_statistic.std():.4f}")
                self.logger.info(f"Test Spearman Rank (mean +- 1 std): {df_tmp.spearman_statistic.mean():.4f} +- "
                            f"{df_tmp.spearman_statistic.std():.4f}")
