import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .models import _MODELS


def calc_regression_metrics(y_true, y_pred, ranking_metrics=True):
    """
    Calculate performance metrics for the regression model.
    The naive baseline model used in this codebase naively predicts
    a constant value for the y_pred vector so fro this case, it does not
    make sense to report kendall tau or spearman rank metrics.
    
    Args:
        y_true: np.array of true values.
        y_pred: np.array of values predicted from a regression model.
        ranking_metrics: boolean indicating whether to return ranking metrics.
    """
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = root_mean_squared_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)

    metrics = {
        'MAE': MAE,
        'RMSE': RMSE,
        'R2': R2
    }

    if ranking_metrics:
        kendalltau = stats.kendalltau(y_true, y_pred)
        metrics['kendall_tau_statistic'] = kendalltau.statistic
        metrics['kendall_tau_pvalue'] = kendalltau.pvalue

        spearman = stats.spearmanr(y_true, y_pred)
        metrics['spearman_statistic'] = spearman.statistic
        metrics['spearman_pvalue'] = spearman.pvalue

    return metrics


def naive_baseline(y, splits, logger):
    """
    Naively predicts the mean value from the training set for all test molecules.
    """
    dfs_summary_tmp = [] 

    for i, (train_indices, val_indices, test_indices) in enumerate(splits):
        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]

        logger.info(f'Mean target value for training fold {i}:   {y_train.mean():.3f}')
        logger.info(f'Mean target value for validation fold {i}: {y_val.mean():.3f}')
        logger.info(f'Mean target value for testing fold {i}:    {y_test.mean():.3f}\n')

        # baseline model uses mean of the training set
        y_pred_train = [y_train.mean()] * len(y_train)
        y_pred_val = [y_train.mean()] * len(y_val)
        y_pred_test = [y_train.mean()] * len(y_test)

        metrics = calc_regression_metrics(y_train, y_pred_train, ranking_metris=True)
        dfs_summary_tmp.append(['train', i, metrics['MAE'], metrics['RMSE'], metrics['R2']])

        metrics = calc_regression_metrics(y_val, y_pred_val, ranking_metris=True)
        dfs_summary_tmp.append(['val', i, metrics['MAE'], metrics['RMSE'], metrics['R2']])

        metrics = calc_regression_metrics(y_test, y_pred_test, ranking_metris=True)
        dfs_summary_tmp.append(['test', i, metrics['MAE'], metrics['RMSE'], metrics['R2']])

    cols = ['set', 'split', 
            'MAE', 'RMSE', 'R2', 
            ]
    df_summary = pd.DataFrame(dfs_summary_tmp, columns=cols)

    # log summary statistics for the test set
    df_tmp = df_summary.query("set == 'test'")
    # df.std() uses dof=1 by default
    logger.info(f"Test MAE (mean +- 1 std): {df_tmp.MAE.mean():.4f} +- "
                f"{df_tmp.MAE.std():.4f}")
    logger.info(f"Test RMSE (mean +- 1 std): {df_tmp.RMSE.mean():.4f} +- "
                f"{df_tmp.RMSE.std():.4f}")
    logger.info(f"Test R2 (mean +- 1 std): {df_tmp.R2.mean():.4f} +- "
                f"{df_tmp.R2.std():.4f}")

    return df_summary


# Explore hyperparameters
def train_model(model,
                X,
                y,
                splits,
                ):

    dfs_pred_tmp = []
    dfs_summary_tmp = []
    models = []
    scalers = []

    for i, (train_indices, val_indices, test_indices) in enumerate(splits):
        X_train = X[train_indices]
        X_val = X[val_indices]
        X_test = X[test_indices]

        # reshape size to be N x 1
        y_train = y[train_indices].reshape(-1, 1)
        y_val = y[val_indices].reshape(-1, 1)
        y_test = y[test_indices].reshape(-1, 1)

        scaler = StandardScaler()
        scaler.fit(y_train)
        scalers.append(scaler)

        y_train_scaled = scaler.transform(y_train)  # N x 1

        # fit the model to the training data
        model.fit(X_train, y_train_scaled.flatten())
        models.append(model)

        # get predictions. reshape size to be N x 1
        y_pred_train = scaler.inverse_transform(model.predict(X_train).reshape(-1, 1))
        y_pred_val = scaler.inverse_transform(model.predict(X_val).reshape(-1, 1))
        y_pred_test = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1))

        data_dict = {'y_test': y_test.flatten(),
                     'y_pred_test': y_pred_test.flatten(),
                     'split': [i] * len(y_test),
                     }
        dfs_pred_tmp.append(pd.DataFrame(data_dict))

        # store performance metrics
        # training set
        metrics = calc_regression_metrics(y_train, y_pred_train)
        dfs_summary_tmp.append(['train', i, metrics['MAE'], metrics['RMSE'], metrics['R2'],
                                metrics['kendall_tau_statistic'], metrics['kendall_tau_pvalue'],
                                metrics['spearman_statistic'], metrics['spearman_pvalue'],
                                ])

        # validation set
        metrics = calc_regression_metrics(y_val, y_pred_val)
        dfs_summary_tmp.append(['val', i, metrics['MAE'], metrics['RMSE'], metrics['R2'],
                                metrics['kendall_tau_statistic'], metrics['kendall_tau_pvalue'],
                                metrics['spearman_statistic'], metrics['spearman_pvalue'],
                                ])

        # testing set
        metrics = calc_regression_metrics(y_test, y_pred_test)
        dfs_summary_tmp.append(['test', i, metrics['MAE'], metrics['RMSE'], metrics['R2'],
                                metrics['kendall_tau_statistic'], metrics['kendall_tau_pvalue'],
                                metrics['spearman_statistic'], metrics['spearman_pvalue'],
                                ])

    cols = ['set', 'split', 
            'MAE', 'RMSE', 'R2', 
            'kendall_tau_statistic', 'kendall_tau_pvalue',
            'spearman_statistic', 'spearman_pvalue',
            ]
    df_summary = pd.DataFrame(dfs_summary_tmp, columns=cols)

    df_predictions = pd.concat(dfs_pred_tmp)

    return df_summary, df_predictions, models, scalers


# todo: could maybe use the `class_weight` parameter if our dataset is unbalanced
class Objective:
    def __init__(self,
                 X,
                 y,
                 splits,
                 model_type,
                 _MODELS=_MODELS,
                 random_state=42,
                 ):
        self.X = X
        self.y = y
        self.splits = splits
        self.model_type = model_type

        self._MODELS = _MODELS

        self.random_state = random_state

    def __call__(self, trial):
        regressor_obj = self._MODELS[self.model_type](trial, random_state=self.random_state)

        trial.set_user_attr(key="best_model", value=regressor_obj)

        df_summary, _, _, _ = train_model(regressor_obj,
                                          X=self.X,
                                          y=self.y,
                                          splits=self.splits,
                                          )

        return df_summary.query("set == 'val'")['RMSE'].mean()

# add callback to easily recall the best model parameters
def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])
