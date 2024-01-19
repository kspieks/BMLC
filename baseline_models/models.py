from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor


_MODELS = {}


def register_model_generator(features_generator_name):
    """
    Creates a decorator which registers a model function in a global dictionary.
    """
    def decorator(features_generator):
        _MODELS[features_generator_name] = features_generator
        return features_generator

    return decorator


@register_model_generator('LinearSVR')
def get_LinearSVR(trial, random_state=42):
    params = {
        "C": trial.suggest_float("C", 1e-6, 1e6, log=True),
        "max_iter": 1000,
        "random_state": random_state,
    }

    regressor_obj = LinearSVR(**params)

    return regressor_obj

@register_model_generator('SVR')
def get_SVR(trial, random_state=42):
    """
    Random state is not used for PLS. 
    It is passed in to be consistent with the syntax of the other models.
    """
    params = {
        "kernel": 'rbf',
        "C": trial.suggest_float("C", 1e-5, 1e5, log=True),
        "max_iter": 500,
    }

    regressor_obj = SVR(**params)

    return regressor_obj


@register_model_generator('RF')
def get_RF(trial, random_state=42):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", low=50, high=300, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "random_state": random_state,
    }

    regressor_obj = RandomForestRegressor(**params)

    return regressor_obj


@register_model_generator('XGB')
def get_XGB(trial, random_state=42):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", low=10, high=200, step=10),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 5),    # L1 regularization
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 5),  # L2 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 5),
        "gamma": trial.suggest_int("gamma", 0, 5),  # minimum loss reduction required to make further partition on a leaf node
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),  # lower values require more trees
        "random_state": random_state,
        "n_jobs": 1,
    }
    regressor_obj = XGBRegressor(**params)

    return regressor_obj


@register_model_generator('MLP')
def get_MLP(trial, random_state=42):
    num_layers = trial.suggest_int("num_layers", low=1, high=10, step=1)
    layer_size = trial.suggest_int("layer_size", low=100, high=500, step=50)
    params = {
        "hidden_layer_sizes" : [layer_size for i in range(num_layers)],
        "alpha": trial.suggest_loguniform("alpha", 1e-6, 1e-1),   # L2 regularization
        "learning_rate_init": trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-1),
        "random_state": random_state, 
    }
    regressor_obj = MLPRegressor(**params)

    return regressor_obj


@register_model_generator('PLS')
def get_PLS(trial, random_state=42):
    """
    Random state is not used for PLS. 
    It is passed in to be consistent with the syntax of the other models.
    """
    params = {
        "n_components": trial.suggest_int("n_components", low=1, high=15, step=1)
    }
    regressor_obj = PLSRegression(**params)

    return regressor_obj


@register_model_generator('Lasso')
def get_Lasso(trial, random_state=42):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-6, 1e6, log=True),
        "random_state": random_state,
    }
    regressor_obj = linear_model.Lasso(**params)

    return regressor_obj


@register_model_generator('Ridge')
def get_Ridge(trial, random_state=42):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-6, 1e6, log=True),
        "random_state": random_state,
    }
    regressor_obj = linear_model.Ridge(**params)

    return regressor_obj
