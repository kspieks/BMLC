"""
Collection of functions to plot results from a regression model.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
sns.set()
sns.set_context('talk')
sns.set_style('darkgrid', {'axes.edgecolor': '0.2',
                           'xtick.bottom': True,
                           'ytick.left': True
                          })

# define custom colors
COLOR = '#721f81'  # purple

def fit_linear(X, y):
    """Use OLS to fit a regression line"""
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model

def fit_quadratic(X, y):
    """Use OLS with polynomial order 2"""
    model = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
    model.fit(X, y)
    return model

def draw_best_fit(X, y, ax, estimator='linear', **kwargs):
    estimators = {
        'linear': fit_linear,
        'quadratic': fit_quadratic
    }
    
    estimator = estimators[estimator]
    
    # check that X and y are the same length
    assert len(X) == len(y)
    
    # ensure that X and y are np arrays
    X, y = np.array(X), np.array(y)
    
    # verify that X is a 2D array for scikit-learn estimators
    if X.ndim < 2:
        X = X[:, np.newaxis]  # reshape X into the correct dimensions
    
    # verify that y is a (n,) array
    assert y.ndim == 1
    
    # use the estimator to fit the data
    model = estimator(X, y)
    
    # get the current working axis
    ax = ax or plt.gca()
    
    # Plot line of best fit onto the axes that were passed in.
    # TODO: determine if xlim or X.min(), X.max() are better params
    xr = np.linspace(*ax.get_xlim(), num=100)
    
    label = f"best fit: $R^2$ = {r2_score(X, y):0.3f}"
    ax.plot(xr, model.predict(xr[:, np.newaxis]), label=label, **kwargs)
    return ax

def draw_identity(ax, ax_min, ax_max, **kwargs):
    ax = ax or plt.gca()
    
    # points = np.linspace(*ax.get_xlim(), num=100)
    points = np.linspace(ax_min, ax_max, 100)
    ax.plot(points, points, **kwargs)
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    return ax

def draw_parity_plot(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sns.scatterplot(x=y_true, y=y_pred,
                        color=COLOR, alpha=0.7,
                        size=6,
                        legend=False,
                        # label=None,
                        ax=ax,
                        )

    ax_min = np.floor(min(min(y_true), min(y_pred)))
    ax_max = np.floor(max(max(y_true), max(y_pred)))
    # print(ax_min, ax_max)

    ax = draw_identity(ax, ax_min, ax_max,
                    linestyle='--',
                    color='k', alpha=0.4,
                    linewidth=1.75,
                    label='identity',
                    )

    ax = draw_best_fit(y_true, y_pred, ax,
                    linestyle='--',
                    color='k', alpha=0.9,
                    linewidth=1.75,
                    )

    ax.legend(fontsize=13)

    ax.set_xlabel('True Value')
    ax.set_xlabel('Predicted Value')
    return fig, ax
