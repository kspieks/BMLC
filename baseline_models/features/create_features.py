import numpy as np
from pandarallel import pandarallel

from .featurizers import _FP_FEATURIZERS
from .utils import get_rxn_fp


def create_features(df,
                    smiles_column='smiles',
                    rxn_mode=False,
                    featurizer='morgan',
                    parameters=[{"count": True,
                                 "radius": 2,
                                 "fpSize": 2048,
                                }],
                    n_cpus=1,
                    ):
    """
    Calculates features for a set of SMILES strings in a pd.Dataframe.

    Args:
        df: pandas dataframe which contains a column of SMILES strings.
        smiles_column: name of the column containing the SMILES strings.
        rxn_mode: whether the SMILES strings represent a reaction.
        featurizer: name of the featurizer. Must be a key in _FP_FEATURIZERS.
        parameters: list of parameter dictionaries that specify the settings for the featurizer.
        n_cpus: number of cpus to be used when generating the features. Useful for large datasets.
    
    Returns:
        X: np.array of shape N molecules (or reactions) x N features.
    """

    featurizers = featurizer.split('+')
    if len(featurizers) != len(parameters):
        msg = "Dimension mismatch!\n"
        msg += f"There are {len(featurizers)} featurizers specified and "
        msg += f"{len(parameters)} parameter dictionaries specified in the input arguments.\n"
        msg += "These values should be identical."
        raise ValueError(msg)

    # calculate input features X (i.e., fingerprint vectors) in parallel
    pandarallel.initialize(nb_workers=n_cpus, progress_bar=True)
    fp_arrays = []
    for f, params in zip(featurizers, parameters):
        # reaction mode
        if rxn_mode:
            params['featurizer'] = _FP_FEATURIZERS[f]
            fps = df[smiles_column].parallel_apply(get_rxn_fp, **params)

        # molecule mode
        else:
            fps = df[smiles_column].parallel_apply(_FP_FEATURIZERS[f], **params)

        fp_array = np.stack(fps.values)
        fp_arrays.append(fp_array)

    X = np.hstack(fp_arrays)

    return X
