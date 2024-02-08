import numpy as np
from rdkit import DataStructs


def _sum_digits(n: int, nBits: int) -> int:
    width = len(str(nBits))   # nBits is 1024 or 2048 so width = 4
    mod = np.power(10, width, dtype=np.int32)
    r = 0
    while n:
        r, n = r + n % mod, n//mod
    
    return r


def _hash_fold(nze: dict, nBits: int) -> np.array:
    # create vector of zeros as a placeholder
    vec = np.zeros(nBits, dtype=np.int32)
    bits = list(nze. keys())
    for bit in bits:
        n = _sum_digits (bit, nBits)
        idx = n % nBits
        vec[idx] = nze[bit]

    return vec


def rdkit_to_np(vect, num_bits) -> np.array:
    """
    Helper function to convert a sparse rdkit.DataStructs.cDataStructs.ExplicitBitVect
    or rdkit.DataStructs.cDataStructs.UIntSparseIntVect vector to a dense numpy vector.
    Otherwise, the featurizer generator methods of `GetFingerprintAsNumPy` and
    `GetCountFingerprintAsNumPy` return numpy arrays of dtype uint32, which leads
    to overflow errors when these vectors are subtracted to create the reaction
    representation in `get_rxn_fp`.
    """
    arr = np.zeros((num_bits,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(vect, arr)  # overwrites arr
    return arr


def get_rxn_fp(rxn_smi, featurizer, **params) -> np.array:
    """
    Helper function that creates a fingerprint for a reaction.

    Featurizer is one of the functions from featurizers.py, which accepts a
    SMILES string as input and then return a fingerprint vector.
    """
    rsmi, psmi = rxn_smi.split('>>')
    fp_r = featurizer(rsmi, **params)
    fp_p = featurizer(psmi, **params)
    
    return np.concatenate((fp_r, fp_p - fp_r))
