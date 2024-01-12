"""
A collection of functions that featurize an individual molecule via RDKit.
For all, the input is a SMILES string and the return is a 1D numpy array of the featurization.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors

try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
except ImportError:
    raise ImportError('Failed to import descriptastorus. Please install descriptastorus via '
                      'pip install git+https://github.com/bp-kelley/descriptastorus '
                      'to use RDKit 2D features.')

params = Chem.SmilesParserParams()
params.removeHs = False

_FP_FEATURIZERS = {}


def register_features_generator(features_generator_name):
    """
    Creates a decorator which registers a features generator function in a global dictionary.
    """
    def decorator(features_generator):
        _FP_FEATURIZERS[features_generator_name] = features_generator
        return features_generator

    return decorator


def rdkit_to_np(vect, nbits):
    """Helper function to convert a sparse vector from RDKit to a dense numpy vector."""
    arr = np.zeros((nbits,))
    DataStructs.ConvertToNumpyArray(vect, arr)  # overwrites arr
    return arr


# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedMorganFingerprint
@register_features_generator('morgan_counts')
def calc_morgan_counts_fp(smi,
                          radius=2,
                          num_bits=2048,
                          params=params,
                          ):
    mol = Chem.MolFromSmiles(smi, params)
    feature_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    # convert rdkit.DataStructs.cDataStructs.UIntSparseIntVect to np.array
    return rdkit_to_np(feature_vec, num_bits)


@register_features_generator('morgan_binary')
def calc_morgan_binary_fp(smi,
                          radius=2,
                          num_bits=2048,
                          params=params,
                          ):
    mol = Chem.MolFromSmiles(smi, params)
    feature_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    # convert rdkit.DataStructs.cDataStructs.ExplicitBitVect to np.array
    return rdkit_to_np(feature_vec, num_bits)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprint
@register_features_generator('AtomPair')
def calc_atom_pair_fp(smi, 
                      minPathLen=1, 
                      maxPathLen=30, 
                      nbits=2048, 
                      params=params,
                      ):
    mol = Chem.MolFromSmiles(smi, params)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprint(mol,
                                   minLength=minPathLen,
                                   maxLength=maxPathLen,
                                   nBits=nbits,
                                  )
    # convert rdkit.DataStructs.cDataStructs.IntSparseIntVect to np.array
    return rdkit_to_np(fp, nbits)


@register_features_generator('Avalon')
def calc_avalon_fp(smi,
                   nBits=512,
                   params=params,
                   ):
    mol = Chem.MolFromSmiles(smi, params)
    # convert rdkit.DataStructs.cDataStructs.ExplicitBitVect to np.array
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=nBits)
    
    return np.array(fp)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint
@register_features_generator('MACCS')
def calc_MACCS_fp(smi,
                  params=params,
                  ):
    mol = Chem.MolFromSmiles(smi, params)
    # convert rdkit.DataStructs.cDataStructs.ExplicitBitVect to np.array
    fp = MACCSkeys.GenMACCSKeys(mol)

    return np.array(fp)


@register_features_generator('MQN')
def calc_MQN_fp(smi,
                params=params,
):
    """
    Molecular Quantun Numbers (MQN) Descriptors.
    Consists of 4 categories, but only 42 features total:
    (1) Atom counts
    (2) Bond counts
    (3) Polarity counts
    (4) Topology counts
    """
    mol = Chem.MolFromSmiles(smi, params)
    # features are returned as a list
    fp = rdMolDescriptors.MQNs_(mol)

    return np.array(fp)


# https://github.com/bp-kelley/descriptastorus/blob/master/descriptastorus/descriptors/rdDescriptors.py#L287
@register_features_generator('rdkit_2d')
def calc_rdkit_2d_fp(smi):
    """
    Generates list of 200 2D features for a molecule.

    There are two major categories: 
    (1) physicochemical properties 
    (2) Fraction of a substructure (e.g., 'fr_Al_COO'). 
    Many molecules will have a lot of zeros for the 2nd category of descriptors.

    Just clone the repo and then run python setup.py install within the env 
    """
    generator = rdDescriptors.RDKit2D()
    fp = generator.process(smi)[1:]

    return np.array(fp)


@register_features_generator('rdkit_2d_normalized')
def calc_rdkit_2d_normalized_fp(smi):
    """Generates list of 200 2D normalized features for a molecule."""
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    fp = generator.process(smi)[1:]

    return np.array(fp)


def get_rxn_fp(rxn_smi, featurizer):
    """
    Helper function that creates a fingerprint for a reaction.

    Featurizer is one of the functions from above, which accepts a
    SMILES string as input and then return a fingerprint vector.
    """
    rsmi, psmi = rxn_smi.split('>>')
    fp_r = featurizer(rsmi)
    fp_p = featurizer(psmi)
    
    return np.concatenate((fp_r, fp_p - fp_r))
