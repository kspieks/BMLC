"""
A collection of functions that featurize an individual molecule via RDKit.
For all, the input is a SMILES string and the return is a 1D numpy array of the featurization.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors, rdFingerprintGenerator

try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
except ImportError:
    raise ImportError('Failed to import descriptastorus. Please install descriptastorus via '
                      'pip install git+https://github.com/bp-kelley/descriptastorus '
                      'to use RDKit 2D features.')


_FP_FEATURIZERS = {}


def register_features_generator(features_generator_name):
    """
    Creates a decorator which registers a features generator function in a global dictionary.
    """
    def decorator(features_generator):
        _FP_FEATURIZERS[features_generator_name] = features_generator
        return features_generator

    return decorator


def rdkit_to_np(vect, num_bits):
    """Helper function to convert a sparse vector from RDKit to a dense numpy vector."""
    arr = np.zeros((num_bits,))
    DataStructs.ConvertToNumpyArray(vect, arr)  # overwrites arr
    return arr


# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedMorganFingerprint
# https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator
@register_features_generator('morgan')
def calc_morgan_fp(smi,
                   count=True,
                   radius=2,
                   fpSize=2048,
                   includeChirality=True,
                   ):
    "Extended Connectivity Fingerprint (MorganFingerprint from RDKit)"
    mol = Chem.MolFromSmiles(smi)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fpSize,
        includeChirality=includeChirality,
    )

    return getattr(morgan_gen,
            f'Get{"Count" if count else ""}FingerprintAsNumPy'
            )(mol)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprint
# https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetAtomPairGenerator
@register_features_generator('atompair')
def calc_atompair_fp(smi,
                     count=True,
                     minDistance=1,
                     maxDistance=30,
                     fpSize=2048,
                     includeChirality=True,
                     ):
    mol = Chem.MolFromSmiles(smi)
    atompair_gen = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=minDistance,
        maxDistance=maxDistance,
        fpSize=fpSize,
        includeChirality=includeChirality,
    )
    
    return getattr(atompair_gen,
            f'Get{"Count" if count else ""}FingerprintAsNumPy'
            )(mol)





# https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetRDKitFPGenerator
@register_features_generator('rdkit')
def calc_rdkit_fp(smi,
                  count=True,
                  minPath=1,
                  maxPath=7,
                  fpSize=2048,
                  ):
    mol = Chem.MolFromSmiles(smi)
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=minPath,
        maxPath=maxPath,
        fpSize=fpSize,
    )

    return getattr(rdkit_gen,
            f'Get{"Count" if count else ""}FingerprintAsNumPy'
            )(mol)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetTopologicalTorsionGenerator
@register_features_generator('topologicaltorsion')
def calc_topologicaltorsion_count_fp(smi,
                                     count=True,
                                     torsionAtomCount=4,
                                     fpSize=2048,
                                     includeChirality=True,
                                     ):
    mol = Chem.MolFromSmiles(smi)
    toptorsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        fpSize=fpSize,
        torsionAtomCount=torsionAtomCount,
        includeChirality=includeChirality,
    )

    return getattr(toptorsion_gen,
            f'Get{"Count" if count else ""}FingerprintAsNumPy'
            )(mol)


# https://www.rdkit.org/docs/source/rdkit.Avalon.pyAvalonTools.html
@register_features_generator('avalon')
def calc_avalon_fp(smi, nBits=512, count=True):
    mol = Chem.MolFromSmiles(smi)

    fp = getattr(pyAvalonTools,
                 f'GetAvalon{"Count" if count else ""}FP'
                 )(mol)

    # convert rdkit.DataStructs.cDataStructs.UIntSparseIntVect if count = True
    # or rdkit.DataStructs.cDataStructs.ExplicitBitVect if count = False to np.array
    return rdkit_to_np(fp, nBits)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint
# http://rdkit.org/docs/source/rdkit.Chem.MACCSkeys.html
@register_features_generator('MACCS')
def calc_MACCS_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    # convert rdkit.DataStructs.cDataStructs.ExplicitBitVect to np.array
    fp = MACCSkeys.GenMACCSKeys(mol)

    return np.array(fp)


@register_features_generator('MQN')
def calc_MQN_fp(smi):
    """
    Molecular Quantun Numbers (MQN) Descriptors.
    Consists of 4 categories, but only 42 features total:
    (1) Atom counts
    (2) Bond counts
    (3) Polarity counts
    (4) Topology counts
    """
    mol = Chem.MolFromSmiles(smi)
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
