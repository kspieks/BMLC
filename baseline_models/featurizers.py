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
@register_features_generator('morgan_count')
def calc_morgan_count_fp(smi,
                         radius=2,
                         num_bits=2048,
                         include_chirality=True,
                         ):
    mol = Chem.MolFromSmiles(smi)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=num_bits,
        includeChirality=include_chirality,
    )

    return morgan_gen.GetCountFingerprintAsNumPy(mol)


@register_features_generator('morgan_bit')
def calc_morgan_bit_fp(smi,
                       radius=2,
                       num_bits=2048,
                       include_chirality=True,
                       ):
    mol = Chem.MolFromSmiles(smi)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=num_bits,
        includeChirality=include_chirality,
    )
    
    return morgan_gen.GetFingerprintAsNumPy(mol)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprint
# https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetAtomPairGenerator
@register_features_generator('atompair_count')
def calc_atompair_count_fp(smi,
                           min_path_len=1,
                           max_path_len=30,
                           num_bits=2048,
                           include_chirality=True,
                           ):
    mol = Chem.MolFromSmiles(smi)
    atompair_gen = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=min_path_len,
        maxDistance=max_path_len,
        fpSize=num_bits,
        includeChirality=include_chirality,
    )

    return atompair_gen.GetCountFingerprintAsNumPy(mol)


@register_features_generator('atompair_bit')
def calc_atompair_bit_fp(smi,
                         min_path_len=1,
                         max_path_len=30,
                         num_bits=2048,
                         include_chirality=True,
                         ):
    mol = Chem.MolFromSmiles(smi)
    atompair_gen = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=min_path_len,
        maxDistance=max_path_len,
        fpSize=num_bits,
        includeChirality=include_chirality,
    )

    return atompair_gen.GetFingerprintAsNumPy(mol)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetRDKitFPGenerator
@register_features_generator('rdkit_count')
def calc_rdkit_count_fp(smi,
                        min_path_len=1,
                        max_path_len=7,
                        num_bits=2048,
                        ):
    mol = Chem.MolFromSmiles(smi)
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=min_path_len,
        maxPath=max_path_len,
        fpSize=num_bits,
    )

    return rdkit_gen.GetCountFingerprintAsNumPy(mol)


@register_features_generator('rdkit_bit')
def calc_rdkit_bit_fp(smi,
                      min_path_len=1,
                      max_path_len=7,
                      num_bits=2048,
                      ):
    mol = Chem.MolFromSmiles(smi)
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=min_path_len,
        maxPath=max_path_len,
        fpSize=num_bits,
    )

    return rdkit_gen.GetFingerprintAsNumPy(mol)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetTopologicalTorsionGenerator
@register_features_generator('topologicaltorsion_count')
def calc_topologicaltorsion_count_fp(smi,
                                     torsion_atom_count=4,
                                     num_bits=2048,
                                     include_chirality=True,
                                     ):
    mol = Chem.MolFromSmiles(smi)
    toptorsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        includeChirality=include_chirality,
        torsionAtomCount=torsion_atom_count,
        fpSize=num_bits,
    )

    return toptorsion_gen.GetCountFingerprintAsNumPy(mol)


@register_features_generator('topologicaltorsion_bit')
def calc_topologicaltorsion_bit_fp(smi,
                                   torsion_atom_count=4,
                                   num_bits=2048,
                                   include_chirality=True,
                                   ):
    mol = Chem.MolFromSmiles(smi)
    toptorsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        includeChirality=include_chirality,
        torsionAtomCount=torsion_atom_count,
        fpSize=num_bits,
    )

    return toptorsion_gen.GetFingerprintAsNumPy(mol)


# https://www.rdkit.org/docs/source/rdkit.Avalon.pyAvalonTools.html
@register_features_generator('avalon_count')
def calc_avalon_count_fp(smi, nBits=512):
    mol = Chem.MolFromSmiles(smi)
    # convert rdkit.DataStructs.cDataStructs.UIntSparseIntVect to np.array
    fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=nBits)
    
    return rdkit_to_np(fp, nBits)


@register_features_generator('avalon_bit')
def calc_avalon_bit_fp(smi, nBits=512):
    mol = Chem.MolFromSmiles(smi)
    # convert rdkit.DataStructs.cDataStructs.ExplicitBitVect to np.array
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=nBits)
    
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
