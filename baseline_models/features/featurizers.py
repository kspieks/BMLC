"""
A collection of functions that featurize an individual molecule via RDKit.
For each function, the input is a SMILES string, and the return is a 1D numpy array.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import (Descriptors, MACCSkeys,
                        rdFingerprintGenerator,
                        rdMolDescriptors)
from rdkit.Chem.AtomPairs import Sheridan

try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
except ImportError:
    raise ImportError('Failed to import descriptastorus. Please install descriptastorus via '
                      'pip install git+https://github.com/bp-kelley/descriptastorus '
                      'to use RDKit 2D features.')

from .utils import rdkit_to_np, _hash_fold
from rdkit_2d_features_list import CURRENT_VERSION, RDKIT_PROPS

_FP_FEATURIZERS = {}


def register_features_generator(features_generator_name):
    """
    Creates a decorator which registers a features generator function in a global dictionary.
    """
    def decorator(features_generator):
        _FP_FEATURIZERS[features_generator_name] = features_generator
        return features_generator

    return decorator


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
    """
    Publication: Carhart, R.E. et al. "Atom Pairs as Molecular Features in 
    Structure-Activity Studies: Definition and Applications” J. Chem. Inf. Comp. Sci. 25:64-73 (1985).

    "An atom pair substructure is defined as a triplet of two non-hydrogen atoms and their shortest
    path distance in the molecular graph, i.e. (atom type 1, atom type 2, geodesic distance).
    In the standard RDKit implementation, distinct atom types are defined by tuples of atomic number, 
    number of heavy atom neighbours, aromaticity and chirality. All unique triplets in a molecule
    are enumerated and stored in sparse count or bit vector format."
    https://www.blopig.com/blog/2022/06/exploring-topological-fingerprints-in-rdkit/
    """
    mol = Chem.MolFromSmiles(smi)
    atompair_gen = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=minDistance,
        maxDistance=maxDistance,
        fpSize=fpSize,
        includeChirality=includeChirality,
    )
    fp = getattr(atompair_gen,
                 f'Get{"Count" if count else ""}Fingerprint'
                 )(mol)

    return rdkit_to_np(fp, fpSize)


# https://www.rdkit.org/docs/source/rdkit.Avalon.pyAvalonTools.html
@register_features_generator('avalon')
def calc_avalon_fp(smi, nBits=512, count=True):
    mol = Chem.MolFromSmiles(smi)

    fp = getattr(pyAvalonTools,
                 f'GetAvalon{"Count" if count else ""}FP'
                 )(mol)

    return rdkit_to_np(fp, nBits)


# https://rdkit.org/docs/source/rdkit.Chem.AtomPairs.Sheridan.html
@register_features_generator('donorpair')
def get_donorpair_fp(smi, fpSize=1024):
    mol = Chem.MolFromSmiles(smi)
    sparse_vec = Sheridan.GetBPFingerprint(mol)
    nze = sparse_vec.GetNonzeroElements()

    return _hash_fold(nze, fpSize)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint
# http://rdkit.org/docs/source/rdkit.Chem.MACCSkeys.html
@register_features_generator('MACCS')
def calc_MACCS_fp(smi):
    """ 
    MACCS Keys have no hyperparameters to vary.
    RDKit preserves the MACCS key numbers, so that MACCS key 23 (for example) is bit number 23. 
    Bit 0 is always unset and may be ignored. Only bits 1-166 will be set.
    https://github.com/rdkit/rdkit/issues/1726

    Source code: https://github.com/rdkit/rdkit-orig/blob/master/rdkit/Chem/MACCSkeys.py

    Note that `MACCSkeys.GenMACCSKey` is identical to `rdMolDescriptors.GetMACCSKeysFingerprint`
    https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py#L299
    """
    mol = Chem.MolFromSmiles(smi)
    fp = MACCSkeys.GenMACCSKeys(mol)
    # convert rdkit.DataStructs.cDataStructs.ExplicitBitVect to np.array
    return rdkit_to_np(fp, 167)[1:]     # ignore bit 0


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
    fp = getattr(morgan_gen,
                 f'Get{"Count" if count else ""}Fingerprint'
                 )(mol)

    return rdkit_to_np(fp, fpSize)


@register_features_generator('MQN')
def calc_MQN_fp(smi):
    """
    Molecular Quantun Numbers (MQN) Descriptors.
    Consists of 4 categories, but only 42 features total:
    (1) Atom counts
    (2) Bond counts
    (3) Polarity counts
    (4) Topology counts

    Source code: https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Descriptors/MQN.cpp
    Publication: Nguyen et al. "Classification of organic molecules by molecular quantum numbers."
                 ChemMedChem 4:1803-5 (2009).
    """
    mol = Chem.MolFromSmiles(smi)
    # features are returned as a list. Convert them to a numpy array.
    fp = rdMolDescriptors.MQNs_(mol)

    return np.array(fp, dtype=np.float64)


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
    fp = getattr(rdkit_gen,
                 f'Get{"Count" if count else ""}Fingerprint'
                 )(mol)

    return rdkit_to_np(fp, fpSize)


# https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetTopologicalTorsionGenerator
@register_features_generator('topologicaltorsion')
def calc_topologicaltorsion_fp(smi,
                               count=True,
                               torsionAtomCount=4,
                               fpSize=2048,
                               includeChirality=True,
                               ):
    """
    Topological torsion fingerprints aim to complement the predominantly long-range
    relationships captured in atom pair fingerprints by representing short-range information
    contained in the torsion angles of a molecule. They use the same atom type definitions as
    atom pair fingerprints, but only count four consecutively bonded non-hydrogen atoms along
    with the number of non-hydrogen branches.
    https://www.blopig.com/blog/2022/06/exploring-topological-fingerprints-in-rdkit/

    Publication: Nilakantan, R. et al. "Topological torsion: a new molecular descriptor for SAR applications.
    Comparison with other descriptors" J. Chem. Inf. Comput. Sci. 27, 2, 82-85 (1987).
    """
    mol = Chem.MolFromSmiles(smi)
    toptorsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        fpSize=fpSize,
        torsionAtomCount=torsionAtomCount,
        includeChirality=includeChirality,
    )
    fp = getattr(toptorsion_gen,
                 f'Get{"Count" if count else ""}Fingerprint'
                 )(mol)

    return rdkit_to_np(fp, fpSize)


# https://github.com/bp-kelley/descriptastorus/blob/master/descriptastorus/descriptors/rdDescriptors.py#L287
@register_features_generator('rdkit_2d')
def calc_rdkit_2d_fp(smi, properties=RDKIT_PROPS[CURRENT_VERSION]):
    """
    Generates 2D features for a molecule.
    By default, it generates an array containing 200 features,
    but this can be changed by modifying which properties are calculated.

    There are two major categories: 
    (1) physicochemical properties 
    (2) Fraction of a substructure (e.g., 'fr_Al_COO'). 
    Many molecules will have a lot of zeros for the 2nd category of descriptors.

    Just clone the repo and then run python setup.py install within the env 
    """
    generator = rdDescriptors.RDKit2D(properties=properties)
    # features are returned as a list. Convert them to a numpy array.
    fp = generator.process(smi)[1:]

    return np.array(fp, dtype=np.float64)


@register_features_generator('rdkit_2d_normalized')
def calc_rdkit_2d_normalized_fp(smi, properties=RDKIT_PROPS[CURRENT_VERSION]):
    """
    Generates 2D normalized features for a molecule.
    By default, it generates an array containing 200 features,
    but this can be changed by modifying which properties are calculated.
    """
    generator = rdNormalizedDescriptors.RDKit2DNormalized(properties=properties)
    # features are returned as a list. Convert them to a numpy array.
    fp = generator.process(smi)[1:]

    return np.array(fp, dtype=np.float64)

