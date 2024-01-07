import os

from setuptools import find_packages, setup

__version__ = None


# utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="BMLC",
    version=__version__,
    author="Kevin Spiekermann",
    description="This codebase (Baseline Machine Learning for Cheminformatics) trains several baseline ML models from sklearn using various fingerprint representations from RDKit.",
    url="https://github.com/kspieks/BMLC",
    packages=find_packages(),
    long_description=read('README.md'),
)
