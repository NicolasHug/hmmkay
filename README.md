# Hmmkay [![Build Status](https://travis-ci.org/NicolasHug/hmmkay.svg?branch=master)](https://travis-ci.org/NicolasHug/hmmkay) [![Documentation Status](https://readthedocs.org/projects/hmmkay/badge/?version=latest)](https://hmmkay.readthedocs.io/en/latest/?badge=latest) [![python versions](https://img.shields.io/badge/python-3.6+-blue.svg)](https://github.com/NicolasHug/hmmkay)

Discrete Hidden Markov Models with Numba

## Installation

    pip install hmmkay

Requires Python 3.6 or higher.

## Documentation

Docs are online at
[https://hmmkay.readthedocs.io/en/latest/](https://hmmkay.readthedocs.io/en/latest/)

## Status

Highly experimental, API subjet to change without deprecation.

## Development

The following packages are required for testing:

    pip install pytest hmmlearn scipy

For benchmarks:

    pip install matplotlib hmmlearn

For docs:

    pip install sphinx sphinx_rtd_theme


For development, use [pre-commit
hooks](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
for black and flake8.
