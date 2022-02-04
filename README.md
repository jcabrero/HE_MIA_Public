# HE_MIA: Homomorphic Encryption for Membership Inference Attacks

[![Build Python Wheelhouse](https://github.com/jcabrero/HE_MIA/actions/workflows/release.yml/badge.svg)](https://github.com/jcabrero/HE_MIA/actions/workflows/release.yml)


The repository contains a list of resources to use a homomorphically encrypted inference from any logistic regression whose weights size is smaller than 2<sup>14</sup> floating point elements. 

From the wheelhouse directory, take the version according to your version of Python. Currently it only works in Linux environments. 

## Installation

```
git clone <repo_address>
cd wheelhouse
pip3 install <installer>.whl
```

## Basic usage

```
from HELR import HELR
...
my_private_inferece = HELR(weights, bias)
result = my_private_inference.predict(sample)
...
```
