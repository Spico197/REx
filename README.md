# REx
![REx](https://github.com/Spico197/REx/workflows/REx/badge.svg?branch=main)
![Codecov](https://img.shields.io/codecov/c/github/Spico197/REx)
[![Documentation Status](https://readthedocs.org/projects/rex/badge/?version=main)](https://rex.readthedocs.io/en/main/?badge=main)


A toolkit for Relation Extraction and more...

This project has not finished yet, so be careful when using it.

## Dependencies
- Python>=3.6
  - torch>=1.2.0 : project is developed with torch==1.5.1, should be compatable with >=1.2.0 versions
  - numpy>=1.19.0

## Installation
```bash
pip install -e .

# or you can download and install from pypi
pip install pytorch-rex -i https://pypi.org/simple
```

## QuickStart
```bash
mkdir project_dir && cd project_dir
python -m rex startproject project_name # planned so, but has not finished
```

## Structures
- io: data loading, transformation operations and data building
  - [x] utils: data load/dump
  - [x] instance: common wrapper to make dict structure data able to access their attributes by `obj.attribute`
  - [ ] vocab: vocabulary building
  - [ ] transform: data cleaning/transformation, word cutting, and some utils for constructing data
- core: core tools for data preprocess
  - [ ] dataset: abstract dataset wrapper
  - [ ] loader: wrapper for data loaders
  - [ ] trainer: main runner for model training process
- metrics: evaluation measures, including many kinds of metrics
  - accuracy
  - (all|filtered)_(f1|precision|recall)
    - micro
    - macro
    - average
  - (max|all)\_(all|filtered)\_(prc|auc)
  - p@k
- modules: basic modules
  - embeddings
    - word2vec
    - glove
    - fasttext
    - tf-idf
    - elmo
    - bert
  - encoders
    - sentence encoders
    - bag encoders
  - selectors
    - MaxBag
    - OneBag
    - AttBag
  - decoders
    - crf
      - SimpleCRF
      - ConstraintCRF
    - span
      - SingleSpan
      - DualSpan
- models: entire models
- utils: others for building projects
  - [x] config
  - [x] logger

## Dataset Compatibility Adaptation
- [ ] IPRE preprocess
- [ ] NYT10
- [ ] NYT-H
- [ ] SemEval2010-Task8
- [ ] TACRED
- [ ] ACE05

## Support
- [ ] Chinese sentence level relation extraction
- [ ] Chinese bag level RE
- [ ] English sentence level RE
- [ ] English bag level RE

## Test
```bash
python -m unittest -v
```

## Docs
```bash
cd docs
sphinx-apidoc -o . ..
make html
```

## LICENCE
MIT
