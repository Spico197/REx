# REx
[![REx](https://github.com/Spico197/REx/workflows/REx/badge.svg?branch=main)](https://github.com/Spico197/REx/actions)
[![Codecov](https://img.shields.io/codecov/c/github/Spico197/REx)](https://codecov.io/gh/Spico197/REx)
[![Documentation Status](https://readthedocs.org/projects/rex/badge/?version=main)](https://rex.readthedocs.io/en/main/?badge=main)


A toolkit for Relation Extraction and more...

This project has not finished yet, so be careful when using it, or wait until the first release comes out.

This project is suffering from the second-system effect.
I would like to cut some features to make this going smoothly.

## Dependencies
- Python>=3.6
  - torch>=1.2.0 : project is developed with torch==1.5.1, should be compatable with >=1.2.0 versions
  - numpy>=1.19.0
  - hydra-core==1.0.6
  - loguru==0.5.3

## Installation
```bash
pip install -e .

# or you can download and install from pypi
pip install pytorch-rex -i https://pypi.org/simple
```

## QuickStart

Checkout the `examples`.

## Roadmap
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

### Dataset
- [x] IPRE preprocess
- [ ] NYT10
- [ ] NYT-H
- [ ] SemEval2010-Task8
- [ ] TACRED
- [ ] ACE05

### Tasks
- [x] Chinese sentence level relation extraction
- [ ] Chinese bag level RE
- [ ] English sentence level RE
- [ ] English bag level RE

### Modules & Models

- [x] Piecewise CNN
- [ ] 

## Test
```bash
pip install coverage
coverage run -m unittest -v && coverage report

# or just test without coverage report
make test

# or test with report
make test_report
```

## Docs
```bash
cd docs
sphinx-apidoc -o . ..
make html

# or just
make docs
```

## LICENCE
MIT
