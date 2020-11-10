# REx
A toolkit for Relation Extraction and more...

This project has not finished yet, so be careful when using it.

## Installation
```bash
pip install -e .
```

## QuickStart
```bash
mkdir project_dir && cd project_dir
python -m rex startproject # planned so, but has not finished
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
