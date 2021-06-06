<p align="center">
  <br>
  <img src="docs/REx.png" width="300"/>
  <br>
<p>

<p align="center">
  <a href="https://github.com/Spico197/REx/actions">
    <img alt="Build" src="https://github.com/Spico197/REx/workflows/REx/badge.svg?branch=main">
  </a>
  <a href="https://codecov.io/gh/Spico197/REx">
    <img alt="CodeCoverage" src="https://img.shields.io/codecov/c/github/Spico197/REx">
  </a>
  <a href="https://rex.readthedocs.io/en/main/?badge=main">
    <img alt="Docs" src="https://readthedocs.org/projects/rex/badge/?version=main">
  </a>
</p>

A toolkit for Relation & Event eXtraction (REx) and more...

This project has not been finished yet, so be careful when using it, or wait until the first release comes out.

This project is suffering from the second-system effect.
I would like to cut some features to make this going smoothly.

[Accelerate](https://github.com/huggingface/accelerate) seems to be a very sweet wrapper for multi-GPU, TPU training, we highly recommend you to use such frameworks, instead of adding hard codes on your own.

## 🌴Dependencies
- Python>=3.6
  - torch>=1.2.0 : project is developed with torch==1.5.1, should be compatable with >=1.2.0 versions
  - numpy==1.19.0
  - scikit-learn==0.21.3
  - click==7.1.2
  - omega==2.0.6
  - loguru==0.5.3

## ⚙️Installation
```bash
$ pip install -e .

# or you can download and install from pypi
$ pip install pytorch-rex -i https://pypi.org/simple
```

## 🚀QuickStart

Checkout the `examples`.

## ✈️Abilities

### Dataset
- IPRE preprocess
- NYT10

### Tasks
- Chinese sentence-level relation extraction
- English bag-level relation extraction

### Modules & Models

- Piecewise CNN
- PCNN + ONE
- PCNN + ATT

## 🧪Test
```bash
pip install coverage
coverage run -m unittest -v && coverage report

# or just test without coverage report
make test

# or test with report
make test_report
```

## 📃Docs
```bash
cd docs
sphinx-apidoc -o . ..
make html

# or just
make docs
```

## 🔑LICENCE
MIT
