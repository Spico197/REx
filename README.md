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

## ⚙️Installation

Make sure you have installed all the dependencies below.

- Python>=3.6
  - torch>=1.2.0 : project is developed with torch==1.7.1, should be compatable with >=1.2.0 versions
  - numpy>=1.19.0
  - scikit-learn>=0.21.3
  - click>=7.1.2
  - omegaconf>=2.0.6
  - loguru>=0.5.3
  - tqdm>=4.61.1
  - transformers>=4.8.2

```bash
$ git clone https://github.com/Spico197/REx.git
$ cd REx
$ pip install -e .

# or you can download and install from pypi, not recommend for now
$ pip install pytorch-rex -i https://pypi.org/simple
```

## 🚀QuickStart

Checkout the `examples` folder.

| Name        | Model    | Dataset | Task                                                             |
| :---------- | :------- | :------ | :--------------------------------------------------------------- |
| SentRE-MCML | PCNN     | IPRE    | Sentence-level Multi-class multi-label relation classification   |
| BagRE       | PCNN+ONE | NYT10   | Bag-level relation classification (Multi-Instance Learning, MIL) |
| JointERE    | CasRel   | WebNLG  | Jointly entity relation extraction                               |

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

## 🌴Development

Make sure you have installed the following packages:

- coverage
- flake8
- sphinx
- sphinx_rtd_theme

### Build

```bash
$ make all
```

### Test
```bash
pip install coverage
coverage run -m unittest -v && coverage report

# or just test without coverage report
make test

# or test with report
make test_report
```

### Docs
```bash
cd docs
sphinx-apidoc -o . ..
make html

# or just
make docs
```

## ✉️Update

- v0.0.2: add black formatter and pytest testing
- v0.0.1: change `LabelEncoder.to_binary_labels` into `convert_to_multi_hot` or `convert_to_one_hot`


## 🔑LICENCE
MIT
