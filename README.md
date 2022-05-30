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
    <img src="https://codecov.io/gh/Spico197/REx/branch/main/graph/badge.svg"/>
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

- To create new task for classification, try: `rex new classification <task_name>`
- To create new task for tagging, try: `rex new tagging <task_name>`


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

## ⚗ Development

1. fork to your namespace

2. install `pytorch-rex` with `[dev]` option

```bash
$ pip install pytorch-rex[dev]
```

3. install pre-commit hooks

```bash
$ pre-commit install
```

## ✉️Update

- v0.1.4: move accelerate to `rex.__init__`, update multi process tqdm & logging (only show in the main process in default), remove cache in debug mode, fix bugs in `rex.cmds.new`, add `rank_zero_only` in task dump, `load_best_ckpt` if `resumed_training`
- v0.1.3: fix emb import
- v0.1.1: update registry and add `accelerate` multi-gpu support
- v0.1.0: huge update with lots of new features, check the usage in `examples/IPRE` ~
- v0.0.15: add safe_try to kill ugly statements in example main call
- v0.0.14: update vocab embedding loading to be compatible with other embedding files
- v0.0.13: update vocab, label_encoder, fix bugs in cnn reshaping and crf importing
- v0.0.12: fix crf module import issue
- v0.0.11: fix templates data resources
- v0.0.10: update `utils.config` module, `StaticEmbedding` decoupling, remove eps in metrics, add templates generation entrypoints, add more tests (coverage stat for the whole repo, lots of codes are not test case covered)
- v0.0.9: use `argparse` instead of `click`, move loguru logger into `rex.utils.logging`, add hierarchical config setting
- v0.0.8: fix config loading, change default of `makedirs` and `dump_configfile` to be `True`
- v0.0.7: fix recursive import bug
- v0.0.6: integrate omega conf loading into the inner task, add `load_*_data` option to data managers
- v0.0.5: update ffn
- v0.0.4: return detailed classification information in `mc_prf1`, support nested dict tensor movement
- v0.0.3: fix packaging bug in `setup.py`
- v0.0.2: add black formatter and pytest testing
- v0.0.1: change `LabelEncoder.to_binary_labels` into `convert_to_multi_hot` or `convert_to_one_hot`


## 🔑LICENCE
MIT
