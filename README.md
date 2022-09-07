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

A toolkit for Relation extraction, Event eXtraction (REx) and more...

This project has not been finished yet, so be careful when using it, or wait until the first release comes out.

This project is suffering from the second-system effect.
I would like to cut some features to make this going smoothly.

[Accelerate](https://github.com/huggingface/accelerate) seems to be a very sweet wrapper for multi-GPU, TPU training, we highly recommend you to use such frameworks, instead of adding hard codes on your own.

## ⚙️Installation

[Python](https://www.python.org/) (**>=3.7**) and [pytorch](http://pytorch.org/) (**>=1.7.1**) are preliminaries for REx.

```bash
# 1. install from source
$ git clone https://github.com/Spico197/REx.git
$ cd REx
$ pip install -e .

# 2. install from PyPI, recommended
$ pip install pytorch-rex -i https://pypi.org/simple
```

## 🚀QuickStart

- Create a new task: `rex new task_name`

## 🤝 Contributing

Contributions are greatly appreciated to make this project better.

Please see [Contributing](./CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for more information.

## 🔑LICENCE

Distributed under the MIT licence.

See `LICENSE` for more informatin.
