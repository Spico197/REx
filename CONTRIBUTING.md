# Contributing

♥ Thank you very much for your time to contribute ~ ✨

Feel free to open an issue to discuss about the roadmap or make proposals.

## Code of Conduct

When participating in discussions and contribute code, you are expected to uphold this [code-of-conduct](./CODE_OF_CONDUCT.md).

If you don't have time to skim the boring instructions, that's fine.
But we are willing to build a kind and warm society, and you should **respect** the others in the community.

## Issues

Issues are welcomed to open an issue to help us improve the project ~

## Pull Requests

Pull requests are greatly appreciated to help us develop this project directly.
You could start by fixing a bug, or implementing a desired function in the [roadmap](https://github.com/Spico197/REx/projects/1).

*Before writing any code*, make sure there's a corresponding issue, or function in the project roadmap.

1. Fork the repo to your namespace
2. Install `pytorch-rex` with `[dev]` option

```bash
# from pypi
$ pip install pytorch-rex[dev]

# or from source
$ git clone https://github.com/Spico197/REx.git
$ cd REx
$ pip install -e .[dev]
```

3. Install pre-commit hooks

```bash
$ pre-commit install
```

4. Development
   - Please make your code clean and easy-to-read
   - Please add test cases to `tests/` to verify the function works fine
   - Please use formatters to formmat the code by simply `make format` in the shell
5. Contribute to **your remote repo**
6. Contribute to the REx repo by making pull requests
