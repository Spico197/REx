import os

import setuptools

from rex.version import __version__

readme_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
with open(readme_filepath, "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="pytorch-rex",
    version=__version__,
    author="Tong Zhu",
    author_email="tzhu1997@outlook.com",
    description="A toolkit for Relation Extraction and more...",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/Spico197/REx",
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.21.3",
        "omegaconf>=2.0.6",
        "loguru>=0.5.3",
        "tqdm>=4.61.1",
        "accelerate>=0.9.0",
        "transformers>=4.0.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "coverage",
            "black",
            "isort",
            "flake8",
            "pre-commit",
            "sphinx_rtd_theme",
            "sphinx",
        ]
    },
    include_package_data=True,
    entry_points={"console_scripts": ["rex = rex.cmds:main"]},
)
