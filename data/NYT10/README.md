# NYT10
- URL: https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_nyt10.sh
- Paper: Modeling Relations and Their Mentions without Labeled Text

## Download and convert to json format

1. make sure you've `cd` into this directory
2. change input embedding filepath in `convert.py`
3. run command below

```bash
source download_and_convert.sh
```

And you'll able to find the converted json file in `formatted` folder.

## References

```bibtex
@InProceedings{nyt10,
    author="Riedel, Sebastian
    and Yao, Limin
    and McCallum, Andrew",
    editor="Balc{\'a}zar, Jos{\'e} Luis
    and Bonchi, Francesco
    and Gionis, Aristides
    and Sebag, Mich{\`e}le",
    title="Modeling Relations and Their Mentions without Labeled Text",
    booktitle="Machine Learning and Knowledge Discovery in Databases",
    year="2010",
    publisher="Springer Berlin Heidelberg",
    address="Berlin, Heidelberg",
    pages="148--163",
    isbn="978-3-642-15939-8"
}
```
