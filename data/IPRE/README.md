# IPRE
- URL: https://github.com/SUDA-HLT/IPRE
- Paper: Haitao Wang, Zhengqiu He, Jin Ma, Wenliang Chen, Min Zhang. 2019. IPRE: a Dataset for Inter-Personal Relationship Extraction. https://arxiv.org/abs/1907.12801

## Download and convert to json format
```bash
mkdir raw && cd raw && git clone https://github.com/SUDA-HLT/IPRE.git
python convert.py
```

And you'll able to find the converted json file in `formatted` folder.
