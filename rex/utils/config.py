import os
import re
import sys
import yaml
import json
import argparse
import random
import logging
import datetime
from typing import Optional

import torch
import numpy as np


def set_seed(seed: Optional[int] = 0):
    """
    set random seed

    Args:
        seed (int, optional): random seed. Defaults to 0.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ArgConfig(object):
    def __init__(self, **kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str,
                            help="YAML config filepath")
        parser.add_argument("--local_rank", required=False,
                            default=-1, type=int,
                            help="remained for distributed"
                                 " data parallel support")
        args = parser.parse_args()
        self.local_rank = args.local_rank
        if args.config:
            params = YamlConfig(args.config, **kwargs)
            for key, val in params.__dict__.items():
                setattr(self, key, val)


class YamlConfig(object):
    def __init__(self, config_filepath: str, **kwargs):
        self.config_abs_path = os.path.abspath(config_filepath)
        self.start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # do not use `with` expression here in order to find errors ealier
        fin = open(self.config_abs_path, 'rt', encoding='utf-8')
        __config = yaml.safe_load(fin)
        fin.close()

        for key, val in __config.items():
            setattr(self, key, val)
        
        set_seed(self.random_seed)

        last_dir = os.path.split(self.output_dir)[-1]
        overwrite = False
        if re.match(r'\d{8}_\d{6}', last_dir):
            self.start_time = last_dir
            overwrite = True
        if not overwrite:
            self.output_dir = os.path.join(self.output_dir, self.start_time)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        with open(os.path.join(self.output_dir, "config.yml"),
                  'wt', encoding='utf-8') as fout:
            yaml.dump(self.__dict__, fout, indent=2)

    def __str__(self, indent=0):
        return json.dumps(self.__dict__, indent=indent)
