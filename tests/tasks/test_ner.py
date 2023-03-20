import os

import pytest

from rex.tasks.named_entity_recognition import MrcTaggingTask
from rex.utils.config import ConfigParser


@pytest.mark.skipif(
    not os.path.exists("data/peoples_daily_ner/data/formatted/test.jsonl"),
    reason="data not found",
)
def test_mrc_ner_task():
    cmd_args = ["-dc", "conf/ner/mrc_ner.yaml"]
    config = ConfigParser.parse_cmd(cmd_args=cmd_args)
    config.debug_mode = True
    config.num_epochs = 1
    task = MrcTaggingTask.from_config(config)
    task.train()
