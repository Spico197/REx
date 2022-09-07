from rex.tasks.relation_extraction import (
    MCBagRelationClassificationTask,
    MCMLSentRelationClassificationTask,
)
from rex.utils.config import ConfigParser


def test_bag_re_task():
    cmd_args = ["-dc", "conf/re/bag_nyt10.yaml"]
    config = ConfigParser.parse_cmd(cmd_args=cmd_args)
    config.debug_mode = True
    config.num_epochs = 1
    task = MCBagRelationClassificationTask.from_config(config)
    task.train()


def test_sent_re_task():
    cmd_args = ["-dc", "conf/re/sent_ipre.yaml"]
    config = ConfigParser.parse_cmd(cmd_args=cmd_args)
    config.debug_mode = True
    config.num_epochs = 1
    task = MCMLSentRelationClassificationTask.from_config(config)
    task.train()
