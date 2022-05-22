CONFIG_PARAMS_FILENAME = "task_params.yaml"
CHECKPOINT_DIRNAME = "ckpt"
CHECKPOINT_FILENAME_TEMPLATE = "{}.{}.pth"
BEST_IDENTIFIER = "best"
BEST_CHECKPOINT_FILENAME_TEMPLATE = "{}.best.pth"
LOG_FILENAME = "log.log"

from .relation_extraction import MCMLSentRelationClassificationTask
from .simple_task import SimpleTask
