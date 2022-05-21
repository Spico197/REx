from rex.tasks.base_task import TaskBase
from rex.utils.config import ConfigParser
from rex.utils.logging import logger
from rex.utils.registry import get_registered, register


@register("rex_init_call")
def train(cmd_args=None):
    config = ConfigParser.parse_cmd(cmd_args=cmd_args)
    if "task_type" not in config:
        raise ValueError(
            (
                "`task_type` must exist in config! "
                "If you are using the default base config, `task_type` is not initialised in it."
            )
        )
    TASK_CLASS = get_registered("task", config.task_type)
    task: TaskBase = TASK_CLASS(config)
    logger.info("Task initialised, ready to start")
    task.train()


if __name__ == "__main__":
    train()
