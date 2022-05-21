from rex.tasks.relation_extraction import MCMLBagRelationClassificationTask
from rex.utils.config import get_config_from_cmd
from rex.utils.logging import logger


def main():
    config = get_config_from_cmd()
    task = MCMLBagRelationClassificationTask.from_config(config)
    logger.info(f"task: {type(task)}")

    if not task.config.skip_train:
        try:
            logger.info("Start Training")
            task.train()
        except Exception as err:
            logger.exception(err)

    task.load_best_ckpt()
    preds = task.predict("John was born in China .", "John", "China")
    logger.info("Case: John was born in China. , head: John, tail: China")
    logger.info(f"pred results: {preds}")


if __name__ == "__main__":
    main()
