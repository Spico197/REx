from rex.tasks.relation_extraction import MCMLSentRelationClassificationTask
from rex.utils.config import get_config_from_cmd
from rex.utils.logging import logger


def main():
    config = get_config_from_cmd()
    task = MCMLSentRelationClassificationTask.from_config(config)
    logger.info(f"task: {type(task)}")

    if not task.config.skip_train:
        try:
            logger.info("Start Training")
            task.train()
        except Exception as err:
            logger.exception(err)

    task.load_best_ckpt()
    preds = task.predict("佟湘玉是莫小贝的嫂子。", "佟湘玉", "莫小贝")
    logger.info("case: 佟湘玉是莫小贝的嫂子。, head: 佟湘玉, tail: 莫小贝")
    logger.info(f"pred results: {preds}")


if __name__ == "__main__":
    main()
