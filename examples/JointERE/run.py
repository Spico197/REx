from rex.utils.logging import logger
from rex.utils.config import get_config_from_cmd
from rex.tasks.entity_relation_extraction import EntityRelationExtractionTask


def main():
    config = get_config_from_cmd()
    task = EntityRelationExtractionTask.from_config(config)
    logger.info(f"task: {type(task)}")

    if not task.config.skip_train:
        try:
            logger.info("Start Training")
            task.train()
        except Exception as err:
            logger.exception(err)

    task.load_best_ckpt()
    preds = task.predict(
        "William Anders was born October 17th 1933 and was a member of the Apollo 8 crew operated by NASA . The backup pilot was Buzz Aldrin . William Anders retired on September 1st 1969 ."
    )
    logger.info(
        "Case: William Anders was born October 17th 1933 and was a member of the Apollo 8 crew operated by NASA . The backup pilot was Buzz Aldrin . William Anders retired on September 1st 1969 ."
    )
    logger.info(f"pred results: {preds}")


if __name__ == "__main__":
    main()
