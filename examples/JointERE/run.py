import click
from loguru import logger

from rex.tasks.entity_relation_extraction import EntityRelationExtractionTask


CONFIG_PATH_TYPE = click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True
)


@click.command()
@click.option(
    "-c", "--config-filepath", type=CONFIG_PATH_TYPE, help="configuration filepath"
)
def main(config_filepath):
    task = EntityRelationExtractionTask.from_configfile(config_filepath)
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
