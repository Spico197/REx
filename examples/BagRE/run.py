import click
from loguru import logger

from rex.tasks.relation_extraction import MCMLBagRelationClassificationTask


CONFIG_PATH_TYPE = click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True
)


@click.command()
@click.option(
    "-c", "--config-filepath", type=CONFIG_PATH_TYPE, help="configuration filepath"
)
def main(config_filepath):
    task = MCMLBagRelationClassificationTask.from_configfile(config_filepath)
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
