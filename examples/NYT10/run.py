import click
from omegaconf import OmegaConf
from loguru import logger

from rex.utils.initialization import init_all
from rex.tasks.relation_extraction import MCMLBagRelationClassificationTask


CONFIG_PATH_TYPE = click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)


@click.command()
@click.option("-c", "--config-filepath", type=CONFIG_PATH_TYPE, help='configuration filepath')
def main(config_filepath):
    config = OmegaConf.load(config_filepath)
    init_all(config.task_dir, config.random_seed, True, config)
    logger.info(config)
    task = MCMLBagRelationClassificationTask(config)
    logger.info(f"task: {type(task)}")

    if not config.skip_train:
        try:
            logger.info("Start Training")
            task.train()
        except Exception as err:
            logger.exception(err)

    task.load("/data4/tzhu/REx/examples/NYT10/outputs/re_bag_NYT10/ckpt/PCNNOne.best.pth")
    preds = task.predict("John was born in China .", "John", "China")
    logger.info("Case: John was born in China. , head: John, tail: China")
    logger.info(f"pred results: {preds}")


if __name__ == "__main__":
    main()
