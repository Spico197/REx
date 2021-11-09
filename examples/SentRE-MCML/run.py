import click
from omegaconf import OmegaConf
from loguru import logger

from rex.utils.initialization import init_all
from rex.tasks.relation_extraction import MCMLSentRelationClassificationTask


CONFIG_PATH_TYPE = click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True
)


@click.command()
@click.option(
    "-c", "--config-filepath", type=CONFIG_PATH_TYPE, help="configuration filepath"
)
def main(config_filepath):
    config = OmegaConf.load(config_filepath)
    init_all(config.task_dir, config.random_seed, True, config)
    logger.info(config)
    task = MCMLSentRelationClassificationTask(config)
    logger.info(f"task: {type(task)}")

    if not config.skip_train:
        try:
            logger.info("Start Training")
            task.train()
        except Exception as err:
            logger.exception(err)

    task.load(
        "/data4/tzhu/REx/examples/IPRE/outputs/re_sent_IPRE/ckpt/SentPCNN.best.pth"
    )
    preds = task.predict("佟湘玉是莫小贝的嫂子。", "佟湘玉", "莫小贝")
    logger.info("case: 佟湘玉是莫小贝的嫂子。, head: 佟湘玉, tail: 莫小贝")
    logger.info(f"pred results: {preds}")


if __name__ == "__main__":
    main()
