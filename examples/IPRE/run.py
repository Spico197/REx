import os

import hydra
from omegaconf import DictConfig
from loguru import logger

from rex.tasks.relation_extraction import MCMLSentRelationClassificationTask
from rex.utils.initialization import set_seed_and_log_path


@hydra.main()
def main(config: DictConfig):
    set_seed_and_log_path(config.random_seed, os.path.join(config.task_dir, "log.log"))
    logger.info(config)
    task = MCMLSentRelationClassificationTask(config)
    logger.info(f"task: {type(task)}")

    if not config.skip_train:
        try:
            logger.info("Start Training")
            task.train()
        except Exception as err:
            logger.exception(err)

    task.load("/data4/tzhu/REx/examples/IPRE/outputs/re_sent_IPRE/ckpt/SentPCNN.best.pth")
    preds = task.predict("佟湘玉是莫小贝的嫂子。", "佟湘玉", "莫小贝")
    logger.info("case: 佟湘玉是莫小贝的嫂子。, head: 佟湘玉, tail: 莫小贝")
    logger.info(f"pred results: {preds}")


if __name__ == "__main__":
    main()
