from pathlib import Path

from omegaconf import OmegaConf
import torch
import torch.optim as optim
from rex.utils.config import ConfigParser
from rex.utils.logging import logger
from rex.utils.io import dump_json, load_line_json
from rex.data.dataset import CachedDataset
from rex.data.manager import CachedManager
from rex.utils.tensor_move import move_to_cuda_device
from rex.utils.progress_bar import tqdm
from rex.tasks.base_task import TaskBase
from rex.utils.initialization import init_all
from rex.utils.tagging import get_entities_from_tag_seq
from rex.metrics.tagging import tagging_prf1

from .data import CachedTaggingTransform, tagging_collate_fn
from .model import LSTMCRFModel


class NERTask(TaskBase):
    def __init__(self, config):
        super().__init__(config)

        self.middle_path = Path(config.task_dir).joinpath("middle")
        self.middle_path.mkdir(parents=True, exist_ok=True)

        self.transform = CachedTaggingTransform(config.max_seq_len)
        self.data_manager = CachedManager(
            config.train_filepath,
            config.dev_filepath,
            config.test_filepath,
            CachedDataset,
            self.transform,
            load_line_json,
            config.train_batch_size,
            config.eval_batch_size,
            tagging_collate_fn,
            debug_mode=config.debug_mode,
            load_train=not config.skip_train,
            load_dev=not config.skip_train,
            load_test=not config.skip_final_eval,
        )

        self.model = LSTMCRFModel(
            self.transform.vocab.size,
            config.emb_size,
            config.hidden_size,
            num_lstm_layers=config.num_lstm_layers,
            num_tags=self.transform.label_encoder.num_tags,
            dropout=config.dropout,
        )
        self.model = self.model.to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def eval(self, dataset_name, verbose=False, postfix=""):
        self.model.eval()
        name2loader = {
            "train": self.data_manager.train_eval_loader,
            "dev": self.data_manager.dev_loader,
            "test": self.data_manager.test_loader,
        }
        loader = tqdm(
            name2loader[dataset_name], desc=f"{dataset_name} Eval", ncols=80, ascii=True
        )
        seqs = []
        tags = []
        preds = []
        for batch in loader:
            tags.extend(batch["tags"].detach().cpu().tolist())
            del batch["tags"]
            seqs.extend(batch["inputs"].detach().cpu().tolist())
            out = self.model(**batch)
            preds.extend(out)

        gold_ents_all = []
        pred_ents_all = []
        # num_illegal_tag = 0
        dump_data = []
        for seq, tag, pred in zip(seqs, tags, preds):
            seq = self.transform.vocab.convert_ids_to_tokens(seq)
            tag = self.transform.label_encoder.decode(tag)
            pred = self.transform.label_encoder.decode(pred)
            # num_illegal_tag += get_num_illegal_tags_from_tag_seq(pred)
            gold_ents = get_entities_from_tag_seq(seq, tag)
            pred_ents = get_entities_from_tag_seq(seq, pred)
            gold_ents_all.append(gold_ents)
            pred_ents_all.append(pred_ents)
            dump_data.append(
                {
                    "seq": seq,
                    "gold_tags": tag,
                    "pred_tags": pred,
                    "gold_ents": gold_ents,
                    "pred_ents": pred_ents,
                }
            )

        measures = tagging_prf1(gold_ents_all, pred_ents_all)

        if verbose:
            logger.info(f"{dataset_name} Eval Measures: {measures}")

        dump_json(
            dump_data, self.middle_path.joinpath(f"{dataset_name}.{postfix}.json")
        )

        return measures["micro"]["f1"]

    @torch.no_grad()
    def predict(self, string, verbose=False):
        self.model.eval()
        batch = self.transform.predict_transform(string)
        tensor_batch = self.data_manager.collate_fn([batch])
        if "labels" in tensor_batch:
            del tensor_batch["labels"]
        outs = self.model(**tensor_batch)
        pred_ents_per_sent = []
        pred_tag = outs["preds"][0]
        # in case of OOV
        refreshed_pred_ents = []
        pred_tag = self.transform.label_encoder.decode(pred_tag)
        pred_ents = get_entities_from_tag_seq(string, pred_tag)
        for ent_type in pred_ents:
            for ent in pred_ents[ent_type]:
                refreshed_pred_ents.append(
                    (string[ent[2] : ent[3]], ent[1], ent[2], ent[3])
                )
        pred_ents_per_sent.append(refreshed_pred_ents)

        if verbose:
            logger.info(f"Pred ents: {pred_ents_per_sent}")

        return pred_ents_per_sent

    def print_final_record(self):
        logger.info(
            (
                f"Best Epoch: {self.best_epoch}, Dev: {self.history['dev'][self.best_epoch]}, "
                f"Best Test: {self.history['test'][self.best_epoch]}"
            )
        )

    def train(self):
        for epoch_idx in range(self.config.num_epochs):
            logger.info(f"Epoch: {epoch_idx}/{self.config.num_epochs}")
            self.model.train()
            loader = tqdm(self.data_manager.train_loader, desc=f"Train(e{epoch_idx})")
            for batch in loader:
                batch = move_to_cuda_device(batch, self.config.device)
                result = self.model(**batch)
                loss = result["loss"]
                loader.set_postfix({"loss": loss.item()})
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            logger.info(loader)

            train_measures = self.eval("train", verbose=True, postfix=f"{epoch_idx}")
            self.history["train"].append(train_measures)
            dev_measures = self.eval("dev", verbose=True, postfix=f"{epoch_idx}")
            self.history["dev"].append(dev_measures)
            test_measures = self.eval("test", verbose=True, postfix=f"{epoch_idx}")
            self.history["test"].append(test_measures)

            measures = dev_measures
            is_best = False
            if measures > self.best_metric:
                is_best = True
                self.best_metric = measures
                self.best_epoch = epoch_idx
                self.no_climbing_cnt = 0
            else:
                self.no_climbing_cnt += 1

            if is_best and self.config.save_best_ckpt:
                self.save_ckpt(f"{epoch_idx}.{100 * dev_measures:.3f}", epoch_idx)
                self.save_ckpt("best", epoch_idx)

            logger.info(
                (
                    f"Epoch: {epoch_idx}, is_best: {is_best}, "
                    f"Train: {100 * train_measures:.3f}, "
                    f"Dev: {100 * dev_measures:.3f}, "
                    f"Test: {100 * test_measures:.3f}"
                )
            )

            if (
                self.config.num_early_stop > 0
                and self.no_climbing_cnt > self.config.num_early_stop
            ):
                break

        logger.info("Trial finished.")
        self.print_final_record()


def main():
    config = ConfigParser.parse_cmd()
    init_all(config.task_dir, config.random_seed, True, config)
    logger.info(OmegaConf.to_object(config))
    task = NERTask(config)
    logger.info(f"task: {type(task)}")

    if not config.skip_train:
        try:
            logger.info("Start Training")
            task.train()
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as err:
            logger.exception(err)
        finally:
            task.print_final_record()

    if not config.skip_final_eval:
        task.load(config.final_eval_model_filepath)
        task.eval("test", verbose=True)

    if not config.skip_predict_example:
        task.load(config.final_eval_model_filepath)
        text = "this is a sentence."
        task.predict(text, verbose=True)


if __name__ == "__main__":
    main()
