import torch
from torch.optim import Adam
from loguru import logger
from omegaconf import OmegaConf

from rex.models.casrel import CasRel
from rex.utils.io import load_line_json, load_json
from rex.utils.progress_bar import tqdm
from rex.utils.tensor_move import move_to_cuda_device
from rex.data.collate_fn import subj_obj_span_collate_fn
from rex.data.transforms.entity_re import (
    StreamBERTSubjObjSpanTransform,
    StreamSubjObjSpanTransform,
)
from rex.data.manager import StreamTransformManager
from rex.data.dataset import StreamTransformDataset
from rex.tasks.base_task import TaskBase
from rex.metrics.triple import measure_triple


class EntityRelationExtractionTask(TaskBase):
    def __init__(self, config: OmegaConf, **kwargs) -> None:
        super().__init__(config, **kwargs)

        rel2id = load_json(config.rel2id_filepath)
        self.transform = StreamBERTSubjObjSpanTransform(
            config.max_seq_len, rel2id, config.bert_model_dir
        )
        self.data_manager = StreamTransformManager(
            config.train_filepath,
            config.dev_filepath,
            config.test_filepath,
            StreamTransformDataset,
            self.transform,
            load_line_json,
            config.train_batch_size,
            config.eval_batch_size,
            subj_obj_span_collate_fn,
            debug_mode=config.debug_mode,
            load_train_data=config.load_train_data,
            load_dev_data=config.load_dev_data,
            load_test_data=config.load_test_data,
        )

        self.model = CasRel(
            config.bert_model_dir,
            self.transform.label_encoder.num_tags,
            pred_threshold=config.pred_threshold,
        )
        self.model.to(self.config.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)

    def train(self):
        for epoch_idx in range(self.config.num_epochs):
            logger.info(f"Epoch: {epoch_idx}/{self.config.num_epochs}")
            self.model.train()
            loader = tqdm(self.data_manager.train_loader, desc=f"Train(e{epoch_idx})")
            for batch in loader:
                batch = move_to_cuda_device(batch, self.config.device)
                used_keys = {
                    "token_ids",
                    "mask",
                    "subj_heads",
                    "subj_tails",
                    "subj_head",
                    "subj_tail",
                    "obj_head",
                    "obj_tail",
                }
                new_batch = {}
                for key, val in batch.items():
                    if key in used_keys:
                        new_batch[key] = val
                result = self.model(**new_batch)
                loss = result["loss"]
                loader.set_postfix({"loss": loss.item()})
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            logger.info(loader)

            dev_measures = self.eval("dev")
            self.history["dev"].append(dev_measures)
            test_measures = self.eval("test")
            self.history["test"].append(test_measures)

            measures = dev_measures
            is_best = False
            if measures["f1"] > self.best_metric:
                is_best = True
                self.best_metric = measures["f1"]
                self.best_epoch = epoch_idx
                self.no_climbing_cnt = 0
            else:
                self.no_climbing_cnt += 1

            if is_best and self.config.save_best_ckpt:
                self.save_ckpt(epoch=epoch_idx)

            logger.info(
                f"Epoch: {epoch_idx}, is_best: {is_best}, Dev: {measures}, Test: {test_measures}"
            )

            if (
                self.config.num_early_stop > 0
                and self.no_climbing_cnt > self.config.num_early_stop
            ):
                break

        logger.info(
            (
                f"Trial finished. Best Epoch: {self.best_epoch}, Dev: {self.history['dev'][self.best_epoch]}, "
                f"Best Test: {self.history['test'][self.best_epoch]}"
            )
        )

    @torch.no_grad()
    def eval(self, dataset_name):
        self.model.eval()
        name2loader = {
            "train": self.data_manager.train_eval_loader,
            "dev": self.data_manager.dev_loader,
            "test": self.data_manager.test_loader,
        }
        loader = tqdm(name2loader[dataset_name], desc=f"{dataset_name} Eval")
        preds = []
        golds = []
        for batch in loader:
            batch = move_to_cuda_device(batch, self.config.device)
            golds.extend(batch["triples"])
            used_keys = {"token_ids", "mask"}
            new_batch = {}
            for key, val in batch.items():
                if key in used_keys:
                    new_batch[key] = val
            result = self.model(**new_batch)
            pred_triples = result["pred"]
            preds.extend(pred_triples)
        logger.info(loader)
        measures = measure_triple(preds, golds)
        return measures["triple"]

    @torch.no_grad()
    def predict(self, text):
        self.model.eval()
        batch = self.transform.predict_transform(
            {
                "text": text,
            }
        )
        tensor_batch = self.data_manager.collate_fn([batch])
        tensor_batch = move_to_cuda_device(tensor_batch, self.config.device)
        result = self.model(
            token_ids=tensor_batch["token_ids"], mask=tensor_batch["mask"]
        )
        pred_triples = result["pred"][0]
        preds = []
        for triple in pred_triples:
            head_ent = batch["token_ids"][triple[0][0] : triple[0][1]]
            head_ent = self.transform.vocab.convert_ids_to_tokens(head_ent)
            head_ent = " ".join(head_ent)
            tail_ent = batch["token_ids"][triple[2][0] : triple[2][1]]
            tail_ent = self.transform.vocab.convert_ids_to_tokens(tail_ent)
            tail_ent = " ".join(tail_ent)
            relation = self.transform.label_encoder.id2label[triple[1]]
            preds.append((head_ent, relation, tail_ent))

        return preds
