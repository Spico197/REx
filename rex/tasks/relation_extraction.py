import torch
from torch.optim import Adam
from loguru import logger
from omegaconf import OmegaConf

from rex.models.sent_pcnn import SentPCNN
from rex.models.bag_pcnn import PCNNOne
from rex.utils.io import load_line_json
from rex.utils.progress_bar import tqdm
from rex.utils.tensor_move import move_to_cuda_device
from rex.data.collate_fn import bag_re_collate_fn, re_collate_fn
from rex.data.transforms.sent_re import CachedMCMLSentRETransform
from rex.data.transforms.bag_re import CachedMCBagRETransform
from rex.data.manager import CachedManager
from rex.data.dataset import CachedBagREDataset, CachedDataset
from rex.tasks.base_task import TaskBase
from rex.metrics.classification import accuracy, mc_prf1, mcml_prf1


class MCMLSentRelationClassificationTask(TaskBase):
    def __init__(self, config: OmegaConf, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.transform = CachedMCMLSentRETransform(config.max_seq_len)
        self.data_manager = CachedManager(
            config.train_filepath,
            config.dev_filepath,
            config.test_filepath,
            CachedDataset,
            self.transform,
            load_line_json,
            config.train_batch_size,
            config.eval_batch_size,
            re_collate_fn,
            debug_mode=config.debug_mode,
            load_train_data=config.load_train_data,
            load_dev_data=config.load_dev_data,
            load_test_data=config.load_test_data,
        )

        self.model = SentPCNN(
            self.transform.vocab,
            config.emb_filepath,
            self.transform.label_encoder.num_tags,
            config.dim_token_emb,
            config.max_seq_len,
            config.dim_pos_emb,
            config.num_filters,
            config.kernel_size,
            dropout=config.dropout,
        )
        self.model.to(self.config.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)

    def train(self):
        for epoch_idx in range(self.config.num_epochs):
            logger.info(f"Epoch: {epoch_idx}/{self.config.num_epochs}")
            self.model.train()
            loader = tqdm(self.data_manager.train_loader, desc=f"Train(e{epoch_idx})")
            for batch in loader:
                del batch["id"]
                batch = move_to_cuda_device(batch, self.config.device)
                result = self.model(**batch)
                loss = result["loss"]
                loader.set_postfix({"loss": loss.item()})
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            logger.info(loader)

            measures = self.eval("dev")
            self.history["dev"].append(measures)
            test_measures = self.eval("test")
            self.history["test"].append(test_measures)

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
            "train": self.data_manager.train_loader,
            "dev": self.data_manager.dev_loader,
            "test": self.data_manager.test_loader,
        }
        loader = tqdm(name2loader[dataset_name], desc=f"{dataset_name} Eval")
        preds = []
        golds = []
        for batch in loader:
            batch = move_to_cuda_device(batch, self.config.device)
            golds.extend(batch["labels"].detach().cpu().tolist())
            del batch["id"]
            del batch["labels"]
            result = self.model(**batch)
            out = result["pred"]
            preds.extend(
                out.ge(self.config.pred_threshold).long().detach().cpu().tolist()
            )
        logger.info(loader)
        measures = mcml_prf1(preds, golds)
        return measures["macro"]

    @torch.no_grad()
    def predict(self, text, head, tail):
        self.model.eval()
        batch = self.transform.predict_transform(
            {"text": text, "head": head, "tail": tail}
        )
        tensor_batch = self.data_manager.collate_fn([batch])
        if "labels" in tensor_batch:
            del tensor_batch["labels"]
        if "id" in tensor_batch:
            del tensor_batch["id"]
        tensor_batch = move_to_cuda_device(tensor_batch, self.config.device)
        result = self.model(**tensor_batch)
        outs = result["pred"]
        outs = outs.ge(self.config.pred_threshold).long().detach().cpu().tolist()[0]
        preds = []
        for type_idx, pred in enumerate(outs):
            if pred == 1:
                preds.append(self.transform.label_encoder.id2label[type_idx])

        return preds


class MCMLBagRelationClassificationTask(TaskBase):
    def __init__(self, config: OmegaConf, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.transform = CachedMCBagRETransform(config.max_seq_len)
        self.data_manager = CachedManager(
            config.train_filepath,
            config.dev_filepath,
            config.test_filepath,
            CachedBagREDataset,
            self.transform,
            load_line_json,
            config.train_batch_size,
            config.eval_batch_size,
            bag_re_collate_fn,
            debug_mode=config.debug_mode,
            load_train_data=config.load_train_data,
            load_dev_data=config.load_dev_data,
            load_test_data=config.load_test_data,
        )

        self.model = PCNNOne(
            self.transform.vocab,
            config.emb_filepath,
            self.transform.label_encoder.num_tags,
            config.dim_token_emb,
            config.max_seq_len,
            config.dim_pos_emb,
            config.num_filters,
            config.kernel_size,
            dropout=config.dropout,
        )
        self.model.to(self.config.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)

        self.history = {"dev": [], "test": []}
        self.no_climbing_cnt = 0
        self.best_metric = -100.0
        self.best_epoch = -1

    def train(self):
        for epoch_idx in range(self.config.num_epochs):
            logger.info(f"Epoch: {epoch_idx}/{self.config.num_epochs}")
            self.model.train()
            loader = tqdm(self.data_manager.train_loader, desc=f"Train(e{epoch_idx})")
            for batch in loader:
                del batch["id"]
                batch = move_to_cuda_device(batch, self.config.device)
                result = self.model(**batch)
                loss = result["loss"]
                loader.set_postfix({"loss": loss.item()})
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            logger.info(loader)

            measures = self.eval("dev")
            self.history["dev"].append(measures)
            test_measures = self.eval("test")
            self.history["test"].append(test_measures)

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
            "train": self.data_manager.train_loader,
            "dev": self.data_manager.dev_loader,
            "test": self.data_manager.test_loader,
        }
        loader = tqdm(name2loader[dataset_name], desc=f"{dataset_name} Eval")
        preds = []
        golds = []
        for batch in loader:
            batch = move_to_cuda_device(batch, self.config.device)
            golds.extend(batch["labels"].detach().cpu().tolist())
            del batch["id"]
            del batch["labels"]
            result = self.model(**batch)
            preds.extend(result["pred"].long().detach().cpu().tolist())
        logger.info(loader)
        measures = mc_prf1(
            preds,
            golds,
            self.transform.label_encoder.num_tags,
            ignore_labels=[self.transform.label_encoder.label2id["NA"]],
        )
        return measures["macro"]

    @torch.no_grad()
    def predict(self, text, head, tail):
        self.model.eval()
        batch = self.transform.predict_transform(
            {"text": text, "head": head, "tail": tail}
        )
        tensor_batch = self.data_manager.collate_fn([batch])
        if "labels" in tensor_batch:
            del tensor_batch["labels"]
        if "id" in tensor_batch:
            del tensor_batch["id"]
        tensor_batch = move_to_cuda_device(tensor_batch, self.config.device)
        result = self.model(**tensor_batch)
        outs = result["pred"]
        preds = []
        for pred in outs:
            preds.append(self.transform.label_encoder.id2label[pred])

        return preds
