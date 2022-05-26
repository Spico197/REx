import torch.optim as optim

from rex.data.data_manager import DataManager
from rex.data.dataset import CachedDataset
from rex.metrics.tagging import tagging_prf1
from rex.tasks.simple_task import SimpleTask
from rex.utils.io import load_jsonlines
from rex.utils.registry import register
from rex.utils.tagging import get_entities_from_tag_seq

from .model import LSTMCRFModel
from .transform import CachedTaggingTransform


@register("task")
class TaggingTask(SimpleTask):
    def __init__(self, config) -> None:
        super().__init__(config)

    def init_transform(self):
        return CachedTaggingTransform(
            self.config.max_seq_len,
            self.config.label2id_filepath,
            self.config.plm_dir,
        )

    def init_data_manager(self):
        return DataManager(
            self.config.train_filepath,
            self.config.dev_filepath,
            self.config.test_filepath,
            CachedDataset,
            self.transform,
            load_jsonlines,
            self.config.train_batch_size,
            self.config.eval_batch_size,
            self.transform.collate_fn,
            use_stream_transform=False,
            debug_mode=self.config.debug_mode,
            dump_cache_dir=self.config.data_cache_dir,
        )

    def init_model(self):
        m = LSTMCRFModel(
            self.config.plm_dir,
            num_lstm_layers=self.config.num_lstm_layers,
            num_tags=self.transform.label_encoder.num_tags,
            dropout=self.config.dropout,
        )
        return m

    def init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def get_eval_results(self, input_batches: list, output_batches: list) -> dict:
        preds = []
        golds = []
        for batch_input, batch_output in zip(input_batches, output_batches):
            batch_tokens = batch_input["tokens"]
            batch_gold = batch_input["ner_tags"]
            batch_pred = batch_output["pred"]
            for tokens, ins_gold, ins_pred in zip(batch_tokens, batch_gold, batch_pred):
                ins_gold = ins_gold[: len(tokens)]
                ins_pred = ins_pred[: len(tokens)]
                ins_gold_ents = get_entities_from_tag_seq(tokens, ins_gold)
                ins_pred = self.transform.label_encoder.decode(ins_pred)
                ins_pred_ents = get_entities_from_tag_seq(tokens, ins_pred)
                golds.append(ins_gold_ents)
                preds.append(ins_pred_ents)
        measures = tagging_prf1(golds, preds)
        return measures

    def predict_api(self, *texts):
        raw_dataset = [self.transform.predict_transform(text) for text in texts]
        loader = self.data_manager.prepare_loader(raw_dataset)
        results = []
        for batch in loader:
            batch_out = self.model(**batch)
            for tokens, ins_pred in zip(batch["tokens"], batch_out["pred"]):
                ins_pred = ins_pred[: len(tokens)]
                ins_pred = self.transform.label_encoder.decode(ins_pred)
                ins_pred_ents = get_entities_from_tag_seq(tokens, ins_pred)
                results.append(ins_pred_ents)
        return results
