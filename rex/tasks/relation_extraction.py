from torch.optim import Adam

from rex.data.collate_fn import re_collate_fn
from rex.data.data_manager import DataManager
from rex.data.dataset import CachedDataset
from rex.data.transforms.sent_re import CachedMCMLSentRETransform
from rex.metrics.classification import mcml_prf1
from rex.models.sent_pcnn import SentPCNN
from rex.tasks.simple_task import SimpleTask
from rex.utils.dict import PrettyPrintDict
from rex.utils.io import load_jsonlines
from rex.utils.registry import register
from rex.utils.tensor_move import detach_cpu_list


@register("task")
class MCMLSentRelationClassificationTask(SimpleTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_transform(self):
        return CachedMCMLSentRETransform(
            self.config.max_seq_len,
            self.config.rel2id_filepath,
            self.config.emb_filepath,
        )

    def init_data_manager(self):
        dm = DataManager(
            self.config.train_filepath,
            self.config.dev_filepath,
            self.config.test_filepath,
            CachedDataset,
            self.transform,
            load_jsonlines,
            self.config.train_batch_size,
            self.config.eval_batch_size,
            re_collate_fn,
            use_stream_transform=False,
            debug_mode=self.config.debug_mode,
            dump_cache_dir=self.config.data_cache_dir,
        )
        return dm

    def init_model(self):
        m = SentPCNN(
            self.transform.label_encoder.num_tags,
            self.transform.vocab.size,
            self.config.dim_token_emb,
            self.config.max_seq_len,
            self.config.dim_pos_emb,
            self.config.num_filters,
            self.config.kernel_size,
            self.config.dropout,
        )
        m.token_embedding.from_pretrained(self.config.emb_filepath)
        return m

    def init_optimizer(self):
        return Adam(self.model.parameters(), lr=self.config.learning_rate)

    def get_eval_results(self, input_batches: list, output_batches: list) -> dict:
        preds = []
        golds = []
        for batch_input, batch_output in zip(input_batches, output_batches):
            batch_gold = detach_cpu_list(batch_input["labels"])
            batch_pred = detach_cpu_list(
                batch_output["pred"].ge(self.config.pred_threshold).long()
            )
            golds.extend(batch_gold)
            preds.extend(batch_pred)
        measures = mcml_prf1(preds, golds)
        return PrettyPrintDict(measures)

    def predict_api(self, text, head, tail):
        self.model.eval()
        batch = self.transform.predict_transform(
            {"text": text, "head": head, "tail": tail}
        )
        tensor_batch = self.data_manager.collate_fn([batch])
        if "labels" in tensor_batch:
            del tensor_batch["labels"]
        if "id" in tensor_batch:
            del tensor_batch["id"]
        result = self.model(**tensor_batch)
        outs = result["pred"]
        outs = detach_cpu_list(outs.ge(self.config.pred_threshold).long())[0]
        preds = []
        for type_idx, pred in enumerate(outs):
            if pred == 1:
                preds.append(self.transform.label_encoder.id2label[type_idx])

        return preds
