import torch.nn as nn

from rex.models.global_pointer import BertTokenEncoder, GlobalPointer


class USM(nn.Module):
    def __init__(
        self, plm_dir: str, biaffine_size: int = 512, use_rope=True, dropout=0
    ) -> None:
        super().__init__()

        encoder = BertTokenEncoder(plm_dir)
        hs = encoder.hidden_size

        # head could be behind of tail
        # ttl: token - token: none: H2T, H2H, T2T
        self.ttl_pointer = GlobalPointer(
            encoder,
            3,
            hs,
            biaffine_size,
            use_rope=use_rope,
            dropout=dropout,
            tri_mask="none",
        )
        # ltl: label - token: tril: L2H, L2T
        self.ltl_pointer = GlobalPointer(
            encoder,
            2,
            hs,
            biaffine_size,
            use_rope=use_rope,
            dropout=dropout,
            tri_mask="tril",
        )
        # tll: token - label: triu: H2L, T2L
        self.tll_pointer = GlobalPointer(
            encoder,
            2,
            hs,
            biaffine_size,
            use_rope=use_rope,
            dropout=dropout,
            tri_mask="triu",
        )

    def forward(
        self,
        input_ids,
        mask,
        ttl_labels=None,
        ltl_labels=None,
        tll_labels=None,
        label_map=None,
        is_eval=None,
        **kwargs
    ):
        ttl_results = self.ttl_pointer(
            input_ids, mask, labels=ttl_labels, is_eval=is_eval, **kwargs
        )
        ltl_results = self.ltl_pointer(
            input_ids, mask, labels=ltl_labels, is_eval=is_eval, **kwargs
        )
        tll_results = self.tll_pointer(
            input_ids, mask, labels=tll_labels, is_eval=is_eval, **kwargs
        )
        results = [ttl_results, ltl_results, tll_results]

        ret = {}
        if all("loss" in r for r in results):
            ret["loss"] = sum(r["loss"] for r in results)
        if is_eval and all("pred" in r for r in results):
            ret["pred"] = self.decode(
                ttl_results["pred"],
                ltl_results["pred"],
                tll_results["pred"],
                label_map,
            )
        return ret

    def decode(
        self,
        ttl_pred: list[tuple],
        ltl_pred: list[tuple],
        tll_pred: list[tuple],
        label_map: list[dict[int, dict]],  # `m` or `p`,
    ):
        # pred: batch[[(h, t, type_index)], ...]
        # label_map: batch[{label index: {"type": "m"/"p", "string": "person"}, ...}, ...]
        batch_preds = []

        for ttl, ltl, tll, lm in zip(ttl_pred, ltl_pred, tll_pred, label_map):
            # token - token
            h2t = [(x[0], x[1]) for x in filter(lambda x: x[2] == 0, ttl)]
            hs = set(x[0] for x in h2t)
            ts = set(x[1] for x in h2t)
            h2h = [
                (x[0], x[1])
                for x in filter(lambda x: x[2] == 1 and x[0] in hs and x[1] in hs, ttl)
            ]
            t2t = [
                (x[0], x[1])
                for x in filter(lambda x: x[2] == 2 and x[0] in ts and x[1] in ts, ttl)
            ]
            pairs = []
            for hh, th in h2h:
                for ht, tt in t2t:
                    if (hh, ht) in h2t and (th, tt) in h2t:
                        pairs.append(((hh, ht), (th, tt)))

            # label - token
            l2h = [
                (x[0], x[1])
                for x in filter(lambda x: x[2] == 0 and x[0] in lm and x[1] in hs, ltl)
            ]
            l2t = [
                (x[0], x[1])
                for x in filter(lambda x: x[2] == 1 and x[0] in lm and x[1] in ts, ltl)
            ]
            l2ht = []
            for l1, h in l2h:
                for l2, t in l2t:
                    if l1 == l2 and (h, t) in h2t:
                        l2ht.append(((lm[l1]["string"], lm[l1]["type"], l1), (h, t)))
            ls = set(lb[2] for lb in l2ht)

            # token - label
            h2l = [
                (x[0], x[1])
                for x in filter(lambda x: x[2] == 0 and x[0] in hs and x[1] in ls, tll)
            ]
            t2l = [
                (x[0], x[1])
                for x in filter(lambda x: x[2] == 1 and x[0] in ts and x[1] in ls, tll)
            ]
            ht2l = []
            for h, l1 in h2l:
                for t, l2 in t2l:
                    if l1 == l2 and (h, t) in h2t and lm[l1]["type"] == "p":
                        ht2l.append(((h, t), (lm[l1]["string"], "p", l1)))

            # merge
            # ents: [((start, end + 1), "type"), ...]
            ents = [
                (ent[1], ent[0][0]) for ent in filter(lambda x: x[0][1] == "m", l2ht)
            ]
            ents = list(set(ents))
            # relations: [(head start, head end + 1), "relation", (tail start, tail end + 1)]
            relations = []
            for (hh, ht), (th, tt) in pairs:
                for l1, (h1, t1) in l2ht:
                    if l1[1] == "p":
                        for (h2, t2), l2 in ht2l:
                            if l2[1] == "p":
                                if (
                                    hh == h1
                                    and ht == t1
                                    and th == h2
                                    and tt == t2
                                    and l1[2] == l2[2]
                                ):
                                    relations.append(((hh, ht), l1[0], (th, tt)))
            relations = list(set(relations))
            batch_preds.append({"ents": ents, "relations": relations})

        return batch_preds
