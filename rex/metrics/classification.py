from collections import defaultdict


def accuracy(preds, golds):
    if len(preds) == 0 or len(golds) == 0:
        raise ValueError("Preds or golds is empty.")
    correct = total = 0
    for pred, gold in zip(preds, golds):
        total += 1
        if pred == gold:
            correct += 1
    return correct / total


def mcml_prf1(preds, golds, eps=1e-12):
    if len(preds) == 0 or len(golds) == 0:
        raise ValueError("Preds or golds is empty.")
    measure_results = defaultdict(lambda: {"p": 0.0, "r": 0.0, "f1": 0.0})
    supports = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0})
    for pred, gold in zip(preds, golds):
        if len(pred) != len(gold):
            raise ValueError(f"Pred: {pred} cannot be matched with gold: {gold} in length.")
        for type_idx, (p, g) in enumerate(zip(pred, gold)):
            if p == 1 and g == 1:
                supports[type_idx]['tp'] += 1
            elif p == 1 and g == 0:
                supports[type_idx]['fp'] += 1
            elif p == 0 and g == 1:
                supports[type_idx]['fn'] += 1
    g_tp = g_fp = g_fn = 0.0
    ps = []
    rs = []
    f1s = []
    for type_idx in supports:
        measure_results[type_idx]['p'] = p \
            = supports[type_idx]['tp'] / (supports[type_idx]['tp'] + supports[type_idx]['fp'] + eps)
        measure_results[type_idx]['r'] = r \
            = supports[type_idx]['tp'] / (supports[type_idx]['tp'] + supports[type_idx]['fn'] + eps)
        measure_results[type_idx]['f1'] = f1 = 2 * p * r / (p + r + eps)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        g_tp += supports[type_idx]['tp']
        g_fp += supports[type_idx]['fp']
        g_fn += supports[type_idx]['fn']

    g_p = measure_results['micro']['p'] = g_tp / (g_tp + g_fp + eps)
    g_r = measure_results['micro']['r'] = g_tp / (g_tp + g_fn + eps)
    measure_results['micro']['f1'] = 2 * g_p * g_r / (g_p + g_r + eps)

    measure_results['macro']['p'] = sum(ps) / len(ps)
    measure_results['macro']['r'] = sum(rs) / len(rs)
    measure_results['macro']['f1'] = sum(f1s) / len(f1s)

    return measure_results


def mc_prf1(preds, golds, num_classes, eps=1e-12):
    # TODO: add multi-class prf1 support
    raise NotImplementedError
