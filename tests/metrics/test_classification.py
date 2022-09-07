import math

import pytest

from rex.metrics import classification


def test_accuracy():
    with pytest.raises(ValueError):
        classification.accuracy([], [])

    assert 0.5 == classification.accuracy([0, 1, 2, 3], [4, 1, 2, 6])


def test_mcml_prf1():
    pass


def test_mc_prf1():
    with pytest.raises(ValueError):
        classification.mc_prf1([], [])

    preds = [0, 1, 0, 1, 1]
    golds = [1, 1, 0, 0, 1]

    results = classification.mc_prf1(
        preds, golds, num_classes=2, ignore_labels=None, label_idx2name=None
    )

    assert results["micro"]["tp"] == 3
    assert results["micro"]["fp"] == 2
    assert results["micro"]["fn"] == 2

    assert (
        results["micro"]["p"]
        == results["micro"]["r"]
        == results["micro"]["f1"]
        == 3 / 5
    )

    assert (
        results["types"][0]["tp"]
        == results["types"][0]["fp"]
        == results["types"][0]["fn"]
        == 1
    )
    assert (
        results["types"][0]["p"]
        == results["types"][0]["r"]
        == results["types"][0]["f1"]
        == 1 / 2
    )

    assert results["types"][1]["tp"] == 2
    assert results["types"][1]["fp"] == results["types"][1]["fn"] == 1
    assert (
        results["types"][1]["p"]
        == results["types"][1]["r"]
        == results["types"][1]["f1"]
        == 2 / 3
    )

    assert results["macro"]["tp"] == 1.5
    assert results["macro"]["fp"] == 1
    assert results["macro"]["fn"] == 1

    assert results["macro"]["p"] == results["macro"]["r"] == results["macro"]["f1"]
    assert math.isclose(results["macro"]["p"], 7 / 12)

    assert isinstance(results["micro"]["p"], float)
