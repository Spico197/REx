from rex.utils.dict import PrettyPrintDefaultDict, _pretty_format, get_dict_content


def test_pretty_dict():
    pdict = PrettyPrintDefaultDict(list)
    example_data = {
        "p": 0.8296513175405908,
        "r": 0.8296563175405908,
        "f1": 0.8296513175405908,
        "tp": 123901293,
        "fp": 123123,
        "fn": 9080,
    }
    pdict["micro"] = example_data
    pdict["macro"] = example_data
    pdict["types"] = {"type1": example_data}

    assert (
        f"{pdict}"
        == '{\n  "micro": {\n    "p": "82.965 %",\n    "r": "82.966 %",\n    "f1": "82.965 %",\n    "tp": 123901293,\n    "fp": 123123,\n    "fn": 9080\n  },\n  "macro": {\n    "p": "82.965 %",\n    "r": "82.966 %",\n    "f1": "82.965 %",\n    "tp": 123901293,\n    "fp": 123123,\n    "fn": 9080\n  },\n  "types": {\n    "type1": {\n      "p": "82.965 %",\n      "r": "82.966 %",\n      "f1": "82.965 %",\n      "tp": 123901293,\n      "fp": 123123,\n      "fn": 9080\n    }\n  }\n}'
    )
    assert (
        pdict.__str__(indent=None)
        == '{"micro": {"p": "82.965 %", "r": "82.966 %", "f1": "82.965 %", "tp": 123901293, "fp": 123123, "fn": 9080}, "macro": {"p": "82.965 %", "r": "82.966 %", "f1": "82.965 %", "tp": 123901293, "fp": 123123, "fn": 9080}, "types": {"type1": {"p": "82.965 %", "r": "82.966 %", "f1": "82.965 %", "tp": 123901293, "fp": 123123, "fn": 9080}}}'
    )
    assert (
        pdict.jsonify()
        == '{"micro": {"p": 0.8296513175405908, "r": 0.8296563175405908, "f1": 0.8296513175405908, "tp": 123901293, "fp": 123123, "fn": 9080}, "macro": {"p": 0.8296513175405908, "r": 0.8296563175405908, "f1": 0.8296513175405908, "tp": 123901293, "fp": 123123, "fn": 9080}, "types": {"type1": {"p": 0.8296513175405908, "r": 0.8296563175405908, "f1": 0.8296513175405908, "tp": 123901293, "fp": 123123, "fn": 9080}}}'
    )

    pdict["none"] = None
    assert (
        pdict.__str__(indent=0, add_percentage_symbol=False)
        == '{\n"micro": {\n"p": "82.965",\n"r": "82.966",\n"f1": "82.965",\n"tp": 123901293,\n"fp": 123123,\n"fn": 9080\n},\n"macro": {\n"p": "82.965",\n"r": "82.966",\n"f1": "82.965",\n"tp": 123901293,\n"fp": 123123,\n"fn": 9080\n},\n"types": {\n"type1": {\n"p": "82.965",\n"r": "82.966",\n"f1": "82.965",\n"tp": 123901293,\n"fp": 123123,\n"fn": 9080\n}\n},\n"none": null\n}'
    )
    assert dict(pdict) == pdict.to_dict()

    assert _pretty_format(2) == 2


def test_get_dict_val():
    dict_item = {"1": 2, "3": {"4": 5, "6": {"7": 8}}}
    assert get_dict_content(dict_item, "1") == 2
    assert get_dict_content(dict_item, "3") == {"4": 5, "6": {"7": 8}}
    assert get_dict_content(dict_item, "3.4") == 5
    assert get_dict_content(dict_item, "3.6") == {"7": 8}
    assert get_dict_content(dict_item, "3.6.7") == 8
