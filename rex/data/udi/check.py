import os

from rex.utils.io import find_files, load_jsonlines
from rex.utils.progress_bar import pbar


def start_endp1_span_is_valid(span: list, text: str):
    return len(span) == 2 and 0 <= span[0] < span[1] <= len(text)


def check_udi_instance(instance: dict):
    assert isinstance(instance["id"], str)
    assert isinstance(instance["instruction"], str)
    assert instance["instruction"].strip() == instance["instruction"]
    assert isinstance(instance["schema"], dict)
    for key in instance["schema"]:
        assert key in ["cls", "ent", "rel", "event"]
        if key in ["cls", "ent", "rel"]:
            assert isinstance(instance["schema"][key], list)
            assert all(
                isinstance(x, str) and len(x) > 0 and x.strip() == x
                for x in instance["schema"][key]
            )
        elif key == "event":
            assert isinstance(instance["schema"]["event"], dict)
            for event_type in instance["schema"]["event"]:
                assert isinstance(instance["schema"]["event"][event_type], list)
                assert all(
                    isinstance(x, str) and len(x) > 0 and x.strip() == x
                    for x in instance["schema"]["event"][event_type]
                )
        else:
            raise ValueError
    assert isinstance(instance["ans"], dict)
    for key in instance["ans"]:
        assert key in ["cls", "ent", "rel", "event", "span"]
        if key == "cls":
            assert isinstance(instance["ans"]["cls"], list)
            assert all(
                isinstance(x, str) and len(x) > 0 and x.strip() == x
                for x in instance["ans"]["cls"]
            )
            assert all(x in instance["schema"]["cls"] for x in instance["ans"]["cls"])
        elif key == "ent":
            assert isinstance(instance["ans"]["ent"], list)
            assert all(isinstance(x, dict) for x in instance["ans"]["ent"])
            for ent in instance["ans"]["ent"]:
                assert (
                    isinstance(ent["type"], str) and ent["type"].strip() == ent["type"]
                )
                assert ent["type"] in instance["schema"]["ent"]
                assert (
                    isinstance(ent["text"], str) and ent["text"].strip() == ent["text"]
                )
                assert len(ent["text"]) > 0
                assert instance["text"][ent["span"][0] : ent["span"][1]] == ent["text"]
                assert isinstance(ent["span"], list)
                assert len(ent["span"]) == 2
                assert start_endp1_span_is_valid(ent["span"], instance["text"])
                assert all(isinstance(x, int) for x in ent["span"])
        elif key == "rel":
            assert isinstance(instance["ans"]["rel"], list)
            assert all(isinstance(x, dict) for x in instance["ans"]["rel"])
            for rel in instance["ans"]["rel"]:
                assert isinstance(rel["relation"], str)
                assert rel["relation"] in instance["schema"]["rel"]
                assert isinstance(rel["head"], dict)
                assert len(rel["head"]["text"]) > 0
                assert len(rel["tail"]["text"]) > 0
                assert start_endp1_span_is_valid(rel["head"]["span"], instance["text"])
                assert start_endp1_span_is_valid(rel["tail"]["span"], instance["text"])
                assert (
                    instance["text"][rel["head"]["span"][0] : rel["head"]["span"][1]]
                    == rel["head"]["text"]
                )
                assert isinstance(rel["tail"], dict)
                assert (
                    instance["text"][rel["tail"]["span"][0] : rel["tail"]["span"][1]]
                    == rel["tail"]["text"]
                )
        elif key == "event":
            assert isinstance(instance["ans"]["event"], list)
            assert all(isinstance(x, dict) for x in instance["ans"]["event"])
            for event in instance["ans"]["event"]:
                assert event["event_type"] in instance["schema"]["event"]
                assert isinstance(event["trigger"], dict)
                assert len(event["trigger"]["text"]) > 0
                assert event["trigger"]["text"] in instance["text"]
                assert start_endp1_span_is_valid(
                    event["trigger"]["span"], instance["text"]
                )
                assert (
                    instance["text"][
                        event["trigger"]["span"][0] : event["trigger"]["span"][1]
                    ]
                    == event["trigger"]["text"]
                )
                for arg in event["args"]:
                    assert (
                        arg["role"] in instance["schema"]["event"][event["event_type"]]
                    )
                    assert isinstance(arg["text"], str)
                    assert len(arg["text"]) > 0
                    assert start_endp1_span_is_valid(arg["span"], instance["text"])
                    assert (
                        instance["text"][arg["span"][0] : arg["span"][1]] == arg["text"]
                    )
        elif key == "span":
            assert isinstance(instance["ans"]["span"], list)
            assert all(isinstance(x, dict) for x in instance["ans"]["span"])
            for span in instance["ans"]["span"]:
                assert isinstance(span["text"], str)
                assert len(span["text"]) > 0
                assert start_endp1_span_is_valid(span["span"], instance["text"])
                assert (
                    instance["text"][span["span"][0] : span["span"][1]] == span["text"]
                )
        else:
            raise ValueError
    assert isinstance(instance["text"], str)
    assert isinstance(instance["bg"], str)
    for key in ["ent", "rel", "event"]:
        if instance["schema"].get(key):
            assert len(instance["text"]) > 0
    if "span" in instance["ans"]:
        assert len(instance["text"]) > 0
    assert instance["instruction"] or instance["text"] or instance["bg"]


def is_valid_udi_instance(instance: dict):
    ok = True
    try:
        check_udi_instance(instance)
    except:
        ok = False
    return ok


def find_jsonl(folder: str):
    return find_files(r".*\.jsonl", folder, recursive=True)


def main():
    filepaths = find_jsonl(".")
    bar = pbar(filepaths)
    for filepath in bar:
        data = load_jsonlines(filepath)
        data_ok = True
        for ins in data:
            if not ins["instruction"]:
                bar.write(f"No Instruction: {filepath}")
                break
            ok = True
            # ok = is_valid_udi_instance(ins)
            try:
                check_udi_instance(ins)
            except Exception as err:
                ok = False
                bar.write(f"❌ {filepath}")
                raise err
            if not ok:
                data_ok = False
                break
        if not data_ok:
            bar.write(f"❌ {filepath}")


if __name__ == "__main__":
    main()
