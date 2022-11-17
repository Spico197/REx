from typing import Dict, Iterable, List, Optional


def validate_instance_has_the_same_keys(data: List[dict]) -> set:
    """Validate if all data have the same keys"""
    all_keys = set()
    for d in data:
        if len(all_keys) == 0:
            all_keys.update(d.keys())
        else:
            assert (
                set(d.keys()) == all_keys
            ), "Data instances does not have the same keys!"
    return all_keys


def group_instances_into_batch(
    instances: List[dict], keys: Optional[Iterable] = None
) -> dict:
    if not keys:
        keys = validate_instance_has_the_same_keys(instances)

    final_data = {key: [] for key in keys}
    for ins in instances:
        for key in keys:
            final_data[key].append(ins[key])
    return final_data


def decompose_batch_into_instances(batch: Dict[str, list]) -> List[dict]:
    keys = batch.keys()
    batch_size = 0
    for key in keys:
        batch_size = len(batch[key])
        break

    instances = []
    for i in range(batch_size):
        ins = {key: batch[key][i] for key in keys}
        instances.append(ins)

    return instances
