import torch


def re_collate_fn(data):
    final_data = {        
        "id": [],
        "token_ids": [],
        "mask": [],
        "labels": [],
        "head_pos": [],
        "tail_pos": []
    }
    for d in data:
        for key in final_data:
            final_data[key].append(d[key])

    final_data['token_ids'] = torch.tensor(final_data['token_ids'], dtype=torch.long)
    final_data['mask'] = torch.tensor(final_data['mask'], dtype=torch.long)
    if all(x is not None for x in final_data['labels']):
        final_data['labels'] = torch.tensor(final_data['labels'], dtype=torch.long)
    else:
        final_data['labels'] = None
    final_data['head_pos'] = torch.tensor(final_data['head_pos'], dtype=torch.long)
    final_data['tail_pos'] = torch.tensor(final_data['tail_pos'], dtype=torch.long)

    return final_data
