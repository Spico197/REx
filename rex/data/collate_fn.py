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


def bag_re_collate_fn(data):
    final_data = {        
        "id": [],
        "token_ids": [],
        "mask": [],
        "labels": [],
        "head_pos": [],
        "tail_pos": [],
        "scopes": [],
    }
    scope = []; scope_begin = 0; labels = []
    # data is a batch, for every bag in the batch:
    for bag in data:
        scope_ = [scope_begin]
        # for every instance in the bag
        for ins in bag:
            for key in final_data:
                if key != 'scopes':
                    final_data[key].append(ins[key])
            scope_begin += 1
        labels.append(ins['labels'])
        scope_.append(scope_begin)
        scope.append(scope_)
    final_data['scopes'] = scope
    
    final_data['token_ids'] = torch.tensor(final_data['token_ids'], dtype=torch.long)
    final_data['mask'] = torch.tensor(final_data['mask'], dtype=torch.long)
    if all(x is not None for x in final_data['labels']):
        final_data['labels'] = torch.tensor(labels, dtype=torch.long)
    else:
        final_data['labels'] = None
    final_data['head_pos'] = torch.tensor(final_data['head_pos'], dtype=torch.long)
    final_data['tail_pos'] = torch.tensor(final_data['tail_pos'], dtype=torch.long)

    return final_data
