# -*- coding: utf-8 -*-
import torch
import json
import random
import numpy as np

from collections import defaultdict


def set_random_seed(seed):
    """Set Random State"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_tensor_to_device(x, device):
    """Move Tensor or Dict,List to the device"""
    if isinstance(x, dict):
        return {key: move_tensor_to_device(item, device) for key, item in x.items()}

    elif isinstance(x, (list, tuple)):
        return [move_tensor_to_device(item, device) for item in x]

    elif isinstance(x, torch.Tensor):
        return x.to(device)

    else:
        return x


def read_bag_from_nyt10_json(path_to_nyt10):
    """Read bags from NYT10 Json File"""
    bags = defaultdict(list)
    with open(path_to_nyt10, 'r') as f:
        for sample in json.load(f):
            relation = sample['relation']
            head = sample['head']['word']
            tail = sample['tail']['word']
            text = sample['sentence']
            bags[relation, head, tail].append({'head': head, 'head_pos': _pos(head, text),
                                               'tail': tail, 'tail_pos': _pos(tail, text),
                                               'relation': relation,
                                               'text': text
                                               })

    return bags


def read_instance_from_nyt10_json(path_to_nyt10):
    """Read instance from NYT10 Json File"""
    out = []
    with open(path_to_nyt10, 'r') as f:
        for sample in json.load(f):
            relation = sample['relation']
            head = sample['head']['word']
            tail = sample['tail']['word']
            text = sample['sentence']
            out.append({'head': head, 'head_pos': _pos(head, text),
                        'tail': tail, 'tail_pos': _pos(tail, text),
                        'relation': relation,
                        'text': text,
                        })

    return out


def _pos(x, text):
    start = text.find(x)
    return start, start + len(x)
