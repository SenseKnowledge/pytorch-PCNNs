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


def read_bag_from_nyt10m_text(path_to_nyt10m):
    """Read bags from NYT10m Json File"""
    bags = defaultdict(list)
    with open(path_to_nyt10m, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            relation = sample['relation']
            head = sample['h']['name']
            tail = sample['t']['name']
            text = sample['text']
            bags[relation, head, tail].append({'head': head, 'head_pos': sample['h']['pos'],
                                               'tail': tail, 'tail_pos': sample['t']['pos'],
                                               'relation': relation,
                                               'text': text
                                               })

    return bags


def read_instance_from_nyt10m_text(path_to_nyt10m):
    """Read instance from NYT10m Json File"""
    out = []
    with open(path_to_nyt10m, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            relation = sample['relation']
            head = sample['h']['name']
            tail = sample['t']['name']
            text = sample['text']
            out.append({'head': head, 'head_pos': sample['h']['pos'],
                        'tail': tail, 'tail_pos': sample['t']['pos'],
                        'relation': relation,
                        'text': text
                        })

    return out
