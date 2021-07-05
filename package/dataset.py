# -*- coding: utf-8 -*-
from package.utils import read_bag_from_nyt10_json, read_instance_from_nyt10_json
from torch.utils.data import DataLoader, Dataset
import random
import torch

relation2id = \
    {'NA': 0,
     '/location/neighborhood/neighborhood_of': 1,
     '/location/fr_region/capital': 2,
     '/location/cn_province/capital': 3,
     '/location/in_state/administrative_capital': 4,
     '/base/locations/countries/states_provinces_within': 5,
     '/business/company/founders': 6,
     '/location/country/languages_spoken': 7,
     '/people/person/place_of_birth': 8,
     '/people/deceased_person/place_of_death': 9,
     '/location/it_region/capital': 10,
     '/people/family/members': 11,
     '/location/us_state/capital': 12,
     '/location/us_county/county_seat': 13,
     '/people/profession/people_with_this_profession': 14,
     '/location/br_state/capital': 15,
     '/location/in_state/legislative_capital': 16,
     '/sports/sports_team/location': 17,
     '/people/person/religion': 18,
     '/location/in_state/judicial_capital': 19,
     '/business/company_advisor/companies_advised': 20,
     '/people/family/country': 21,
     '/time/event/locations': 22,
     '/business/company/place_founded': 23,
     '/location/administrative_division/country': 24,
     '/people/ethnicity/included_in_group': 25,
     '/location/mx_state/capital': 26,
     '/location/province/capital': 27,
     '/people/person/nationality': 28,
     '/business/person/company': 29,
     '/business/shopping_center_owner/shopping_centers_owned': 30,
     '/business/company/advisors': 31,
     '/business/shopping_center/owner': 32,
     '/people/person/ethnicity': 33,
     '/people/deceased_person/place_of_burial': 34,
     '/people/ethnicity/geographic_distribution': 35,
     '/people/person/place_lived': 36,
     '/business/company/major_shareholders': 37,
     '/broadcast/producer/location': 38,
     '/broadcast/content/location': 39,
     '/business/business_location/parent_company': 40,
     '/location/jp_prefecture/capital': 41,
     '/film/film/featured_film_locations': 42,
     '/people/place_of_interment/interred_here': 43,
     '/location/de_state/capital': 44,
     '/people/person/profession': 45,
     '/business/company/locations': 46,
     '/location/country/capital': 47,
     '/location/location/contains': 48,
     '/location/country/administrative_divisions': 49,
     '/people/person/children': 50,
     '/film/film_location/featured_in_films': 51,
     '/film/film_festival/location': 52}


class NYT10TrainDataset(Dataset):
    """Training Dataset for NYT10"""

    def __init__(self, path_to_nyt, tokenizer, bag_size, max_length):
        self.bags = read_bag_from_nyt10_json(path_to_nyt)
        self.bags_id = list(self.bags.keys())
        self.tokenizer = tokenizer

        self.bag_size = bag_size
        self.max_length = max_length
        self.num_relations = len(relation2id)

    def _pad_item(self, tokens, pos_hs, pos_ts, masks):
        max_length = max([len(token) for token in tokens])
        tokens_pad, masks_pad, pos_hs_pad, pos_ts_pad = [], [], [], []
        for token, mask, pos_h, pos_t in zip(tokens, masks, pos_hs, pos_ts):

            if len(token) < max_length:
                token += ['[PAD]'] * (max_length - len(token))
                mask += [0] * (max_length - len(mask))
                pos_h += [0] * (max_length - len(pos_h))
                pos_t += [0] * (max_length - len(pos_t))

            tokens_pad.append(self.tokenizer.convert_tokens_to_ids(token[:self.max_length]))
            masks_pad.append(mask[:self.max_length])
            pos_hs_pad.append(pos_h[:self.max_length])
            pos_ts_pad.append(pos_t[:self.max_length])
        return tokens_pad, pos_hs_pad, pos_ts_pad, masks_pad

    def _decode_bag(self, bag):
        bag_tokens, bag_masks, bag_pos_h, bag_pos_t = [], [], [], []
        for instance in bag:
            text = instance['text']
            x1, y1 = instance['head_pos']
            x2, y2 = instance['tail_pos']

            if y2 <= x1:
                x1, y1, x2, y2 = x2, y2, x1, y1

            token = ['[CLS]']
            token += self.tokenizer.tokenize(text[:x1])
            token += self.tokenizer.tokenize(text[x1:y1])
            pos1 = len(token)
            token += self.tokenizer.tokenize(text[y1:x2])
            token += self.tokenizer.tokenize(text[x2:y2])
            pos2 = len(token)
            token += self.tokenizer.tokenize(text[y2:])
            token += ['[SEP]']

            pos1 = min(pos1, self.max_length)
            pos2 = min(pos2, self.max_length)

            mask = [0] + [1] * (pos1 - 1) + [2] * (pos2 - pos1) + [3] * (len(token) - pos2 - 1) + [0]

            pos_h, pos_t = [], []
            for i in range(len(token)):
                pos_h.append(min(i - pos1 + self.max_length, 2 * self.max_length - 1))
                pos_t.append(min(i - pos2 + self.max_length, 2 * self.max_length - 1))

            bag_tokens.append(token)
            bag_masks.append(mask)
            bag_pos_h.append(pos_h)
            bag_pos_t.append(pos_t)

        return bag_tokens, bag_masks, bag_pos_h, bag_pos_t

    def collate_fn(self, batch):
        """collate_fn for 'torch.utils.data.DataLoader' """

        tokens, masks, labels, pos_hs, pos_ts = [], [], [], [], []
        for label, bag in batch:
            labels.append(label)

            bag_tokens, bag_masks, bag_pos_h, bag_pos_t = self._decode_bag(bag)
            bag_tokens_pad, bag_masks_pad, bag_pos_h_pad, bag_pos_t_pad = self._pad_item(bag_tokens, bag_masks,
                                                                                         bag_pos_h, bag_pos_t)

            tokens.append(torch.LongTensor(bag_tokens_pad))
            masks.append(torch.LongTensor(bag_masks_pad))
            pos_hs.append(torch.LongTensor(bag_pos_h_pad))
            pos_ts.append(torch.LongTensor(bag_pos_t_pad))

        return {'bags_of_token_ids': tokens, 'bags_of_pos_h_ids': pos_hs, 'bags_of_pos_t_ids': pos_ts,
                'bags_of_mask': masks, 'label': torch.LongTensor(labels)}

    def fast_collate_fn(self, batch):
        """fast version collate_fn for 'torch.utils.data.DataLoader' """

        tokens, masks, labels, pos_hs, pos_ts, scopes = [], [], [], [], [], []
        for label, bag in batch:
            labels.append(label)
            scopes.append((0, len(bag)) if len(scopes) == 0 else (scopes[-1][1], scopes[-1][1] + len(bag)))

            bag_tokens, bag_masks, bag_pos_h, bag_pos_t = self._decode_bag(bag)
            tokens.extend(bag_tokens)
            masks.extend(bag_masks)
            pos_hs.extend(bag_pos_h)
            pos_ts.extend(bag_pos_t)

        tokens_pad, pos_hs_pad, pos_ts_pad, masks_pad = self._pad_item(tokens, pos_hs, pos_ts, masks)

        tokens_pad = torch.LongTensor(tokens_pad)
        masks_pad = torch.LongTensor(masks_pad)
        pos_hs_pad = torch.LongTensor(pos_hs_pad)
        pos_ts_pad = torch.LongTensor(pos_ts_pad)
        labels = torch.LongTensor(labels)
        scopes = torch.LongTensor(scopes)

        return {'bags_of_token_ids': tokens_pad, 'bags_of_pos_h_ids': pos_hs_pad, 'bags_of_pos_t_ids': pos_ts_pad,
                'bags_of_mask': masks_pad, 'label': labels, 'scope': scopes}

    def __len__(self):
        return len(self.bags_id)

    def __getitem__(self, idx):
        key = self.bags_id[idx]
        bag = self.bags[key]
        relation = key[0]

        # resize bag
        if len(bag) > self.bag_size:
            bag = random.sample(bag, self.bag_size)

        return (relation2id[relation] if relation in relation2id else 0), bag


class NYT10TestDataset(Dataset):
    """Testing Dataset for NYT10"""

    def __init__(self, path_to_nyt, tokenizer, max_length):
        self.instances = read_instance_from_nyt10_json(path_to_nyt)
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.num_relations = len(relation2id)

    def collate_fn(self, batch):
        """collate_fn for 'torch.utils.data.DataLoader'
        """

        tokens, masks, labels, pos_hs, pos_ts = [], [], [], [], []
        for label, instance in batch:
            labels.append(label)

            text = instance['text']
            x1, y1 = instance['head_pos']
            x2, y2 = instance['tail_pos']

            if y2 <= x1:
                x1, y1, x2, y2 = x2, y2, x1, y1

            token = ['[CLS]']
            token += self.tokenizer.tokenize(text[:x1])
            token += self.tokenizer.tokenize(text[x1:y1])
            pos1 = len(token)
            token += self.tokenizer.tokenize(text[y1:x2])
            token += self.tokenizer.tokenize(text[x2:y2])
            pos2 = len(token)
            token += self.tokenizer.tokenize(text[y2:])
            token += ['[SEP]']

            pos1 = min(pos1, self.max_length)
            pos2 = min(pos2, self.max_length)

            mask = [0] + [1] * (pos1 - 1) + [2] * (pos2 - pos1) + [3] * (len(token) - pos2 - 1) + [0]

            pos_h, pos_t = [], []
            for i in range(len(token)):
                pos_h.append(min(i - pos1 + self.max_length, 2 * self.max_length - 1))
                pos_t.append(min(i - pos2 + self.max_length, 2 * self.max_length - 1))

            tokens.append(token)
            masks.append(mask)
            pos_hs.append(pos_h)
            pos_ts.append(pos_t)

        # padding
        max_length = max([len(token) for token in tokens])
        tokens_pad, masks_pad, pos_hs_pad, pos_ts_pad = [], [], [], []
        for token, mask, pos_h, pos_t in zip(tokens, masks, pos_hs, pos_ts):

            if len(token) < max_length:
                token += ['[PAD]'] * (max_length - len(token))
                mask += [0] * (max_length - len(mask))
                pos_h += [0] * (max_length - len(pos_h))
                pos_t += [0] * (max_length - len(pos_t))

            tokens_pad.append(self.tokenizer.convert_tokens_to_ids(token[:self.max_length]))
            masks_pad.append(mask[:self.max_length])
            pos_hs_pad.append(pos_h[:self.max_length])
            pos_ts_pad.append(pos_t[:self.max_length])

        return {'token_ids': torch.LongTensor(tokens_pad),
                'pos_h_ids': torch.LongTensor(pos_hs_pad), 'pos_t_ids': torch.LongTensor(pos_ts_pad),
                'mask': torch.LongTensor(masks_pad), 'label': torch.LongTensor(labels)}

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        relation = instance['relation']

        return relation2id[relation] if relation in relation2id else 0, instance
