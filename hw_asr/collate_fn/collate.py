import logging
from typing import List
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = defaultdict(list)

    for i in dataset_items:
        result_batch['reference'].append(i['reference'].squeeze(0))
        result_batch['reference_length'].append(i['reference'].shape[-1])
        result_batch['target'].append(i['target'].squeeze(0))
        result_batch['mix'].append(i['mix'].squeeze(0))
        result_batch['speaker_id'].append(i['speaker_id'])
        result_batch['path'].append(i['path'])

    for v in result_batch:
        if v == 'reference':
            result_batch[v] = pad_sequence(result_batch[v], batch_first=True).unsqueeze(1)
        elif v == 'reference_length':
            result_batch[v] = torch.LongTensor(result_batch[v])
        elif v == 'target':
            result_batch[v] = pad_sequence(result_batch[v], batch_first=True).unsqueeze(1)
        elif v == 'mix':
            result_batch[v] = pad_sequence(result_batch[v], batch_first=True).unsqueeze(1)
        elif v == 'speaker_id':
            result_batch[v] = torch.LongTensor(result_batch[v])
        elif v == 'path':
            result_batch[v] = result_batch[v]

    return result_batch
