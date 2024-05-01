#!/usr/bin/env python3
from tira.third_party_integrations import ir_datasets
from tqdm import tqdm
import pandas as pd

DATASETS = {
    '2022-06a': 'longeval-heldout-20230513-training',
    '2022-06b': 'longeval-train-20230513-training',
    '2022-07': 'longeval-short-july-20230513-training',
    '2022-09': 'longeval-long-september-20230513-training',

    '2023-01': 'longeval-2023-01-20240423-training',
    '2023-06': 'longeval-2023-06-20240418-training',
    '2023-08': 'longeval-2023-08-20240418-training',
}

from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client

# This method ensures that that PyTerrier is loaded so that it also works in the TIRA sandbox
ensure_pyterrier_is_loaded()
import pyterrier as pt

tira = Client()

for d, ds_id in tqdm(DATASETS.items()):
    pt_dataset = pt.get_dataset(f'irds:ir-benchmarks/{ds_id}')
    index = tira.pt.index('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', pt_dataset)