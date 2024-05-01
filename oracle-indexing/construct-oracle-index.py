#!/usr/bin/env python3
from tira.third_party_integrations import ir_datasets
from tqdm import tqdm
import pandas as pd

DATASETS = {
    '2023-01': 'longeval-2023-01-20240423-training',
    '2022-06a': 'longeval-heldout-20230513-training',
    '2022-06b': 'longeval-train-20230513-training',
    '2022-07': 'longeval-short-july-20230513-training',
    '2022-09': 'longeval-long-september-20230513-training'
}

oracle_index = []

for ds_id, d in tqdm(DATASETS.items()):
    dataset = ir_datasets.load(f'ir-benchmarks/{d}')
    queries_dict = {i.query_id: i.query for i in dataset.queries_iter()}
    docs_store = dataset.docs_store()

    for i in tqdm(dataset.qrels_iter()):
        if i.relevance < 0:
            continue
        doc = docs_store.get(i.doc_id)
        if doc is None:
            continue
        oracle_index += [{'query_id': i.query_id, 'doc_id': i.doc_id, 'relevance': i.relevance, 'query': queries_dict[i.query_id], 'doc': doc.default_text(), 'dataset': d}]

    del docs_store

pd.DataFrame(oracle_index).to_json('oracle-index.jsonl.gz', index=False, lines=True, orient='records')
