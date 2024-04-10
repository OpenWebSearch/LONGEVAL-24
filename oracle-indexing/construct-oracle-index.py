#!/usr/bin/env python3
from tira.third_party_integrations import ir_datasets
from tqdm import tqdm
import pandas as pd

dataset = ir_datasets.load('ir-lab-padua-2024/longeval-tiny-train-20240315-training')
queries_dict = {i.query_id: i.query for i in dataset.queries_iter()}
docs_store = dataset.docs_store()

oracle_index = []

for i in tqdm(dataset.qrels_iter()):
    if i.relevance <= 0:
        continue
    doc = docs_store.get(i.doc_id)
    if doc is None:
        continue
    oracle_index += [{'query_id': i.query_id, 'doc_id': i.doc_id, 'relevance': i.relevance, 'query': queries_dict[i.query_id], 'doc': doc.default_text()}]

pd.DataFrame(oracle_index).to_json('oracle-index.jsonl.gz', index=False, lines=True, orient='records')
