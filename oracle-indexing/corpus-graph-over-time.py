#!/usr/bin/env python3
import pathlib
import pandas as pd
import argparse
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
from tqdm import tqdm
import random

def load_oracle_index(file_name):
    ret = pd.read_json(file_name, orient='records', lines=True)
    return [i.to_dict() for _, i in ret.iterrows() if i['relevance'] > 0]

def parse_args():
    parser = argparse.ArgumentParser(description='Construct neighbors')
    parser.add_argument('--input-dataset', type=str, help='Input file', default='cranfield')
    parser.add_argument('--output-dir', type=str, help='Output file', required=True)
    parser.add_argument('--query-document-pairs', type=str, help='Output file', default=str((pathlib.Path(__file__).parent.resolve() / 'oracle-index.jsonl.gz').absolute()), required=False)
    parser.add_argument('--top-k', type=int, help='Number of neighbors', default=15)
    parser.add_argument('--top-terms-per-document', type=int, help='Number of terms per document', default=50)
    return parser.parse_args()

def tokenise_query(query, seed=42, top_terms_per_document=50):
    import pyterrier as pt
    tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
    tokens_raw = tokeniser.getTokens(query)

    if top_terms_per_document <= 1:
        return ' '.join(tokens_raw)

    tokens_count = {}
    for t in tokens_raw:
        if t not in tokens_count:
            tokens_count[t] = 1
            
        tokens_count[t] += 1
    
    query = list(tokens_count.keys())
    random.Random(seed).shuffle(query)
    query = sorted(query, key=lambda i: tokens_count[i], reverse=True)

    return ' '.join(query[:top_terms_per_document])

def corpus_graph_over_time(retrieval_pipeline, oracle_index, top_terms_per_document):
    ret = []

    for i in tqdm(oracle_index):
        query_by_example = tokenise_query(i['doc'], top_terms_per_document=top_terms_per_document)
        results = retrieval_pipeline.search(query_by_example)
        del results['query']
        del results['qid']
        del results['docid']

        i['top_bm25_results'] = [i.to_dict() for _, i in results.iterrows()]
        del i['doc']
        ret += [i]
    
    return pd.DataFrame(ret)

if __name__ == '__main__':
    args = parse_args()
    oracle_index = load_oracle_index(args.query_document_pairs)

    # This method ensures that that PyTerrier is loaded so that it also works in the TIRA sandbox
    tira = Client()
    ensure_pyterrier_is_loaded()
    import pyterrier as pt

    index = tira.pt.index('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', args.input_dataset)
    bm25 = pt.BatchRetrieve(index, wmodel='BM25', num_results=args.top_k)

    output = corpus_graph_over_time(bm25, oracle_index, args.top_terms_per_document)

    output.to_json(pathlib.Path(args.output_dir) / 'corpus-graph-over-time.jsonl.gz', index=False, lines=True, orient='records')