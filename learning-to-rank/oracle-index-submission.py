#!/usr/bin/env python3
import pyterrier as pt
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
import pandas as pd
from tqdm import tqdm
from reverted_index import construct_reverted_index_of_the_past

LAG_TO_DATASET_ID = {
    'lag1': 'longeval-2023-01-20240423-training',
    'lag6': 'longeval-2023-06-20240418-training',
    'lag8': 'longeval-2023-08-20240418-training'
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lag', type=str, help='The dataset to run on', choices=['lag1', 'lag6', 'lag8'])
    return parser.parse_args()

def main(lag):
    oracle_datasets=('longeval-train-20230513-training', 'longeval-heldout-20230513-training', 'longeval-short-july-20230513-training', 'longeval-long-september-20230513-training', 'longeval-2023-01-20240423-training')
    runs_to_fuse = {
        'lag1': '../runs/ows_ltr_all/ows_ltr_all.train_2024',
        'lag6': '../runs/ows_ltr_all/ows_ltr_all.lag6',
        'lag8': '../runs/ows_ltr_all/ows_ltr_all.lag8',
    }


    ensure_pyterrier_is_loaded()
    tira = Client()

    reverted_index_of_the_past = construct_reverted_index_of_the_past(tira, pt.get_dataset('irds:ir-benchmarks/' + LAG_TO_DATASET_ID[lag]), oracle_datasets=oracle_datasets)

    bm25 = pt.io.read_results(runs_to_fuse[lag])

    ret = []
    for _, i in tqdm(bm25.iterrows()):
        i = i.to_dict()
        score = i['score']
        if i['qid'] in reverted_index_of_the_past and i['docno'] in reverted_index_of_the_past[i['qid']]:
            score += (5* max(0.1, reverted_index_of_the_past[i['qid']][i['docno']][1]))
        
        i['score'] = score
        ret += [i]

    ret = pd.DataFrame(ret)
    pt.io.write_results(ret, f'ows_bm25_reverted_index/ows_bm25_reverted_index.{lag}')


if __name__ == '__main__':
    args = parse_args()
    main(args.lag)