from collections import defaultdict

import numpy as np
import pandas as pd
from tira.pyterrier_util import TiraApplyFeatureTransformer

from reverted_index import construct_reverted_index_of_the_past


def one_hot_encode(labels: list[str]):
    return lambda x: np.array([1 if x == label else 0 for label in labels])


def get_query_features(tira, dataset):
    qpp = tira.pt.query_features('ir-benchmarks/qpptk/all-predictors', dataset)
    intent_prediction = tira.pt.query_features(
        'ir-benchmarks/dossier/pre-retrieval-query-intent', dataset,
        feature_selection=['intent_prediction'],
        map_features={'intent_prediction': one_hot_encode(['Instrumental', 'Factual', 'Navigational', 'Transactional', 'Abstain'])}
    )
    query_health_classification = tira.pt.query_features('ir-benchmarks/fschlatt/query-health-classification', dataset)

    return intent_prediction ** qpp ** query_health_classification


def get_document_features(tira, dataset):
    document_health_classification = tira.pt.doc_features('ir-benchmarks/fschlatt/document-health-classification', dataset)
    genre_mlp_classifier = tira.pt.doc_features('ir-benchmarks/tu-dresden-01/genre-mlp', dataset,
                                                feature_selection=['probability_Discussion', 'probability_Shop', 'probability_Download',
                                                                   'probability_Articles', 'probability_Help', 'probability_Linklists',
                                                                   'probability_Porttrait private', 'probability_Protrait non private'])
    spacy_features = tira.pt.doc_features('ir-benchmarks/tu-dresden-04/spacy-document-features', dataset)

    return document_health_classification ** genre_mlp_classifier ** spacy_features


def get_query_document_features(tira, dataset):
    return (
        tira.pt.from_submission('workshop-on-open-web-search/fschlatt/rank-zephyr', dataset) **
        tira.pt.from_submission('ir-benchmarks/fschlatt/sparse-cross-encoder-4-512', dataset) **
        tira.pt.from_submission('ir-benchmarks/fschlatt/castorini-list-in-t5-150', dataset) **
        tira.pt.from_submission('ir-benchmarks/tira-ir-starter/SBERT multi-qa-mpnet-base-cos-v1 (tira-ir-starter-beir)', dataset) **
        tira.pt.from_submission('ir-benchmarks/tira-ir-starter/MonoT5 Base (tira-ir-starter-gygaggle)', dataset) **
        tira.pt.from_submission('ir-benchmarks/tira-ir-starter/ColBERT Re-Rank (tira-ir-starter-pyterrier)', dataset) **
        tira.pt.from_submission('ir-benchmarks/tira-ir-starter/ANCE Base Cosine (tira-ir-starter-beir)', dataset) **
        tira.pt.from_submission('ir-benchmarks/tira-ir-starter/BM25 Re-Rank (tira-ir-starter-pyterrier)', dataset) **
        tira.pt.from_submission('ir-benchmarks/tira-ir-starter/PL2 Re-Rank (tira-ir-starter-pyterrier)', dataset) **
        tira.pt.from_submission('ir-benchmarks/tira-ir-starter/DirichletLM Re-Rank (tira-ir-starter-pyterrier)', dataset) **
        tira.pt.from_submission('ir-benchmarks/tira-ir-starter/DLH Re-Rank (tira-ir-starter-pyterrier)', dataset) **
        tira.pt.from_submission('ir-benchmarks/tira-ir-starter/LGD Re-Rank (tira-ir-starter-pyterrier)', dataset)
    )


def keyquery_score_dict(tira, dataset_id, offset, expansion_method, retrieval_model, fb_terms, fb_docs):
    f = tira.get_run_output(f'ir-benchmarks/ows/time-keyquery-offset-{offset}', dataset_id)
    f = pd.read_json(f'{f}/{expansion_method}_{retrieval_model}_{fb_terms}_{fb_docs}.jsonl.gz', lines=True)
    default_value = np.array([0, 0])
    ret = defaultdict(lambda: defaultdict(lambda: default_value))

    for _, row in f.iterrows():
        ret[row['qid']][row['docno']] = np.array([1, row['score']])

    return ret


def get_keyquery_features(tira, dataset):
    dataset_id = dataset.irds_ref().dataset_id()
    return TiraApplyFeatureTransformer(keyquery_score_dict(tira, dataset_id, '2022', 'bo1', 'BM25', 30, 5), name='keyquery_features')


def get_reverted_index_features(tira, dataset):
    reverted_index_of_the_past = construct_reverted_index_of_the_past(tira, dataset)
    return TiraApplyFeatureTransformer(reverted_index_of_the_past, name='reverted_index_features')


def get_all_features(tira, dataset):
    return (
        get_query_features(tira, dataset) **
        get_document_features(tira, dataset) **
        get_query_document_features(tira, dataset) **
        get_keyquery_features(tira, dataset) **
        get_reverted_index_features(tira, dataset)
    )
