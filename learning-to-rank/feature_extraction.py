from collections import defaultdict

import numpy as np
import pandas as pd
from tira.pyterrier_util import TiraApplyFeatureTransformer, TiraNamedFeatureTransformer

from reverted_index import construct_reverted_index_of_the_past


def one_hot_encode(labels: list[str]):
    return lambda x: np.array([1 if x == label else 0 for label in labels])


def query_features(tira, dataset):
    intents = ['Instrumental', 'Factual', 'Navigational', 'Transactional', 'Abstain']

    qpp = tira.pt.query_features('ir-benchmarks/qpptk/all-predictors', dataset)
    intent_prediction = tira.pt.query_features(
        'ir-benchmarks/dossier/pre-retrieval-query-intent', dataset,
        feature_selection=['intent_prediction'],
        map_features={'intent_prediction': one_hot_encode(intents)}
    )
    query_health_classification = tira.pt.query_features('ir-benchmarks/fschlatt/query-health-classification', dataset)

    intent_prediction.feature_names = [f'intent_{intent}' for intent in intents]
    intent_prediction.feature_categories = ['query' for _ in intents]

    qpp.feature_names = [f'qpp_{x}' for x in qpp.feature_names]
    query_health_classification.feature_names = [f'qhc_{x}' for x in query_health_classification.feature_names]

    return intent_prediction ** qpp ** query_health_classification


def document_features(tira, dataset):
    document_health_classification = tira.pt.doc_features('ir-benchmarks/fschlatt/document-health-classification', dataset)
    genre_mlp_classifier = tira.pt.doc_features('ir-benchmarks/tu-dresden-01/genre-mlp', dataset,
                                                feature_selection=['probability_Discussion', 'probability_Shop', 'probability_Download',
                                                                   'probability_Articles', 'probability_Help', 'probability_Linklists',
                                                                   'probability_Porttrait private', 'probability_Protrait non private'])
    spacy_features = tira.pt.doc_features('ir-benchmarks/tu-dresden-04/spacy-document-features', dataset)

    return document_health_classification ** genre_mlp_classifier ** spacy_features


def get_rerankers(tira, dataset, rerankers, category='reranker'):
    rerankers = [
        TiraNamedFeatureTransformer(tira.pt.from_submission(identifier, dataset), name, category)
        for name, identifier in rerankers.items()
    ]

    result = rerankers[0]
    for reranker in rerankers[1:]:
        result = result ** reranker

    return result


def base_rerankers(tira, dataset):
    return get_rerankers(tira, dataset, {
        'bm25': 'ir-benchmarks/tira-ir-starter/BM25 Re-Rank (tira-ir-starter-pyterrier)',
        'pl2': 'ir-benchmarks/tira-ir-starter/PL2 Re-Rank (tira-ir-starter-pyterrier)',
        'dirichlet_lm': 'ir-benchmarks/tira-ir-starter/DirichletLM Re-Rank (tira-ir-starter-pyterrier)',
        'dlh': 'ir-benchmarks/tira-ir-starter/DLH Re-Rank (tira-ir-starter-pyterrier)',
        'lgd': 'ir-benchmarks/tira-ir-starter/LGD Re-Rank (tira-ir-starter-pyterrier)'
    }, category='lexical')


def llm_rerankers(tira, dataset):
    return get_rerankers(tira, dataset, {
        'rank_zephyr': 'workshop-on-open-web-search/fschlatt/rank-zephyr',
        'sparse_cross_encoder': 'ir-benchmarks/fschlatt/sparse-cross-encoder-4-512',
        'list_in_t5': 'ir-benchmarks/fschlatt/castorini-list-in-t5-150',
        'sbert': 'ir-benchmarks/tira-ir-starter/SBERT multi-qa-mpnet-base-cos-v1 (tira-ir-starter-beir)',
        'monot5': 'ir-benchmarks/tira-ir-starter/MonoT5 Base (tira-ir-starter-gygaggle)',
        'colbert': 'ir-benchmarks/tira-ir-starter/ColBERT Re-Rank (tira-ir-starter-pyterrier)',
        'ance': 'ir-benchmarks/tira-ir-starter/ANCE Base Cosine (tira-ir-starter-beir)'
    }, category='neural')


def keyquery_score_dict(tira, dataset_id, offset, expansion_method, retrieval_model, fb_terms, fb_docs):
    f = tira.get_run_output(f'ir-benchmarks/ows/time-keyquery-offset-{offset}', dataset_id)
    f = pd.read_json(f'{f}/{expansion_method}_{retrieval_model}_{fb_terms}_{fb_docs}.jsonl.gz', lines=True)
    default_value = np.array([0, 0])
    ret = defaultdict(lambda: defaultdict(lambda: default_value))

    for _, row in f.iterrows():
        ret[row['qid']][row['docno']] = np.array([1, row['score']])

    return ret


def keyquery_features(tira, dataset):
    dataset_id = dataset.irds_ref().dataset_id()
    transformer = TiraApplyFeatureTransformer(keyquery_score_dict(tira, dataset_id, '2022', 'bo1', 'BM25', 30, 5), name='keyquery_features')
    return TiraNamedFeatureTransformer(transformer, ['keyquery_exists', 'keyquery_score'], 'keyquery')


def reverted_index_features(tira, dataset):
    reverted_index_of_the_past = construct_reverted_index_of_the_past(tira, dataset)
    transformer = TiraApplyFeatureTransformer(reverted_index_of_the_past, name='reverted_index_features')
    return TiraNamedFeatureTransformer(transformer, ['reverted_index_exists', 'reverted_index_max', 'reverted_index_mean'], 'reverted_index')


def wows_only_features(tira, dataset):
    return query_features(tira, dataset) ** document_features(tira, dataset)


def wows_base_rerank_features(tira, dataset):
    return wows_only_features(tira, dataset) ** base_rerankers(tira, dataset)


def wows_all_rerank_features(tira, dataset):
    return wows_base_rerank_features(tira, dataset) ** llm_rerankers(tira, dataset)


def wows_rerank_and_keyquery_features(tira, dataset):
    return wows_all_rerank_features(tira, dataset) ** keyquery_features(tira, dataset)


def wows_rerank_and_reverted_index_features(tira, dataset):
    return wows_all_rerank_features(tira, dataset) ** reverted_index_features(tira, dataset)


def all_features(tira, dataset):
    return wows_rerank_and_keyquery_features(tira, dataset) ** reverted_index_features(tira, dataset)
