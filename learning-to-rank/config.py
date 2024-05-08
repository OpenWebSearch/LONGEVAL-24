from feature_extraction import *


MODELS = {
    'wows_only': wows_only_features,
    'wows_base_rerank': wows_base_rerank_features,
    'wows_all_rerank': wows_all_rerank_features,
    'wows_rerank_and_keyquery': wows_rerank_and_keyquery_features,
    'wows_rerank_and_reverted_index': wows_rerank_and_reverted_index_features,
    'all': all_features,
}

DATASETS = {
    'train_2023': 'longeval-train-20230513-training', 
    'WT':         'longeval-heldout-20230513-training',
    'ST':         'longeval-short-july-20230513-training',
    'LT':         'longeval-long-september-20230513-training',
    'train_2024': 'longeval-2023-01-20240423-training',
    'lag6':       'longeval-2023-06-20240418-training',
    'lag8':       'longeval-2023-08-20240418-training',
}

AVAILABLE_DATASETS_PER_MODEL = {
    'wows_only': list(DATASETS.keys()),
    'wows_base_rerank': list(DATASETS.keys()),
    'wows_all_rerank': ['train_2023', 'WT', 'train_2024', 'lag6', 'lag8'],
    'wows_rerank_and_keyquery': ['train_2024', 'lag6', 'lag8'],
    'wows_rerank_and_reverted_index': ['train_2024', 'lag6', 'lag8'],
    'all': ['train_2024', 'lag6', 'lag8'],
}
