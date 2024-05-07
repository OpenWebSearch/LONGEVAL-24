from collections import defaultdict

import numpy as np
from tira.ir_datasets_util import translate_irds_id_to_tirex
from statistics import mean

def construct_reverted_index_of_the_past(tira, inference_dataset, oracle_datasets=('longeval-train-20230513-training', 'longeval-heldout-20230513-training', 'longeval-short-july-20230513-training', 'longeval-long-september-20230513-training')):
    """constructs a "reverted index of the past". Attention, with the default oracle_datasets configuration, this can only be applied to the 2023 versions of LongEval. I.e., in this setting, we use all of the 2022 data for the construction of the reverted index, this means, we can only use these features on the 2023 data of LongEval. """
    similar_documents = get_similar_documents(tira, inference_dataset, oracle_datasets)
    similar_documents_dict = __query_to_similar_docs(similar_documents)

    qid_to_doc_to_score = {}
    for i in inference_dataset.irds_ref().queries_iter():
        query = __normalize_queries(i.text)
        qid = i.query_id
        if query not in similar_documents_dict:
            continue

        if qid not in qid_to_doc_to_score:
             qid_to_doc_to_score[qid] = {}

        for j in similar_documents_dict[query]:
            for hit in j['top_bm25_results']:
                docno = hit['docno']
                if docno not in qid_to_doc_to_score[qid]:
                    qid_to_doc_to_score[qid][docno] = []
            
                qid_to_doc_to_score[qid][docno] += [hit['score']]
    
    default_value = np.array([0, 0, 0])
    ret = defaultdict(lambda: defaultdict(lambda: default_value))
    for qid in qid_to_doc_to_score:
        for docno in qid_to_doc_to_score[qid]:
            ret[qid][docno] = np.array([1, max(qid_to_doc_to_score[qid][docno]), mean(qid_to_doc_to_score[qid][docno])])
    
    return ret

def __query_to_similar_docs(sim_docs):
    ret = {}

    for i in sim_docs:
        if i['relevance'] < 1:
            continue
        query = __normalize_queries(i['query'])
        if query not in ret:
            ret[query] = []
        ret[query] += [i]

    return ret

def get_similar_documents(tira, inference_dataset, oracle_datasets=('longeval-train-20230513-training', 'longeval-heldout-20230513-training', 'longeval-short-july-20230513-training', 'longeval-long-september-20230513-training')):
    """
    Attention, with the default oracle_datasets configuration, this can only be applied to the 2023 versions of LongEval. I.e., in this setting, we use all of the 2022 data for the oracle construction.
    """
    import gzip
    import json
    ret = []
    
    file_name = tira.get_run_output('ir-benchmarks/ows/isobaric-warbler', translate_irds_id_to_tirex(inference_dataset)) + '/corpus-graph-over-time.jsonl.gz'
    with gzip.open(file_name, 'rt') as f:
        for l in f:
            l = json.loads(l)
            if oracle_datasets and l['dataset'] in oracle_datasets:
                ret.append(l)
    return ret

def __normalize_queries(q):
    return q.lower().strip()

def get_overlapping_queries(dataset, sim_docs):
    overlapping_queries = set([__normalize_queries(i['query']) for i in sim_docs])
    irds_dataset = dataset.irds_ref()
    queries = {i.query_id:i.text for i in irds_dataset.queries_iter()}
    return {v:k for k,v in queries.items() if __normalize_queries(v) in overlapping_queries}
