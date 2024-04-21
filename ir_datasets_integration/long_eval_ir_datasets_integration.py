"""This python file registers new ir_datasets classes for 'longeval'.
   You can find the ir_datasets documentation here: https://github.com/allenai/ir_datasets/.
   This file is intended to work inside the Docker image.
"""
import ir_datasets
from ir_datasets.formats import TrecDocs, TrecQueries, TrecQrels, TsvQueries
from typing import NamedTuple, Dict
from ir_datasets.datasets.base import Dataset
from ir_datasets.util import LocalDownload
from ir_datasets.indices import PickleLz4FullStore
from pathlib import Path

class LongEvalDocs(TrecDocs):
    def __init__(self, dlc):
        self._dlc = LocalDownload(Path(dlc))
        super().__init__(self._dlc)

    def docs_store(self):
        return PickleLz4FullStore(
            path=f'{self._dlc.path(force=False)}.pklz4',
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field='doc_id',
            index_fields=['doc_id'],
        )

ir_datasets.registry.register('longeval/2023-06', Dataset(
    LongEvalDocs('/root/.ir_datasets/longeval/test-collection/2023_06/English/Documents/Trec/'),
    TsvQueries(LocalDownload(Path('/root/.ir_datasets/longeval/test-collection/2023_06/English/Queries/test.tsv')))
))

ir_datasets.registry.register('longeval/2023-08', Dataset(
    LongEvalDocs('/root/.ir_datasets/longeval/test-collection/2023_08/English/Documents/Trec/'),
    TsvQueries(LocalDownload(Path('/root/.ir_datasets/longeval/test-collection/2023_08/English/Queries/test.tsv')))
))
