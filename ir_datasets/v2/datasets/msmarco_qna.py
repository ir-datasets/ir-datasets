"""MS MARCO (QnA) -- a v2 dataset family.

Unlike every other v2 family so far, this one can't be a thin declarative
wrapper: v1's ``MsMarcoQnAManager`` builds the docs docstore *and* every
split's queries/qrels/run files together, in one pass over the train/dev/eval
source files (doc ids are assigned by hashing passage text against
``msmarco-passage``, so there's no way to derive one split's files without
building the others -- see the manager's own docstring in
``ir_datasets/datasets/msmarco_qna.py``). So this file reuses that manager
directly rather than re-deriving its logic, and wires each derived file
through ``manager.file_ref(...)`` -- an object with the same ``.stream()``/
``.path()`` shape a v2 ``source=`` expects, so ``TsvQueries``/``TrecQrels``/
``TrecScoredDocs`` work unmodified. Only the docs Table needs a custom
``Parser`` (below), since its handler is ``manager.docs_store`` directly
rather than something built through the source+parser pipeline.

The ``train``/``dev``/``eval`` Resources are still declared and passed as
``source=`` to the docs Table below (as a list) purely so the graph records
the right ``derived_from`` edges to them; the manager reads them through its
own dlc-shaped wrapper (``.gunzip()`` on each), not through the Table's
``source`` attribute.
"""
from ir_datasets.datasets.msmarco_qna import (
    MsMarcoQnADoc, MsMarcoQnAEvalQuery, MsMarcoQnAManager, MsMarcoQnAQuery,
)
from ir_datasets.formats import DocstoreBackedDocs

from ir_datasets.v2 import Benchmark, DocTable, Resource, TrecQrels, TrecScoredDocs, TsvQueries, irds
from ir_datasets.v2.datasets.msmarco_passage import DUA
from ir_datasets.v2.formats import Parser
from ir_datasets.v2.sources import default_cache_path

CITATION = 'dblp:conf/nips/NguyenRSGTMD16'

QRELS_DEFS = {
    1: 'Marked by annotator as a contribution to their answer',
    0: 'Not marked by annotator as a contribution to their answer',
}


class _QnADocsParser(Parser):
    name = 'MsMarcoQnADocs'

    def __init__(self, manager):
        self.manager = manager

    def build(self, source, node):
        return DocstoreBackedDocs(self.manager.docs_store, docs_cls=MsMarcoQnADoc,
                                  namespace=node.name, lang=node.lang)


# license verified 2026-09-30: https://microsoft.github.io/msmarco/ (Terms and Conditions: non-commercial research use only; applies to all MS MARCO datasets)
with irds.defaults(dua=DUA, lang='en', license='https://microsoft.github.io/msmarco/'):
    # Files
    # -----------------------------------------
    train_file = Resource('msmarco-qna-train.json.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarco/train_v2.1.json.gz'],
        hash='md5:576230a745a06943c3a49e76acea1d9d',
        size=1_112_116_929,
    )
    dev_file = Resource('msmarco-qna-dev.json.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarco/dev_v2.1.json.gz'],
        hash='md5:5e14839f31c933560fbb3bae4ce67829',
        size=138_303_699,
    )
    eval_file = Resource('msmarco-qna-eval.json.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarco/eval_v2.1_public.json.gz'],
        hash='md5:5fcca9336c7486498c3e1cf81fa89f74',
        size=133_851_237,
    )

    # The manager needs a directory to write its derived per-split files
    # (queries.tsv/qrels/run) and the shared docs docstore into -- the same
    # readable, provider-partitioned base every other v2 family's docstore
    # uses (see ``nodes.DocTable.docstore_path``), computed directly (rather
    # than via ``docs.docstore_path`` below) since the manager must exist
    # before ``docs`` -- its Parser needs ``manager`` already built.
    _base_path = default_cache_path(irds.prefix, 'msmarco-qna')
    _base_path.mkdir(parents=True, exist_ok=True)
    manager = MsMarcoQnAManager(train_file.gunzip(), dev_file.gunzip(), eval_file.gunzip(), _base_path)

    # Tables
    # -----------------------------------------
    docs = DocTable('msmarco-qna',
        source=[train_file, dev_file, eval_file],
        parser=_QnADocsParser(manager),
    )

    train_queries = TsvQueries('msmarco-qna-train-queries', cls=MsMarcoQnAQuery,
        source=manager.file_ref('train.queries.tsv'))
    train_qrels = TrecQrels('msmarco-qna-train-qrels',
        source=manager.file_ref('train.qrels'), defs=QRELS_DEFS)
    train_scoreddocs = TrecScoredDocs('msmarco-qna-train-scoreddocs',
        source=manager.file_ref('train.run'))

    dev_queries = TsvQueries('msmarco-qna-dev-queries', cls=MsMarcoQnAQuery,
        source=manager.file_ref('dev.queries.tsv'))
    dev_qrels = TrecQrels('msmarco-qna-dev-qrels',
        source=manager.file_ref('dev.qrels'), defs=QRELS_DEFS)
    dev_scoreddocs = TrecScoredDocs('msmarco-qna-dev-scoreddocs',
        source=manager.file_ref('dev.run'))

    eval_queries = TsvQueries('msmarco-qna-eval-queries', cls=MsMarcoQnAEvalQuery,
        source=manager.file_ref('eval.queries.tsv'))
    eval_scoreddocs = TrecScoredDocs('msmarco-qna-eval-scoreddocs',
        source=manager.file_ref('eval.run'))

    # Benchmarks
    # -----------------------------------------
    train = Benchmark('msmarco-qna-train',
        docs=docs, queries=train_queries, qrels=train_qrels, scoreddocs=train_scoreddocs,
        citation=CITATION,
        desc='Official train set. Queries are natural-language questions with '
             'free-text answers; qrels/scoreddocs mark which passages the '
             "annotator selected as contributing to the answer.")
    dev = Benchmark('msmarco-qna-dev',
        docs=docs, queries=dev_queries, qrels=dev_qrels, scoreddocs=dev_scoreddocs,
        citation=CITATION,
        desc='Official dev set.')
    eval_ = Benchmark('msmarco-qna-eval',
        docs=docs, queries=eval_queries, scoreddocs=eval_scoreddocs,
        citation=CITATION,
        desc='Official eval set for the MS MARCO QnA leaderboard (qrels hidden).')


# Registration
# -----------------------------------------
irds.register(
    train, dev, eval_,
)
