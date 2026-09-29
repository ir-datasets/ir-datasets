"""MS MARCO (passage) v2 -- a v2 dataset family.

The "v2" here is MS MARCO's own corpus revision (a bigger, re-segmented
passage collection released for the TREC Deep Learning 2021+ tracks), not
this package's naming -- confusing, but it's their name, not ours.

Reuses v1's ``MsMarcoV2Passages`` as the docs parser, same "thin layer over
v1 machinery" principle as every other v2 family -- with the same fix
``msmarco_document.py``'s ``_MsMarcoDocs`` and ``formats._V1TsvDocs`` already
apply: v1's default docstore/extraction path lands next to the *source*'s
cache path (opaque, content-addressed), so ``_MsMarcoV2Passages`` below
redirects it to ``node.docstore_path`` instead.

TREC-DL 2021-2023 (which judge this corpus) is not defined here -- see
``trec_dl.py``, which imports ``docs`` (below) by reference rather than
duplicating it.

Not covered here (left for a future pass): the ``dedup`` subset (the
collection restricted to positions from a de-duplicated id list) -- it's a
corpus variant, not a benchmark, and would need a second docs Table sharing
the same source but a different position filter.
"""
from ir_datasets.datasets.msmarco_passage_v2 import MsMarcoV2Passages

from ir_datasets.v2 import Benchmark, DocTable, Resource, Source, TrecQrels, TrecScoredDocs, TsvQueries, irds
from ir_datasets.v2.datasets.msmarco_passage import DUA
from ir_datasets.v2.formats import Parser

CITATION = 'dblp:conf/nips/NguyenRSGTMD16'
MEASURES = ['RR@10']

QRELS_DEFS = {
    1: 'Based on mapping from v1 of MS MARCO',
}

_MS_HEADERS = {'X-Ms-Version': '2019-12-12'}


class _MsMarcoV2Passages(MsMarcoV2Passages):
    """v1's MsMarcoV2Passages, but with its on-disk extracted-file cache
    redirected to ``node.docstore_path``'s base (readable, format-versioned)
    instead of next to the source's own content-addressed cache path."""

    def __init__(self, dlc, store_path):
        super().__init__(dlc)
        self._store_path = store_path

    def docs_path(self, force=True):
        return str(self._store_path)


class _MsMarcoV2PassagesParser(Parser):
    name = 'MsMarcoV2Passages'

    def build(self, source, node):
        return _MsMarcoV2Passages(source, node.docstore_path)


with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    passages_file = Resource('msmarco-passage-v2-passages.tar',
        sources=[Source('https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar', headers=_MS_HEADERS)],
        md5='05946bac48a8ffee62e160213eab3fda',
        size=21_768_192_000,
    )
    train_queries_file = Resource('msmarco-passage-v2-train-queries.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_queries.tsv'],
        md5='1835f44e6792c51aa98eed722a8dcc11',
        size=11_608_838,
    )
    train_qrels_file = Resource('msmarco-passage-v2-train-qrels.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_qrels.tsv'],
        md5='a2e37e9a9c7ca13d6e38be0512a52017',
        size=11_620_946,
    )
    train_scoreddocs_file = Resource('msmarco-passage-v2-train-scoreddocs.txt.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_top100.txt.gz'],
        md5='7cd731ed984fccb2396f11a284cea800',
        size=340_634_991,
    )
    dev1_queries_file = Resource('msmarco-passage-v2-dev1-queries.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_queries.tsv'],
        md5='0fa4c6d64a653142ade9fc61d7484239',
        size=164_507,
    )
    dev1_qrels_file = Resource('msmarco-passage-v2-dev1-qrels.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_qrels.tsv'],
        md5='10f9263260d206d8fb8f13864aea123a',
        size=165_024,
    )
    dev1_scoreddocs_file = Resource('msmarco-passage-v2-dev1-scoreddocs.txt.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_top100.txt.gz'],
        md5='fee817a3ee273be8623379e5d3108c0b',
        size=4_882_727,
    )
    dev2_queries_file = Resource('msmarco-passage-v2-dev2-queries.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_queries.tsv'],
        md5='565b84dfa7ccd2f4251fa2debea5947a',
        size=179_603,
    )
    dev2_qrels_file = Resource('msmarco-passage-v2-dev2-qrels.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_qrels.tsv'],
        md5='8ed8577fa459d34b59cf69b4daa2baeb',
        size=181_612,
    )
    dev2_scoreddocs_file = Resource('msmarco-passage-v2-dev2-scoreddocs.txt.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_top100.txt.gz'],
        md5='da532bf26169a3a2074fae774471cc9f',
        size=5_355_464,
    )

    # Tables
    # -----------------------------------------
    docs = DocTable('msmarco-passage-v2',
        source=passages_file,
        parser=_MsMarcoV2PassagesParser(),
    )

    train_queries = TsvQueries('msmarco-passage-v2-train-queries', source=train_queries_file)
    train_qrels = TrecQrels('msmarco-passage-v2-train-qrels', source=train_qrels_file, defs=QRELS_DEFS)
    train_scoreddocs = TrecScoredDocs('msmarco-passage-v2-train-scoreddocs', source=train_scoreddocs_file.gunzip())

    dev1_queries = TsvQueries('msmarco-passage-v2-dev1-queries', source=dev1_queries_file)
    dev1_qrels = TrecQrels('msmarco-passage-v2-dev1-qrels', source=dev1_qrels_file, defs=QRELS_DEFS)
    dev1_scoreddocs = TrecScoredDocs('msmarco-passage-v2-dev1-scoreddocs', source=dev1_scoreddocs_file.gunzip())

    dev2_queries = TsvQueries('msmarco-passage-v2-dev2-queries', source=dev2_queries_file)
    dev2_qrels = TrecQrels('msmarco-passage-v2-dev2-qrels', source=dev2_qrels_file, defs=QRELS_DEFS)
    dev2_scoreddocs = TrecScoredDocs('msmarco-passage-v2-dev2-scoreddocs', source=dev2_scoreddocs_file.gunzip())

    # Benchmarks
    # -----------------------------------------
    train = Benchmark('msmarco-passage-v2-train',
        docs=docs, queries=train_queries, qrels=train_qrels, scoreddocs=train_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='Official train set for the MS MARCO v2 passage corpus.')
    dev1 = Benchmark('msmarco-passage-v2-dev1',
        docs=docs, queries=dev1_queries, qrels=dev1_qrels, scoreddocs=dev1_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='Official dev set for the MS MARCO v2 passage corpus.')
    dev2 = Benchmark('msmarco-passage-v2-dev2',
        docs=docs, queries=dev2_queries, qrels=dev2_qrels, scoreddocs=dev2_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='A second, held-out dev set for the MS MARCO v2 passage corpus.')


# Registration
# -----------------------------------------
irds.register(
    train, dev1, dev2,
)
