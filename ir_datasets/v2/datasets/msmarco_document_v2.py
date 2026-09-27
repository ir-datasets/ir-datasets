"""MS MARCO (document) v2 -- a v2 dataset family.

Same corpus-revision naming caveat as ``msmarco_passage_v2.py``: the "v2"
here is MS MARCO's own, not this package's.

Reuses v1's ``MsMarcoV2Docs`` as the docs parser, same "thin layer over v1
machinery" principle as every other v2 family -- with the same
docstore-path redirect ``msmarco_document.py``'s ``_MsMarcoDocs`` applies
(v1's default docstore path here is already readable/versioned, so no
redirect is actually needed -- see the note on ``docs_store`` below).

TREC-DL 2021-2023 (which judge this corpus) is not defined here -- see
``trec_dl.py``, which imports ``docs`` (below) by reference rather than
duplicating it, same as it already does for the v1 corpus's own 2019/2020
years. TREC-DL 2019/2020 remapped onto *this* v2 corpus (as v1 offered under
``msmarco-document-v2/trec-dl-2019`` etc.) is dropped rather than migrated:
it would just be a second, redundant mapping of the same qrels onto a
different corpus revision, alongside ``trec_dl.py``'s existing v1-corpus
2019/2020 benchmarks.

Not covered here (left for a future pass, same spirit as
``msmarco_passage_v2.py``'s own noted gap): ``anchor-text`` -- a second,
alternate representation of the same documents, same reason it's skipped in
``msmarco_document.py`` (no ``same_corpus_as``-ish edge kind exists yet to
relate it to the main corpus).
"""
from ir_datasets.datasets.msmarco_document_v2 import MsMarcoV2Docs

from ir_datasets.v2 import Benchmark, DocTable, Resource, Source, TrecQrels, TrecScoredDocs, TsvQueries, irds
from ir_datasets.v2.datasets.msmarco_document import DUA
from ir_datasets.v2.formats import Parser

CITATION = 'dblp:conf/nips/NguyenRSGTMD16'
MEASURES = ['RR@10']

QRELS_DEFS = {
    1: 'Document contains a passage labeled as relevant in msmarco-passage-v2',
}

_MS_HEADERS = {'X-Ms-Version': '2019-12-12'}


class _MsMarcoV2DocsParser(Parser):
    name = 'MsMarcoV2Docs'

    def build(self, source, node):
        # v1's own docs_store() already builds a PickleLz4FullStore at
        # ``<source path>.pklz4`` -- an opaque, content-addressed location
        # once ``source`` is a v2 Resource. Redirecting it to
        # ``node.docstore_path`` would need overriding docs_store() too (as
        # ``_MsMarcoDocs`` does in msmarco_document.py); skipped here since
        # this docstore is large (66GB+) and any existing v1 cache at the
        # legacy path should keep being reused rather than rebuilt -- a
        # correctness/disk-space tradeoff, not an oversight.
        return MsMarcoV2Docs(source)


with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    docs_file = Resource('msmarco-document-v2-docs.tar',
        sources=[Source('https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_doc.tar', headers=_MS_HEADERS)],
        md5='eea90100409a254fdb157b8e4e349deb',
        size=34_648_862_720,
    )
    train_queries_file = Resource('msmarco-document-v2-train-queries.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_queries.tsv'],
        md5='7821d8bef3971e12780a80a89a3e5cbd',
        size=13_511_656,
    )
    train_qrels_file = Resource('msmarco-document-v2-train-qrels.txt',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_qrels.tsv'],
        md5='2f788d031c2ca29c4c482167fa5966de',
        size=12_450_533,
    )
    train_scoreddocs_file = Resource('msmarco-document-v2-train-scoreddocs.txt.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_top100.txt.gz'],
        md5='b4d5915172d5f54bd23c31e966c114de',
        size=424_107_669,
    )
    dev1_queries_file = Resource('msmarco-document-v2-dev1-queries.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_queries.tsv'],
        md5='b05dc19f1d2b8ad729f189328a685aa1',
        size=191_992,
    )
    dev1_qrels_file = Resource('msmarco-document-v2-dev1-qrels.txt',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_qrels.tsv'],
        md5='aad92d731892ccb0cf9c4c2e37e0f0f1',
        size=177_593,
    )
    dev1_scoreddocs_file = Resource('msmarco-document-v2-dev1-scoreddocs.txt.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_top100.txt.gz'],
        md5='4dd27d511748bede545cd7ae3fc92bf4',
        size=5_830_666,
    )
    dev2_queries_file = Resource('msmarco-document-v2-dev2-queries.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_queries.tsv'],
        md5='f000319f1893a7acdd60fdcae0703b95',
        size=209_911,
    )
    dev2_qrels_file = Resource('msmarco-document-v2-dev2-qrels.txt',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_qrels.tsv'],
        md5='f2eead4b192683ae5fbd66f4d3f08b96',
        size=195_474,
    )
    dev2_scoreddocs_file = Resource('msmarco-document-v2-dev2-scoreddocs.txt.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_top100.txt.gz'],
        md5='e03b5404e9027569c1aa794b1408d8a5',
        size=6_412_563,
    )
    # Tables
    # -----------------------------------------
    docs = DocTable('msmarco-document-v2-docs',
        source=docs_file,
        parser=_MsMarcoV2DocsParser(),
    )

    train_queries = TsvQueries('msmarco-document-v2-train-queries', source=train_queries_file)
    train_qrels = TrecQrels('msmarco-document-v2-train-qrels', source=train_qrels_file, defs=QRELS_DEFS)
    train_scoreddocs = TrecScoredDocs('msmarco-document-v2-train-scoreddocs', source=train_scoreddocs_file.gunzip())

    dev1_queries = TsvQueries('msmarco-document-v2-dev1-queries', source=dev1_queries_file)
    dev1_qrels = TrecQrels('msmarco-document-v2-dev1-qrels', source=dev1_qrels_file, defs=QRELS_DEFS)
    dev1_scoreddocs = TrecScoredDocs('msmarco-document-v2-dev1-scoreddocs', source=dev1_scoreddocs_file.gunzip())

    dev2_queries = TsvQueries('msmarco-document-v2-dev2-queries', source=dev2_queries_file)
    dev2_qrels = TrecQrels('msmarco-document-v2-dev2-qrels', source=dev2_qrels_file, defs=QRELS_DEFS)
    dev2_scoreddocs = TrecScoredDocs('msmarco-document-v2-dev2-scoreddocs', source=dev2_scoreddocs_file.gunzip())

    # Benchmarks
    # -----------------------------------------
    train = Benchmark('msmarco-document-v2-train',
        docs=docs, queries=train_queries, qrels=train_qrels, scoreddocs=train_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='Official train set for the MS MARCO v2 document corpus.')
    dev1 = Benchmark('msmarco-document-v2-dev1',
        docs=docs, queries=dev1_queries, qrels=dev1_qrels, scoreddocs=dev1_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='Official dev set for the MS MARCO v2 document corpus.')
    dev2 = Benchmark('msmarco-document-v2-dev2',
        docs=docs, queries=dev2_queries, qrels=dev2_qrels, scoreddocs=dev2_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='A second, held-out dev set for the MS MARCO v2 document corpus.')


# Registration
# -----------------------------------------
irds.register(
    train, dev1, dev2,
)
