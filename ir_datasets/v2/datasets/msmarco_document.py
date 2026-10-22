"""MS MARCO (document) — a v2 dataset family.

Reuses v1's ``MsMarcoTrecDocs`` (a ``TrecDocs`` subclass reformatting the raw
TREC-SGML-ish records into a 4-field ``MsMarcoDocument``) as the docs parser,
same "thin layer over v1 machinery" principle as every other v2 family --
with one small addition: v1's default docstore path lands next to the
*source*'s cache path (opaque, content-addressed), so ``_MsMarcoDocs`` below
redirects it to ``node.docstore_path`` instead, exactly the fix
``formats._V1TsvDocs`` already applies for ``TsvDocs``.

Not covered here (left for a future pass, same as ``msmarco_passage.py``'s
own noted gaps):

* ``trec-dl-hard`` and its 5 folds -- needs an external fold-id list
  (``DL_HARD_QIDS_BYFOLD``) transcribed from v1.
* ``anchor-text`` -- a second, alternate representation of the same
  documents (anchor text instead of title+body); relating it to the main
  corpus would need a same-corpus-ish edge kind, which doesn't exist yet
  (``same_corpus_as`` was declared in ``nodes.py`` at one point but was
  dropped, unused) -- a good next addition, not this one.

TREC-DL (2019/2020), which judges this corpus (and ``msmarco-passage``'s)
against a shared query set, is not defined here -- see ``trec_dl.py``, which
imports ``docs`` (below) by reference rather than duplicating it.
"""
from ir_datasets.datasets.msmarco_document import MsMarcoDocument, MsMarcoTrecDocs
from ir_datasets.indices import DEFAULT_DOCSTORE_OPTIONS, PickleLz4FullStore

from ir_datasets.v2 import Benchmark, DocTable, QrelTable, Resource, RunTable, TrecQrels, TrecScoredDocs, TsvQueries, irds
from ir_datasets.v2.datasets.msmarco_passage import DUA, MEASURES
from ir_datasets.v2.formats import Parser

CITATION = 'dblp:conf/nips/NguyenRSGTMD16'

QRELS_DEFS = {
    1: 'Document contains a passage labeled as relevant in msmarco-passage',
}

ORCAS_QRELS_DEFS = {
    1: 'User click',
}


class _MsMarcoDocs(MsMarcoTrecDocs):
    """v1's MsMarcoTrecDocs, but with the docstore where the v2 node wants it
    (readable, format-versioned) rather than next to the source's
    content-addressed cache path -- the same fix v2's own ``_V1TsvDocs``
    applies for ``TsvDocs`` (see ``formats.py``)."""

    def __init__(self, docs_dlc, store_path):
        super().__init__(docs_dlc)
        self._store_path = store_path

    def docs_store(self, field='doc_id', options=DEFAULT_DOCSTORE_OPTIONS):
        return PickleLz4FullStore(
            path=str(self._store_path),
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=['doc_id'],
            size_hint=self._docstore_size_hint,
            count_hint=self._count_hint,
            options=options,
        )


class _MsMarcoDocsParser(Parser):
    name = 'MsMarcoTrecDocs'

    def build(self, source, node):
        return _MsMarcoDocs(source, node.docstore_path)


# license verified 2026-09-30: https://microsoft.github.io/msmarco/ (Terms and Conditions: non-commercial research use only; applies to all MS MARCO datasets)
# Files
# -----------------------------------------
docs_file = Resource('msmarco-document-docs.trec.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.trec.gz'],
    hash='md5:d4863e4f342982b51b9a8fc668b2d0c0',
    size=8_501_799_926,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
train_queries_file = Resource('msmarco-document-train-queries.tsv.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz'],
    hash='md5:4086d31a9cf2d7b69c4932609058111d',
    size=6_457_962,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
train_qrels_file = Resource('msmarco-document-train-qrels.tsv.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz'],
    hash='md5:9d1609e240113b0504fd2e61cb36d924',
    size=2_385_717,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
train_scoreddocs_file = Resource('msmarco-document-train-scoreddocs.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz'],
    hash='md5:be32fa12eb71e93014c84775d7465976',
    size=403_564_127,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
dev_queries_file = Resource('msmarco-document-dev-queries.tsv.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz'],
    hash='md5:ac20593d71b9c32ab2633230f9cdf10d',
    size=91_837,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
dev_qrels_file = Resource('msmarco-document-dev-qrels.tsv.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz'],
    hash='md5:5eeafaeb4960979a62e7fed93273254e',
    size=38_553,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
dev_scoreddocs_file = Resource('msmarco-document-dev-scoreddocs.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz'],
    hash='md5:ac10255edf321821b0ccd0f123037780',
    size=5_701_839,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
eval_queries_file = Resource('msmarco-document-eval-queries.tsv.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docleaderboard-queries.tsv.gz'],
    hash='md5:50fe4285d64444c9ffc933b66a79f775',
    size=102_131,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
eval_scoreddocs_file = Resource('msmarco-document-eval-scoreddocs.tsv.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/docleaderboard-top100.tsv.gz'],
    hash='md5:a039a00356c09606962f3c07c68d02ef',
    size=6_362_021,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
orcas_queries_file = Resource('msmarco-document-orcas-queries.tsv.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas-doctrain-queries.tsv.gz'],
    hash='md5:519c5f522294406e3b0574d7d53cf233',
    size=104_209_356,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
orcas_qrels_file = Resource('msmarco-document-orcas-qrels.tsv.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas-doctrain-qrels.tsv.gz'],
    hash='md5:3f94db106374be649782022c3018acd0',
    size=109_824_304,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)
orcas_scoreddocs_file = Resource('msmarco-document-orcas-scoreddocs.gz',
    sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas-doctrain-top100.gz'],
    hash='md5:118d0884638fd405e111157a124ef0b2',
    size=10_724_320_629,
    dua=DUA, license='https://microsoft.github.io/msmarco/',
)

# Tables
# -----------------------------------------
docs = DocTable('msmarco-document',
    source=docs_file.gunzip(),
    parser=_MsMarcoDocsParser(),
    lang='en', license='https://microsoft.github.io/msmarco/',
)

train_queries = TsvQueries('msmarco-document-train-queries', source=train_queries_file.gunzip(), lang='en', license='https://microsoft.github.io/msmarco/')
train_qrels = TrecQrels('msmarco-document-train-qrels', source=train_qrels_file.gunzip(), defs=QRELS_DEFS, license='https://microsoft.github.io/msmarco/')
train_scoreddocs = TrecScoredDocs('msmarco-document-train-scoreddocs', source=train_scoreddocs_file.gunzip(), license='https://microsoft.github.io/msmarco/')

dev_queries = TsvQueries('msmarco-document-dev-queries', source=dev_queries_file.gunzip(), lang='en', license='https://microsoft.github.io/msmarco/')
dev_qrels = TrecQrels('msmarco-document-dev-qrels', source=dev_qrels_file.gunzip(), defs=QRELS_DEFS, license='https://microsoft.github.io/msmarco/')
dev_scoreddocs = TrecScoredDocs('msmarco-document-dev-scoreddocs', source=dev_scoreddocs_file.gunzip(), license='https://microsoft.github.io/msmarco/')

eval_queries = TsvQueries('msmarco-document-eval-queries', source=eval_queries_file.gunzip(), lang='en', license='https://microsoft.github.io/msmarco/')
eval_scoreddocs = TrecScoredDocs('msmarco-document-eval-scoreddocs', source=eval_scoreddocs_file.gunzip(), license='https://microsoft.github.io/msmarco/')

orcas_queries = TsvQueries('msmarco-document-orcas-queries', source=orcas_queries_file.gunzip(), lang='en', license='https://microsoft.github.io/msmarco/')
orcas_qrels = TrecQrels('msmarco-document-orcas-qrels', source=orcas_qrels_file.gunzip(), defs=ORCAS_QRELS_DEFS, license='https://microsoft.github.io/msmarco/')
orcas_scoreddocs = TrecScoredDocs('msmarco-document-orcas-scoreddocs', source=orcas_scoreddocs_file.gunzip(), license='https://microsoft.github.io/msmarco/')

# Benchmarks
# -----------------------------------------
train = Benchmark('msmarco-document-train',
    docs=docs, queries=train_queries, qrels=train_qrels, scoreddocs=train_scoreddocs,
    citation=CITATION, metrics=MEASURES,
    license='https://microsoft.github.io/msmarco/')
dev = Benchmark('msmarco-document-dev',
    docs=docs, queries=dev_queries, qrels=dev_qrels, scoreddocs=dev_scoreddocs,
    citation=CITATION, metrics=MEASURES,
    license='https://microsoft.github.io/msmarco/')
eval_ = Benchmark('msmarco-document-eval',
    docs=docs, queries=eval_queries, scoreddocs=eval_scoreddocs,
    citation=CITATION, metrics=MEASURES,
    license='https://microsoft.github.io/msmarco/')
orcas = Benchmark('msmarco-document-orcas',
    docs=docs, queries=orcas_queries, qrels=orcas_qrels, scoreddocs=orcas_scoreddocs,
    citation=CITATION, metrics=MEASURES,
    license='https://microsoft.github.io/msmarco/')


# Registration
# -----------------------------------------
irds.register(
    train, dev, eval_, orcas,
)
