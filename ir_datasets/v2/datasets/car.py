"""TREC CAR (Complex Answer Retrieval) -- a v2 dataset family.

Reuses v1's ``CarDocs``/``CarQueries`` handler classes directly as v2
``parser=`` wrappers, same "thin layer over v1 machinery" every other v2
family uses (see ``beir.py``) -- both classes only ever call ``.stream()`` on
what they're given, and a v2 Source pipeline satisfies that interface just
as well as a v1 download pipeline. ``CarDocs`` is subclassed locally
(``_CarDocs``) purely to point its docstore at ``node.docstore_path`` instead
of v1's fixed ``<home>/car/docs.pklz4`` -- v1 reuses that one path for both
the v1.5 and v2.0 corpora (whichever was read last "wins" the cache), which
is fine for v1's single-``NAME`` registry but would silently corrupt v2's,
where both are live nodes at once.

The corpus is huge (5GB+ compressed) and not mirrored by ir_datasets, so
docs sources are the upstream URL only, no ``Source.mirror()`` fallback.
``.member(path, compression='xz')`` extracts a member straight out of each
tar.xz without needing v1's ``ReTar`` repacking step -- multiple members of
the same big ``train.tar.xz`` (one pair per fold) each re-stream the already
locally-cached archive, not a re-download.

``trec-y1`` and its ``manual``/``auto`` qrel variants share one queries table;
the plain ``trec-y1`` benchmark has no qrels of its own (same as v1) and
exists mainly so the shared queries table has one visible home.
"""
from ir_datasets.datasets.car import CarDocs as _V1CarDocs, CarQueries as _V1CarQueries
from ir_datasets.indices import DEFAULT_DOCSTORE_OPTIONS, PickleLz4FullStore
from ir_datasets.v2 import Benchmark, DocTable, QueryTable, Resource, TrecQrels, irds
from ir_datasets.v2.formats import Parser

CITATION_CORPUS = 'Dietz2017Car'
CITATION_Y1 = 'dblp:conf/trec/DietzVRC17'
CITATION_TEST200 = 'dblp:conf/ictir/NanniMMD17'

AUTO_QRELS = {
    1: 'Paragraph appears under heading',
}

MANUAL_QRELS = {
    3: 'MUST be mentioned',
    2: 'SHOULD be mentioned',
    1: 'CAN be mentioned',
    0: 'Non-relevant, but roughly on TOPIC',
    -1: 'NO, non-relevant',
    -2: 'Trash',
}


class _CarDocs(_V1CarDocs):
    """v1 CarDocs, but with the docstore where the node wants it (see the
    module docstring -- v1's fixed path collides between v1.5 and v2.0)."""

    def __init__(self, *args, store_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._store_path = store_path

    def docs_store(self, field='doc_id', options=DEFAULT_DOCSTORE_OPTIONS):
        return PickleLz4FullStore(
            path=str(self._store_path),
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=['doc_id'],
            count_hint=self._count_hint,
            options=options,
        )


class _CarDocsParser(Parser):
    name = 'CarDocs'

    def build(self, source, node):
        return _CarDocs(source, count_hint=node.count_hint, store_path=node.docstore_path)


class _CarQueriesParser(Parser):
    name = 'CarQueries'

    def build(self, source, node):
        return _V1CarQueries(source)


# license verified 2026-09-30: https://trec-car.cs.unh.edu/ footer (TREC-CAR Dataset, CC BY-SA 3.0; based on Wikipedia)
with irds.defaults(license='CC-BY-SA-3.0'):
    # Files
    # -----------------------------------------
    docs_v15_file = Resource('car-v1.5-docs.tar.xz',
        sources=['http://trec-car.cs.unh.edu/datareleases/v1.5/paragraphcorpus-v1.5.tar.xz'],
        hash='md5:4d006dd67cbc11541ed7f87b875cb990',
        size=5_114_258_812,
    )
    docs_v20_file = Resource('car-v2.0-docs.tar.xz',
        sources=['http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz'],
        hash='md5:a404e9256d763ddcacc3da1e34de466a',
        size=5_085_726_092,
    )
    trec_y1_queries_file = Resource('car-trec-y1-queries.tar.xz',
        sources=['http://trec-car.cs.unh.edu/datareleases/v1.5/benchmarkY1test.public-v1.5.tar.xz'],
        hash='md5:6ab490517accd2a2cb4848c0f160bc8d',
        size=40_508,
    )
    trec_y1_qrels_file = Resource('car-trec-y1-qrels.tar.gz',
        sources=['http://trec-car.cs.unh.edu/datareleases/v1.5/trec-car-2017-qrels.tar.gz'],
        hash='md5:1ab7cf01c341757af1bb3db2aedd020f',
        size=4_334_569,
    )
    test200_file = Resource('car-test200.tar.xz',
        sources=['http://trec-car.cs.unh.edu/datareleases/v1.5/test200-v1.5.tar.xz'],
        hash='md5:a7d8ea41f933b2ef49f06d782e908d13',
        size=1_307_336,
    )
    train_file = Resource('car-train.tar.xz',
        sources=['http://trec-car.cs.unh.edu/datareleases/v1.5/train-v1.5.tar.xz'],
        hash='md5:70eb3cf1d9358614f9d96dcd2565dc2b',
        size=2_591_721_692,
    )

    # Tables
    # -----------------------------------------
    docs_v15 = DocTable('car-v1.5-docs',
        source=docs_v15_file.member('paragraphcorpus/paragraphcorpus.cbor', compression='xz'),
        parser=_CarDocsParser(),
        lang='en',
        count_hint=29_678_367,
        citation=CITATION_CORPUS,
    )
    docs_v20 = DocTable('car-v2.0-docs',
        source=docs_v20_file.member('paragraphCorpus/dedup.articles-paragraphs.cbor', compression='xz'),
        parser=_CarDocsParser(),
        lang='en',
        count_hint=29_794_697,
        citation=CITATION_CORPUS,
    )

    trec_y1_queries = QueryTable('car-trec-y1-queries',
        source=trec_y1_queries_file.member('benchmarkY1test.public/test.benchmarkY1test.cbor.outlines', compression='xz'),
        parser=_CarQueriesParser(),
        lang='en',
        count_hint=2_287,
        citation=CITATION_Y1,
    )
    trec_y1_manual_qrels = TrecQrels('car-trec-y1-manual-qrels',
        source=trec_y1_qrels_file.member('TREC_CAR_2017_qrels/manual.benchmarkY1test.cbor.hierarchical.qrels'),
        defs=MANUAL_QRELS, count_hint=29_571, citation=CITATION_Y1)
    trec_y1_auto_qrels = TrecQrels('car-trec-y1-auto-qrels',
        source=trec_y1_qrels_file.member('TREC_CAR_2017_qrels/automatic.benchmarkY1test.cbor.hierarchical.qrels'),
        defs=AUTO_QRELS, count_hint=5_820, citation=CITATION_Y1)

    test200_queries = QueryTable('car-test200-queries',
        source=test200_file.member('test200/train.test200.cbor.outlines', compression='xz'),
        parser=_CarQueriesParser(),
        lang='en',
        count_hint=1_987)
    test200_qrels = TrecQrels('car-test200-qrels',
        source=test200_file.member('test200/train.test200.cbor.hierarchical.qrels', compression='xz'),
        defs=AUTO_QRELS, count_hint=4_706)

    train_folds = {}
    _FOLD_COUNTS = {
        'fold0': (467_946, 1_054_369),
        'fold1': (466_596, 1_052_398),
        'fold2': (469_323, 1_061_162),
        'fold3': (463_314, 1_046_784),
        'fold4': (468_789, 1_061_911),
    }
    for _fold, (_qcount, _rcount) in _FOLD_COUNTS.items():
        _queries = QueryTable(f'car-train-{_fold}-queries',
            source=train_file.member(f'train/train.{_fold}.cbor.outlines', compression='xz'),
            parser=_CarQueriesParser(),
            lang='en',
            count_hint=_qcount)
        _qrels = TrecQrels(f'car-train-{_fold}-qrels',
            source=train_file.member(f'train/train.{_fold}.cbor.hierarchical.qrels', compression='xz'),
            defs=AUTO_QRELS, count_hint=_rcount)
        train_folds[_fold] = (_queries, _qrels)

    # Benchmarks
    # -----------------------------------------
    trec_y1 = Benchmark('car-trec-y1',
        docs=docs_v15, queries=trec_y1_queries,
        desc='TREC CAR 2017 (Y1) benchmark test queries, over the v1.5 paragraph corpus.')
    trec_y1_manual = Benchmark('car-trec-y1-manual',
        docs=docs_v15, queries=trec_y1_queries, qrels=trec_y1_manual_qrels,
        desc='TREC CAR 2017 (Y1) benchmark, with manual (NIST assessor) relevance judgments.')
    trec_y1_auto = Benchmark('car-trec-y1-auto',
        docs=docs_v15, queries=trec_y1_queries, qrels=trec_y1_auto_qrels,
        desc='TREC CAR 2017 (Y1) benchmark, with automatic (Wikipedia heading-derived) relevance judgments.')

    test200 = Benchmark('car-test200',
        docs=docs_v15, queries=test200_queries, qrels=test200_qrels,
        citation=CITATION_TEST200,
        desc='TREC CAR v1.5, a small 200-query sample of the train set, useful for quick tests.')

    train_benchmarks = {}
    for _fold, (_queries, _qrels) in train_folds.items():
        train_benchmarks[_fold] = Benchmark(f'car-train-{_fold}',
            docs=docs_v15, queries=_queries, qrels=_qrels,
            desc=f'TREC CAR v1.5 training data, {_fold} of 5 (for training rankers/embeddings, '
                 'not for evaluation).')


# Registration
# -----------------------------------------
irds.register(docs_v15, docs_v20, trec_y1, trec_y1_manual, trec_y1_auto, test200,
              *train_benchmarks.values())
