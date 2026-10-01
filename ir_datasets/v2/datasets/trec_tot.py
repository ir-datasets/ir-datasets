"""TREC Tip-of-the-Tongue (ToT) -- a v2 dataset family spanning three years
of the track (2023, 2024, 2025), migrated from TWO legacy v1 modules
(``ir_datasets/datasets/trec_tot.py`` for 2023+2024, ``trec_tot_2025.py`` for
2025) into this single v2 file. Both legacy modules declare the exact same
``NAME = 'trec-tot'`` and together register one coherent ``trec-tot/...`` id
space -- the split into two ``.py`` files was purely a v1-internal
implementation convenience (2025's much larger corpus needed a different
random-access storage engine, see below), not a track/namespace boundary, so
it does not belong in v2's "split TREC tasks into their own file" convention
either.

2023 and 2024 share the same shape: a single downloadable zip per year (2023:
one zip holding the corpus plus every split's queries/qrels under a
``TREC-TOT/`` prefix; 2024: a separate corpus-only zip, plus, per query
split, yet another separate zip -- v1's own ``_init()`` downloads 2024 test
queries from ``dlc['2024-test']``, not ``dlc['2024']``), read via v1's
``JsonlDocs``/``JsonlQueries`` handlers reused unmodified, wrapped in thin
local ``Parser``s -- there is no general-purpose v2 ``JsonlDocs``/
``JsonlQueries`` node in ``formats.py`` the way there is for ``TsvDocs``, same
as ``istella22.py``'s reasoning. v2's ``.zip_member()`` pipeline op replaces
v1's ``ZipExtract``; docs additionally ``.cache(path)`` the extracted member
to a real, legacy-compatible file (at the same path v1's own ``Cache(...)``
used, so an existing v1-populated cache is picked up rather than
re-extracted) -- needed because ``JsonlDocs`` builds a random-access
docstore, and ``ZipExtract.path()`` alone returns the *zip's* path, not the
member's, so a bare ``.zip_member()`` cannot stand in for a real file the way
``.stream()`` can. Queries and qrels never need a real file (only
``.stream()``), so they read straight off the ``.zip_member()`` pipeline
expression with no caching. Qrels go through v2's general-purpose
``TrecQrels`` format node directly (no local parser needed at all, since
``TrecQrels`` only ever streams).

2025 corpus/offsets are a genuinely different storage engine: one big gzipped
JSONL corpus file plus a separate gzipped JSONL "offsets" index (per-doc
``doc_id``/``offset_start``/``offset_end``) that lets a custom ``Docstore``
subclass do byte-range random access straight into the corpus file (each
record was gzip-compressed independently, so a byte range read out of the
file is itself a complete, independently-decompressible gzip member) rather
than building a full ``PickleLz4FullStore`` over the whole ~6.4M-doc corpus.
``TrecToT2025Doc``/``JsonlWithOffsetsDocsStore``/``TrecToT2025DocsStore``/
``JsonlDocumentsWithOffsets``/``JsonlDocumentOffset``, all defined in legacy
``trec_tot_2025.py``, are reused completely unmodified via a thin ``Parser``
that constructs ``JsonlDocumentsWithOffsets(docs_resource, offsets_resource)``
directly from the two v2 ``Resource``s -- confirmed safe because that class's
``docs_iter``/``docs_store`` only ever call ``.path()``/``.stream()`` on what
they're given, the same "a v2 Resource satisfies the same duck type as a v1
dlc" reasoning ``c4.py``/``clinicaltrials.py``/``highwire.py`` rely on.
``docs_count()`` is hardcoded to ``6407814`` inside ``JsonlDocumentsWithOffsets``
itself (not passed in) -- used verbatim as this table's ``count_hint``. All
five 2025 splits (train/dev1/dev2/dev3/test) share the exact same docs+offsets
pair, so there is exactly one ``trec-tot-2025-docs`` node, never re-declared
per split. ``test`` has no qrels (v1's own ``_init()`` comment: "datasets that
currently do not have qrels"), the same queries-only shape as
``trec-tot-2024-test``.

``docs/trec-tot.yaml``/``docs/trec-tot-2025.yaml`` declare no ``bibtex_ids``;
each Benchmark cites its year's track overview (added in the DBLP pass).
"""
import ir_datasets
from ir_datasets.datasets.trec_tot import (
    QUERY_MAP, TipOfTheTongueDoc, TipOfTheTongueDoc2024, TipOfTheTongueQuery,
    TipOfTheTongueQuery2024,
)
from ir_datasets.datasets.trec_tot_2025 import JsonlDocumentsWithOffsets
from ir_datasets.v2 import Benchmark, DocTable, QueryTable, Resource, TrecQrels, irds
from ir_datasets.v2.formats import Parser

BASE_PATH = ir_datasets.util.home_path() / 'trec-tot'

QREL_DEFS = {0: 'Not Relevant', 1: 'Relevant'}

# Different field mapping than 2023's QUERY_MAP -- 2025's queries.jsonl has a
# different shape (just an id + a "query" field), see module docstring.
QUERY_MAP_2025 = {'text': 'query', 'query_id': 'query_id'}


class _JsonlDocsParser(Parser):
    """Wraps v1's ``JsonlDocs`` directly. ``source`` is the
    ``.zip_member(...)`` pipeline expression set at the DocTable's own
    construction site; this parser only adds the ``.cache(path)`` step
    ``JsonlDocs`` needs for its random-access docstore -- see module
    docstring for why a bare ``.zip_member()`` isn't enough."""
    name = 'JsonlDocs'

    def __init__(self, doc_cls, cache_path):
        self.doc_cls = doc_cls
        self.cache_path = cache_path

    def build(self, source, node):
        from ir_datasets.formats import JsonlDocs
        docs_src = source.cache(self.cache_path)
        return JsonlDocs(docs_src, doc_cls=self.doc_cls, lang=node.lang,
                          count_hint=node.count_hint,
                          docstore_path=str(node.docstore_path))


class _JsonlQueriesParser(Parser):
    """Wraps v1's ``JsonlQueries`` directly, straight over ``source`` -- no
    caching needed, ``JsonlQueries`` only ever streams."""
    name = 'JsonlQueries'

    def __init__(self, query_cls=None, mapping=None):
        self.query_cls = query_cls
        self.mapping = mapping

    def build(self, source, node):
        from ir_datasets.formats import JsonlQueries
        kwargs = {'mapping': self.mapping, 'lang': node.lang}
        if self.query_cls is not None:
            kwargs['query_cls'] = self.query_cls
        return JsonlQueries(source, **kwargs)


class _TrecToT2025DocsParser(Parser):
    """Wraps v1's offset-based random-access docstore
    (``JsonlDocumentsWithOffsets``/``TrecToT2025DocsStore``) completely
    unmodified -- see module docstring. ``source`` is the two-element
    ``[corpus_resource, offsets_resource]`` list set on the DocTable."""
    name = 'JsonlDocumentsWithOffsets'

    def build(self, source, node):
        docs_resource, offsets_resource = source
        return JsonlDocumentsWithOffsets(docs_resource, offsets_resource)


with irds.defaults(lang='en'):
    # Files -- 2023
    # -----------------------------------------
    corpus_2023_file = Resource('trec-tot-2023-corpus.zip',
        sources=['https://surfdrive.surf.nl/files/index.php/s/FaEK4xc6Xp2JcAJ/download'],
        hash='md5:f84fe82cb80e3ee1072576c8d6c4a417',
    )

    # Tables -- 2023
    # -----------------------------------------
    docs_2023 = DocTable('trec-tot-2023-docs',
        source=corpus_2023_file.zip_member('TREC-TOT/corpus.jsonl'),
        parser=_JsonlDocsParser(TipOfTheTongueDoc, BASE_PATH / '2023/corpus.jsonl'),
        count_hint=231_852)

    train_2023_queries = QueryTable('trec-tot-2023-train-queries',
        source=corpus_2023_file.zip_member('TREC-TOT/train/queries.jsonl'),
        parser=_JsonlQueriesParser(TipOfTheTongueQuery, mapping=QUERY_MAP),
        count_hint=150)
    train_2023_qrels = TrecQrels('trec-tot-2023-train-qrels',
        source=corpus_2023_file.zip_member('TREC-TOT/train/qrel.txt'),
        defs=QREL_DEFS, count_hint=150)

    dev_2023_queries = QueryTable('trec-tot-2023-dev-queries',
        source=corpus_2023_file.zip_member('TREC-TOT/dev/queries.jsonl'),
        parser=_JsonlQueriesParser(TipOfTheTongueQuery, mapping=QUERY_MAP),
        count_hint=150)
    dev_2023_qrels = TrecQrels('trec-tot-2023-dev-qrels',
        source=corpus_2023_file.zip_member('TREC-TOT/dev/qrel.txt'),
        defs=QREL_DEFS, count_hint=150)

    # Benchmarks -- 2023
    # -----------------------------------------
    trec_tot_2023_train = Benchmark('trec-tot-2023-train',
        citation='dblp:conf/trec/ArguelloBDKM23',
        docs=docs_2023, queries=train_2023_queries, qrels=train_2023_qrels,
        desc='TREC Tip-of-the-Tongue 2023: train query set.')
    trec_tot_2023_dev = Benchmark('trec-tot-2023-dev',
        citation='dblp:conf/trec/ArguelloBDKM23',
        docs=docs_2023, queries=dev_2023_queries, qrels=dev_2023_qrels,
        desc='TREC Tip-of-the-Tongue 2023: dev query set.')

    # Files -- 2024
    # -----------------------------------------
    corpus_2024_file = Resource('trec-tot-2024-corpus.zip',
        sources=['https://zenodo.org/records/13370657/files/corpus.jsonl.zip?download=1'],
        hash='md5:4ea86770817e46a06fea5c94f596409c',
    )
    test_2024_queries_file = Resource('trec-tot-2024-test-queries.zip',
        sources=['https://zenodo.org/records/13370657/files/test-2024.zip?download=1'],
        hash='md5:3d0a4d83957ee6a1398afefbc96162fa',
    )

    # Tables -- 2024
    # -----------------------------------------
    docs_2024 = DocTable('trec-tot-2024-docs',
        source=corpus_2024_file.zip_member('corpus.jsonl'),
        parser=_JsonlDocsParser(TipOfTheTongueDoc2024, BASE_PATH / '2024/corpus.jsonl'),
        count_hint=3_185_450)

    # license verified 2026-09-30 (2024/2025 queries and qrels; corpora left unlicensed, Wikipedia-derived text):
    # Zenodo records 13370657, 15356599, 15869078 list CC-BY-4.0 (https://zenodo.org/api/records/15356599)
    test_2024_queries = QueryTable('trec-tot-2024-test-queries',
        license='CC-BY-4.0',
        source=test_2024_queries_file.zip_member('test-2024/queries.jsonl'),
        parser=_JsonlQueriesParser(TipOfTheTongueQuery2024),
        count_hint=600)

    # Benchmarks -- 2024
    # -----------------------------------------
    trec_tot_2024_test = Benchmark('trec-tot-2024-test',
        citation='dblp:conf/trec/Arguello00KHK024',
        docs=docs_2024, queries=test_2024_queries,
        desc='TREC Tip-of-the-Tongue 2024: test query set (queries only; '
             'v1 does not wire up qrels for this dataset).')

    # Files -- 2025
    # -----------------------------------------
    corpus_2025_file = Resource('trec-tot-2025-corpus.jsonl.gz',
        sources=['https://zenodo.org/records/15356599/files/trec-tot-2025-corpus.jsonl.gz'],
        hash='md5:a2c82398aa86df6a68c8706b9b462bf2',
    )
    offsets_2025_file = Resource('trec-tot-2025-offsets.jsonl.gz',
        sources=['https://zenodo.org/records/15356599/files/trec-tot-2025-offsets.jsonl.gz'],
        hash='md5:00678e3155d962bb244e034e6401b79b',
    )

    train_2025_queries_file = Resource('trec-tot-2025-train-queries.jsonl',
        sources=['https://zenodo.org/records/15356599/files/train-2025-queries.jsonl'],
        hash='md5:288b7707b4e897f7447aac2cc2f613be',
    )
    train_2025_qrels_file = Resource('trec-tot-2025-train-qrels.txt',
        sources=['https://zenodo.org/records/15356599/files/train-2025-qrel.txt'],
        hash='md5:10a3c727fc5806ec4510f7a071b57cd7',
    )
    dev1_2025_queries_file = Resource('trec-tot-2025-dev1-queries.jsonl',
        sources=['https://zenodo.org/records/15356599/files/dev1-2025-queries.jsonl'],
        hash='md5:b87c2f51d058de844e258a69b02e70fc',
    )
    dev1_2025_qrels_file = Resource('trec-tot-2025-dev1-qrels.txt',
        sources=['https://zenodo.org/records/15356599/files/dev1-2025-qrel.txt'],
        hash='md5:0c913ce8b5b287c73a6dfac662971e82',
    )
    dev2_2025_queries_file = Resource('trec-tot-2025-dev2-queries.jsonl',
        sources=['https://zenodo.org/records/15356599/files/dev2-2025-queries.jsonl'],
        hash='md5:b174a128a255e92d0d54b76465d596b5',
    )
    dev2_2025_qrels_file = Resource('trec-tot-2025-dev2-qrels.txt',
        sources=['https://zenodo.org/records/15356599/files/dev2-2025-qrel.txt'],
        hash='md5:4548eb41e639905384aa017c69129bfc',
    )
    dev3_2025_queries_file = Resource('trec-tot-2025-dev3-queries.jsonl',
        sources=['https://zenodo.org/records/15356599/files/dev3-2025-queries.jsonl'],
        hash='md5:259c11645694a3c5230b66c7852d4d80',
    )
    dev3_2025_qrels_file = Resource('trec-tot-2025-dev3-qrels.txt',
        sources=['https://zenodo.org/records/15356599/files/dev3-2025-qrel.txt'],
        hash='md5:48ab0d24a5946861546e54064238477f',
    )
    test_2025_queries_file = Resource('trec-tot-2025-test-queries.jsonl',
        sources=['https://zenodo.org/records/15869078/files/test-2025-queries.jsonl'],
        hash='md5:374cdc9142240f8bc9e4b071c35713f8',
    )

    # Tables -- 2025
    # -----------------------------------------
    # Shared by every 2025 split -- see module docstring.
    docs_2025 = DocTable('trec-tot-2025-docs',
        source=[corpus_2025_file, offsets_2025_file],
        parser=_TrecToT2025DocsParser(),
        count_hint=6_407_814)

    train_2025_queries = QueryTable('trec-tot-2025-train-queries',
        license='CC-BY-4.0',
        source=train_2025_queries_file,
        parser=_JsonlQueriesParser(mapping=QUERY_MAP_2025),
        count_hint=143)
    train_2025_qrels = TrecQrels('trec-tot-2025-train-qrels',
        license='CC-BY-4.0',
        source=train_2025_qrels_file, defs=QREL_DEFS, count_hint=143)

    dev1_2025_queries = QueryTable('trec-tot-2025-dev1-queries',
        license='CC-BY-4.0',
        source=dev1_2025_queries_file,
        parser=_JsonlQueriesParser(mapping=QUERY_MAP_2025),
        count_hint=142)
    dev1_2025_qrels = TrecQrels('trec-tot-2025-dev1-qrels',
        license='CC-BY-4.0',
        source=dev1_2025_qrels_file, defs=QREL_DEFS, count_hint=142)

    dev2_2025_queries = QueryTable('trec-tot-2025-dev2-queries',
        license='CC-BY-4.0',
        source=dev2_2025_queries_file,
        parser=_JsonlQueriesParser(mapping=QUERY_MAP_2025),
        count_hint=143)
    dev2_2025_qrels = TrecQrels('trec-tot-2025-dev2-qrels',
        license='CC-BY-4.0',
        source=dev2_2025_qrels_file, defs=QREL_DEFS, count_hint=143)

    dev3_2025_queries = QueryTable('trec-tot-2025-dev3-queries',
        license='CC-BY-4.0',
        source=dev3_2025_queries_file,
        parser=_JsonlQueriesParser(mapping=QUERY_MAP_2025),
        count_hint=536)
    dev3_2025_qrels = TrecQrels('trec-tot-2025-dev3-qrels',
        license='CC-BY-4.0',
        source=dev3_2025_qrels_file, defs=QREL_DEFS, count_hint=536)

    test_2025_queries = QueryTable('trec-tot-2025-test-queries',
        license='CC-BY-4.0',
        source=test_2025_queries_file,
        parser=_JsonlQueriesParser(mapping=QUERY_MAP_2025),
        count_hint=622)

    # Benchmarks -- 2025
    # -----------------------------------------
    trec_tot_2025_train = Benchmark('trec-tot-2025-train',
        citation='dblp:journals/corr/abs-2601-20671',
        docs=docs_2025, queries=train_2025_queries, qrels=train_2025_qrels,
        desc='TREC Tip-of-the-Tongue 2025: train query set.')
    trec_tot_2025_dev1 = Benchmark('trec-tot-2025-dev1',
        citation='dblp:journals/corr/abs-2601-20671',
        docs=docs_2025, queries=dev1_2025_queries, qrels=dev1_2025_qrels,
        desc='TREC Tip-of-the-Tongue 2025: dev-1 query set (the original 2023 dev set).')
    trec_tot_2025_dev2 = Benchmark('trec-tot-2025-dev2',
        citation='dblp:journals/corr/abs-2601-20671',
        docs=docs_2025, queries=dev2_2025_queries, qrels=dev2_2025_qrels,
        desc='TREC Tip-of-the-Tongue 2025: dev-2 query set (the original 2023 test set).')
    trec_tot_2025_dev3 = Benchmark('trec-tot-2025-dev3',
        citation='dblp:journals/corr/abs-2601-20671',
        docs=docs_2025, queries=dev3_2025_queries, qrels=dev3_2025_qrels,
        desc='TREC Tip-of-the-Tongue 2025: dev-3 query set (the original 2024 test set).')
    trec_tot_2025_test = Benchmark('trec-tot-2025-test',
        citation='dblp:journals/corr/abs-2601-20671',
        docs=docs_2025, queries=test_2025_queries,
        desc='TREC Tip-of-the-Tongue 2025: test query set (queries only; '
             'no qrels released yet).')


# Registration
# -----------------------------------------
irds.register(
    docs_2023, trec_tot_2023_train, trec_tot_2023_dev,
    docs_2024, trec_tot_2024_test,
    docs_2025, trec_tot_2025_train, trec_tot_2025_dev1, trec_tot_2025_dev2,
    trec_tot_2025_dev3, trec_tot_2025_test,
)
