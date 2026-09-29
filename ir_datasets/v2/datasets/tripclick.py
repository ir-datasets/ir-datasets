"""TripClick -- a v2 dataset family (clinical-search click data from the Trip
Database, offering both raw click-derived relevance and "dctr" click-through-
rate-binned relevance, over head/torso/tail query-frequency slices).

All of the corpus (``docs_grp_*.txt``), topics and (non-dctr) qrels live
inside a single DUA-gated tar.gz (v1's ``dlc['benchmark']``); the val/test BM25
run files needed for ``scoreddocs`` live in two further DUA-gated tar.gz files
(v1's ``dlc['dlfiles']``/``dlc['dlfiles_runs_test']``). None of the three is
automatically downloadable -- TripClick's data must be requested directly from
the Trip Database -- so each is a plain ``Resource`` behind ``Source.external()``,
at ``<home>/external/tripclick-*.tar.gz`` (v1's old
``<home>/tripclick/*.tar.gz`` still read as fallback), same shape as ``disks45.py``. v1's own ``TarExtractAll``/``RelativePath`` pipeline
(unmodified) is reused to read individual topics/qrels/run files back out of
those shared extracted trees -- same "raw ``Resource`` stays the declared
``source=``, the extracted tree is read inside the ``Parser``" split
``istella22.py``/``csl.py`` use, so ``derived_from`` edges point at the real
DUA-gated archive rather than nothing.

The head/torso/tail query-frequency slices are each their own topics/qrels
files upstream -- no filtering needed, just three separate ``QueryTable``/
``QrelTable`` pairs per split (train, val, test). The *combined* ``train``/
``val``/``test`` benchmarks are not filtered views of anything; they are v1's
own concatenation of the three slices' queries/qrels/scoreddocs
(``ConcatQueries``/``ConcatQrels``/``ConcatScoreddocs``, reused unmodified),
re-parsed from the three slices' own raw files rather than referencing the
already-built per-slice ``Table`` objects -- same "rebuild from the underlying
Resources" shape ``argsme.py``'s ``argsme-2020-04-01-docs`` uses for its own
five-way concatenation.

"dctr" qrels (``train/head/dctr``, ``val/head/dctr``) share their queries (and,
for val, their scoreddocs) with the plain ``head`` benchmark -- only the qrels
file differs (click-through-rate-binned relevance instead of raw
clicked/not-clicked) -- so those benchmarks just point at the same
``QueryTable``/``RunTable`` objects the plain ``head`` benchmark uses, with a
different ``QrelTable``. Test slices have no qrels at all, preserved as-is
from v1 (an unjudged leaderboard-style test set).

``train``'s ``docpairs`` reuses v1's ``DocPairGenerator`` unmodified: it
re-links the docpairs source file's raw (query text, doc text) pairs back to
this family's own ``doc_id``/``query_id`` values by content-hashing every doc
and query once -- expensive, so it caches its output TSV to
``<home>/tripclick/train.docpairs`` on first use, exactly as v1 does. It needs
already-built v1 handlers for the corpus and the combined training queries, so
it is constructed from ``docs.handler``/``train_queries.handler`` (built once
at import time -- cheap; it does no I/O until iterated) rather than through a
``Parser``, and handed to v2's own ``TsvDocPairs`` as its ``source=`` (a
``DocPairGenerator`` already satisfies the ``Readable`` ``path()``/``stream()``
duck type, so no extra wrapper is needed here).

**The ``logs`` subset.** v1's ``tripclick/logs`` dataset combines a partial-
metadata doc collection (title/URL only, distinct from the main corpus) with
``TripClickQlogs``/``BaseQlogs`` -- the raw TripClick search-session log
itself. Both are migrated here as standalone nodes, not a ``Benchmark`` (v1
never bundles them into one either -- same "log is a sibling artifact, not a
facet" shape ``aol_ia.py`` uses for AOL's own query log; see ``QlogTable`` in
``nodes.py``):

* ``tripclick-logs-docs`` (a ``DocTable``, ``TripClickPartialDoc`` rows --
  ``doc_id``/``title``/``url``, no ``text``, a different schema from the main
  ``tripclick`` corpus) reuses v1's own ``FixAllarticles``/``Cache`` pipeline
  over ``allarticles.txt`` unmodified -- the raw file has quoting/line-
  splitting problems that ``FixAllarticles`` streams a correction over before
  v2's ``TsvDocs`` parses it, exactly as v1's ``TsvDocs(Cache(FixAllarticles(...)))``
  does. ``Cache`` still writes its corrected copy to the same fixed
  ``<home>/tripclick/allarticles-fixed.tsv`` v1 uses (not a v2 docstore path)
  since it is `FixAllarticles`'s own cache, upstream of where a v2 ``Parser``
  gets involved -- same shape as ``train_docpairs``' ``DocPairGenerator``
  cache above.
* ``tripclick-logs-qlogs`` (a ``QlogTable``, ``TripClickQlog`` rows) reuses
  v1's own ``TripClickQlogs`` wrapper unmodified, over the same ``logs``
  tar.gz's extracted ``**/*.json`` session files (``TarExtractAll``, the same
  v1 pipeline this file already uses for ``_topics_and_qrels``/``_val_runs``/
  ``_test_runs``).

Both read from a fourth DUA-gated ``Resource`` (v1's ``dlc['logs']``,
``logs.tar.gz``), not yet declared elsewhere in this file since nothing else
here needs it.
"""
import ir_datasets
from ir_datasets.datasets.tripclick import (
    ConcatQrels as _V1ConcatQrels,
    ConcatQueries as _V1ConcatQueries,
    ConcatScoreddocs as _V1ConcatScoreddocs,
    DocPairGenerator as _V1DocPairGenerator,
    FixAllarticles as _V1FixAllarticles,
    QREL_DCTR_DEFS,
    QREL_DEFS,
    QTYPE_MAP,
    TripClickPartialDoc,
    TripClickQlogs as _V1TripClickQlogs,
)
from ir_datasets.util import Cache, RelativePath, TarExtract, TarExtractAll
from ir_datasets.v2 import (
    Benchmark, DocPairTable, DocTable, QlogTable, QrelTable, QueryTable, Resource, RunTable,
    Source, TrecDocs, TsvDocPairs, irds,
)
from ir_datasets.v2.formats import Parser

NAME = 'tripclick'
BASE_PATH = ir_datasets.util.home_path() / NAME

DUA = ("TripClick's source files must be requested directly from the Trip "
       "Database; see <https://tripdatabase.github.io/tripclick/#getting-the-data> "
       "for the data access procedure.")

def DATA_ACCESS(filename):
    return [
        "Request access to the TripClick data from the Trip Database: <https://tripdatabase.github.io/tripclick/#getting-the-data>.",
        f"Once approved, download {filename}.",
    ]

CITATION = 'dblp:conf/sigir/RekabsazLSBE21'


class _TripClickQueriesParser(Parser):
    name = 'TrecQueries'

    def __init__(self, relpath):
        self.relpath = relpath

    def build(self, source, node):
        from ir_datasets.formats import GenericQuery
        from ir_datasets.formats import TrecQueries as _V1TrecQueries
        file = RelativePath(_topics_and_qrels, self.relpath)
        return _V1TrecQueries(file, qtype=GenericQuery, qtype_map=QTYPE_MAP, lang=node.lang)


class _TripClickQrelsParser(Parser):
    name = 'TrecQrels'

    def __init__(self, relpath, defs):
        self.relpath = relpath
        self.defs = defs

    def build(self, source, node):
        from ir_datasets.formats import TrecQrels as _V1TrecQrels
        file = RelativePath(_topics_and_qrels, self.relpath)
        return _V1TrecQrels(file, self.defs)


class _TripClickScoredDocsParser(Parser):
    name = 'TrecScoredDocs'

    def __init__(self, base_dlc, relpath):
        self.base_dlc = base_dlc
        self.relpath = relpath

    def build(self, source, node):
        from ir_datasets.formats import TrecScoredDocs as _V1TrecScoredDocs
        file = RelativePath(self.base_dlc, self.relpath)
        return _V1TrecScoredDocs(file)


class _TripClickConcatQueriesParser(Parser):
    name = 'ConcatQueries'

    def __init__(self, relpaths):
        self.relpaths = relpaths

    def build(self, source, node):
        from ir_datasets.formats import GenericQuery
        from ir_datasets.formats import TrecQueries as _V1TrecQueries
        subs = [_V1TrecQueries(RelativePath(_topics_and_qrels, rp), qtype=GenericQuery,
                                qtype_map=QTYPE_MAP, lang=node.lang) for rp in self.relpaths]
        return _V1ConcatQueries(subs)


class _TripClickConcatQrelsParser(Parser):
    name = 'ConcatQrels'

    def __init__(self, relpaths, defs):
        self.relpaths = relpaths
        self.defs = defs

    def build(self, source, node):
        from ir_datasets.formats import TrecQrels as _V1TrecQrels
        subs = [_V1TrecQrels(RelativePath(_topics_and_qrels, rp), self.defs) for rp in self.relpaths]
        return _V1ConcatQrels(subs)


class _TripClickConcatScoredDocsParser(Parser):
    name = 'ConcatScoreddocs'

    def __init__(self, base_dlc, relpaths):
        self.base_dlc = base_dlc
        self.relpaths = relpaths

    def build(self, source, node):
        from ir_datasets.formats import TrecScoredDocs as _V1TrecScoredDocs
        subs = [_V1TrecScoredDocs(RelativePath(self.base_dlc, rp)) for rp in self.relpaths]
        return _V1ConcatScoreddocs(subs)


class _TripClickLogsDocsParser(Parser):
    """Reuses v1's own ``TsvDocs``, over the same
    ``Cache(FixAllarticles(TarExtract(...)))`` correction pipeline v1's
    ``subsets['logs']`` uses, unmodified -- fixed v1-style base path (the
    docstore lands next to ``Cache``'s own corrected-copy path, not under
    v2's per-node ``docstore_path``), same precedent as ``train_docpairs``'
    ``DocPairGenerator`` above."""
    name = 'TsvDocs(FixAllarticles)'

    def build(self, source, node):
        from ir_datasets.formats import TsvDocs as _V1TsvDocs
        pipeline = Cache(_V1FixAllarticles(TarExtract(logs_file, 'logs/allarticles.txt')),
                         BASE_PATH / 'allarticles-fixed.tsv')
        return _V1TsvDocs(pipeline, doc_cls=TripClickPartialDoc, lang=node.lang,
                          count_hint=node.count_hint)


class _TripClickLogsQlogsParser(Parser):
    """Reuses v1's own ``TripClickQlogs`` unmodified, over the same
    ``logs`` tar.gz's extracted session-json tree (``TarExtractAll``, same
    v1 pipeline this file already uses for topics/qrels/run files)."""
    name = 'TripClickQlogs'

    def build(self, source, node):
        return _V1TripClickQlogs(TarExtractAll(logs_file, BASE_PATH / 'logs',
                                               path_globs=['**/*.json']))


class _TripClickTrainDocPairsParser(Parser):
    """Reuses v1's ``DocPairGenerator`` unmodified (see module docstring) --
    ``source=`` on the ``DocPairTable`` stays the raw dlfiles/benchmark
    Resources (for correct ``derived_from`` edges); the generator itself,
    which needs already-built docs/queries handlers to re-link doc/query text
    back to ids, is constructed here rather than passed as ``source=``
    directly."""
    name = 'TsvDocPairs(DocPairGenerator)'

    def build(self, source, node):
        from ir_datasets.formats import TsvDocPairs as _V1TsvDocPairs
        docpair_src = _V1DocPairGenerator(
            TarExtract(dlfiles_file, 'dlfiles/triples.train.tsv'),
            docs.handler, train_queries.handler, BASE_PATH / 'train.docpairs')
        return _V1TsvDocPairs(docpair_src)


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    benchmark_file = Resource('tripclick-benchmark.tar.gz',
        sources=[Source.external('tripclick-benchmark.tar.gz', old_locations=[f'{NAME}/benchmark.tar.gz'], instructions=DATA_ACCESS('benchmark.tar.gz'))],
        hash='md5:6e5d3deeba138750e9a148b538f30a8f',
        dua=DUA,
    )
    dlfiles_file = Resource('tripclick-dlfiles.tar.gz',
        sources=[Source.external('tripclick-dlfiles.tar.gz', old_locations=[f'{NAME}/dlfiles.tar.gz'], instructions=DATA_ACCESS('dlfiles.tar.gz'))],
        hash='md5:1f256c19466b414e365324d8ef21f09c',
        dua=DUA,
    )
    dlfiles_runs_test_file = Resource('tripclick-dlfiles-runs-test.tar.gz',
        sources=[Source.external('tripclick-dlfiles_runs_test.tar.gz', old_locations=[f'{NAME}/dlfiles_runs_test.tar.gz'], instructions=DATA_ACCESS('dlfiles_runs_test.tar.gz'))],
        hash='md5:2b5e98c683a91e19630636b6f83e3b15',
        dua=DUA,
    )
    # Automatically downloadable (no DUA) -- Hofstaetter et al.'s improved
    # training triples, distributed on the Hub.
    hofstaetter_triples_file = Resource('tripclick-hofstaetter-triples.tsv',
        sources=['https://huggingface.co/datasets/sebastian-hofstaetter/tripclick-training/resolve/main/improved_tripclick_train_triple-ids.tsv'],
        hash='md5:8d70808ec06570e02bc4014ed033b5d0',
        size=233_053_452,
    )
    # The raw search-session log tar.gz -- backs the `logs` subset's docs/
    # qlogs tables below, and only those (see module docstring).
    logs_file = Resource('tripclick-logs.tar.gz',
        sources=[Source.external('tripclick-logs.tar.gz', old_locations=[f'{NAME}/logs.tar.gz'], instructions=DATA_ACCESS('logs.tar.gz'))],
        hash='md5:1d3a548685c2fbef9b2076b0b04ba44f',
        dua=DUA,
    )

    # Shared, extracted-once pipelines (v1 machinery, unmodified) -- see
    # module docstring.
    _topics_and_qrels = TarExtractAll(benchmark_file, BASE_PATH / 'topics_and_qrels',
        path_globs=['**/topics.*.txt', '**/qrels.*.txt'])
    _val_runs = TarExtractAll(dlfiles_file, BASE_PATH / 'val_runs',
        path_globs=['**/run.trip.BM25.*.val.txt'])
    _test_runs = TarExtractAll(dlfiles_runs_test_file, BASE_PATH / 'test_runs',
        path_globs=['**/run.trip.BM25.*.test.txt'])

    # Tables
    # -----------------------------------------
    docs = TrecDocs('tripclick-docs',
        source=benchmark_file,
        path_globs=['**/docs_grp_*.txt'],
        parser='tut',
        count_hint=1_523_878,
    )

    ### Train

    train_head_queries = QueryTable('tripclick-train-head-queries',
        source=benchmark_file,
        parser=_TripClickQueriesParser('benchmark/topics/topics.head.train.txt'),
        count_hint=3_529,
    )
    train_head_qrels = QrelTable('tripclick-train-head-qrels',
        source=benchmark_file,
        parser=_TripClickQrelsParser('benchmark/qrels/qrels.raw.head.train.txt', QREL_DEFS),
        defs=QREL_DEFS, count_hint=116_821,
    )
    train_head_dctr_qrels = QrelTable('tripclick-train-head-dctr-qrels',
        source=benchmark_file,
        parser=_TripClickQrelsParser('benchmark/qrels/qrels.dctr.head.train.txt', QREL_DCTR_DEFS),
        defs=QREL_DCTR_DEFS, count_hint=128_420,
    )

    train_torso_queries = QueryTable('tripclick-train-torso-queries',
        source=benchmark_file,
        parser=_TripClickQueriesParser('benchmark/topics/topics.torso.train.txt'),
        count_hint=105_964,
    )
    train_torso_qrels = QrelTable('tripclick-train-torso-qrels',
        source=benchmark_file,
        parser=_TripClickQrelsParser('benchmark/qrels/qrels.raw.torso.train.txt', QREL_DEFS),
        defs=QREL_DEFS, count_hint=966_898,
    )

    train_tail_queries = QueryTable('tripclick-train-tail-queries',
        source=benchmark_file,
        parser=_TripClickQueriesParser('benchmark/topics/topics.tail.train.txt'),
        count_hint=576_156,
    )
    train_tail_qrels = QrelTable('tripclick-train-tail-qrels',
        source=benchmark_file,
        parser=_TripClickQrelsParser('benchmark/qrels/qrels.raw.tail.train.txt', QREL_DEFS),
        defs=QREL_DEFS, count_hint=1_621_493,
    )

    train_queries = QueryTable('tripclick-train-queries',
        source=benchmark_file,
        parser=_TripClickConcatQueriesParser([
            'benchmark/topics/topics.head.train.txt',
            'benchmark/topics/topics.torso.train.txt',
            'benchmark/topics/topics.tail.train.txt',
        ]),
        count_hint=685_649,
    )
    train_qrels = QrelTable('tripclick-train-qrels',
        source=benchmark_file,
        parser=_TripClickConcatQrelsParser([
            'benchmark/qrels/qrels.raw.head.train.txt',
            'benchmark/qrels/qrels.raw.torso.train.txt',
            'benchmark/qrels/qrels.raw.tail.train.txt',
        ], QREL_DEFS),
        defs=QREL_DEFS, count_hint=2_705_212,
    )
    train_docpairs = DocPairTable('tripclick-train-docpairs',
        source=[dlfiles_file, benchmark_file],
        parser=_TripClickTrainDocPairsParser(),
        lang='en',
        count_hint=23_221_224,
    )
    hofstaetter_docpairs = TsvDocPairs('tripclick-train-hofstaetter-triples-docpairs',
        source=hofstaetter_triples_file,
        lang='en',
        count_hint=10_000_000,
    )

    ### Val

    val_head_queries = QueryTable('tripclick-val-head-queries',
        source=benchmark_file,
        parser=_TripClickQueriesParser('benchmark/topics/topics.head.val.txt'),
        count_hint=1_175,
    )
    val_head_qrels = QrelTable('tripclick-val-head-qrels',
        source=benchmark_file,
        parser=_TripClickQrelsParser('benchmark/qrels/qrels.raw.head.val.txt', QREL_DEFS),
        defs=QREL_DEFS, count_hint=64_364,
    )
    val_head_dctr_qrels = QrelTable('tripclick-val-head-dctr-qrels',
        source=benchmark_file,
        parser=_TripClickQrelsParser('benchmark/qrels/qrels.dctr.head.val.txt', QREL_DCTR_DEFS),
        defs=QREL_DCTR_DEFS, count_hint=66_812,
    )
    val_head_scoreddocs = RunTable('tripclick-val-head-scoreddocs',
        source=dlfiles_file,
        parser=_TripClickScoredDocsParser(_val_runs, 'dlfiles/run.trip.BM25.head.val.txt'),
        count_hint=1_166_804,
    )

    val_torso_queries = QueryTable('tripclick-val-torso-queries',
        source=benchmark_file,
        parser=_TripClickQueriesParser('benchmark/topics/topics.torso.val.txt'),
        count_hint=1_175,
    )
    val_torso_qrels = QrelTable('tripclick-val-torso-qrels',
        source=benchmark_file,
        parser=_TripClickQrelsParser('benchmark/qrels/qrels.raw.torso.val.txt', QREL_DEFS),
        defs=QREL_DEFS, count_hint=14_133,
    )
    val_torso_scoreddocs = RunTable('tripclick-val-torso-scoreddocs',
        source=dlfiles_file,
        parser=_TripClickScoredDocsParser(_val_runs, 'dlfiles/run.trip.BM25.torso.val.txt'),
        count_hint=1_170_314,
    )

    val_tail_queries = QueryTable('tripclick-val-tail-queries',
        source=benchmark_file,
        parser=_TripClickQueriesParser('benchmark/topics/topics.tail.val.txt'),
        count_hint=1_175,
    )
    val_tail_qrels = QrelTable('tripclick-val-tail-qrels',
        source=benchmark_file,
        parser=_TripClickQrelsParser('benchmark/qrels/qrels.raw.tail.val.txt', QREL_DEFS),
        defs=QREL_DEFS, count_hint=3_912,
    )
    val_tail_scoreddocs = RunTable('tripclick-val-tail-scoreddocs',
        source=dlfiles_file,
        parser=_TripClickScoredDocsParser(_val_runs, 'dlfiles/run.trip.BM25.tail.val.txt'),
        count_hint=1_166_192,
    )

    val_queries = QueryTable('tripclick-val-queries',
        source=benchmark_file,
        parser=_TripClickConcatQueriesParser([
            'benchmark/topics/topics.head.val.txt',
            'benchmark/topics/topics.torso.val.txt',
            'benchmark/topics/topics.tail.val.txt',
        ]),
        count_hint=3_525,
    )
    val_qrels = QrelTable('tripclick-val-qrels',
        source=benchmark_file,
        parser=_TripClickConcatQrelsParser([
            'benchmark/qrels/qrels.raw.head.val.txt',
            'benchmark/qrels/qrels.raw.torso.val.txt',
            'benchmark/qrels/qrels.raw.tail.val.txt',
        ], QREL_DEFS),
        defs=QREL_DEFS, count_hint=82_409,
    )
    val_scoreddocs = RunTable('tripclick-val-scoreddocs',
        source=dlfiles_file,
        parser=_TripClickConcatScoredDocsParser(_val_runs, [
            'dlfiles/run.trip.BM25.head.val.txt',
            'dlfiles/run.trip.BM25.torso.val.txt',
            'dlfiles/run.trip.BM25.tail.val.txt',
        ]),
        count_hint=3_503_310,
    )

    ### Test (no qrels -- unjudged, preserved as-is from v1)

    test_head_queries = QueryTable('tripclick-test-head-queries',
        source=benchmark_file,
        parser=_TripClickQueriesParser('benchmark/topics/topics.head.test.txt'),
        count_hint=1_175,
    )
    test_head_scoreddocs = RunTable('tripclick-test-head-scoreddocs',
        source=dlfiles_runs_test_file,
        parser=_TripClickScoredDocsParser(_test_runs, 'runs_test/run.trip.BM25.head.test.txt'),
        count_hint=1_159_303,
    )

    test_torso_queries = QueryTable('tripclick-test-torso-queries',
        source=benchmark_file,
        parser=_TripClickQueriesParser('benchmark/topics/topics.torso.test.txt'),
        count_hint=1_175,
    )
    test_torso_scoreddocs = RunTable('tripclick-test-torso-scoreddocs',
        source=dlfiles_runs_test_file,
        parser=_TripClickScoredDocsParser(_test_runs, 'runs_test/run.trip.BM25.torso.test.txt'),
        count_hint=1_161_972,
    )

    test_tail_queries = QueryTable('tripclick-test-tail-queries',
        source=benchmark_file,
        parser=_TripClickQueriesParser('benchmark/topics/topics.tail.test.txt'),
        count_hint=1_175,
    )
    test_tail_scoreddocs = RunTable('tripclick-test-tail-scoreddocs',
        source=dlfiles_runs_test_file,
        parser=_TripClickScoredDocsParser(_test_runs, 'runs_test/run.trip.BM25.tail.test.txt'),
        count_hint=1_165_127,
    )

    test_queries = QueryTable('tripclick-test-queries',
        source=benchmark_file,
        parser=_TripClickConcatQueriesParser([
            'benchmark/topics/topics.head.test.txt',
            'benchmark/topics/topics.torso.test.txt',
            'benchmark/topics/topics.tail.test.txt',
        ]),
        count_hint=3_525,
    )
    test_scoreddocs = RunTable('tripclick-test-scoreddocs',
        source=dlfiles_runs_test_file,
        parser=_TripClickConcatScoredDocsParser(_test_runs, [
            'runs_test/run.trip.BM25.head.test.txt',
            'runs_test/run.trip.BM25.torso.test.txt',
            'runs_test/run.trip.BM25.tail.test.txt',
        ]),
        count_hint=3_486_402,
    )

    ### Logs (search-session log + its partial-metadata docs -- see module
    ### docstring; standalone artifacts, not a Benchmark, same as v1)

    logs_docs = DocTable('tripclick-logs-docs',
        source=logs_file,
        parser=_TripClickLogsDocsParser(),
    )
    logs_qlogs = QlogTable('tripclick-logs-qlogs',
        source=logs_file,
        parser=_TripClickLogsQlogsParser(),
        count_hint=5_317_350,
    )

    # Benchmarks
    # -----------------------------------------
    train_head = Benchmark('tripclick-train-head',
        docs=docs, queries=train_head_queries, qrels=train_head_qrels,
        citation=CITATION, desc='TripClick training queries, head (high-frequency) slice.')
    train_head_dctr = Benchmark('tripclick-train-head-dctr',
        docs=docs, queries=train_head_queries, qrels=train_head_dctr_qrels,
        citation=CITATION,
        desc='TripClick training queries, head slice, with click-through-rate-binned '
             '("dctr") relevance instead of raw clicked/not-clicked.')
    train_torso = Benchmark('tripclick-train-torso',
        docs=docs, queries=train_torso_queries, qrels=train_torso_qrels,
        citation=CITATION, desc='TripClick training queries, torso (mid-frequency) slice.')
    train_tail = Benchmark('tripclick-train-tail',
        docs=docs, queries=train_tail_queries, qrels=train_tail_qrels,
        citation=CITATION, desc='TripClick training queries, tail (low-frequency) slice.')
    train = Benchmark('tripclick-train',
        docs=docs, queries=train_queries, qrels=train_qrels, docpairs=train_docpairs,
        citation=CITATION, desc='TripClick training queries, head+torso+tail combined.')
    train_hofstaetter_triples = Benchmark('tripclick-train-hofstaetter-triples',
        docs=docs, queries=train_queries, qrels=train_qrels, docpairs=hofstaetter_docpairs,
        citation=CITATION,
        desc="TripClick training queries (head+torso+tail combined), with Hofstaetter et "
             "al.'s improved training triples in place of v1's own docpair extraction.")

    val_head = Benchmark('tripclick-val-head',
        docs=docs, queries=val_head_queries, qrels=val_head_qrels, scoreddocs=val_head_scoreddocs,
        citation=CITATION, desc='TripClick validation queries, head (high-frequency) slice.')
    val_head_dctr = Benchmark('tripclick-val-head-dctr',
        docs=docs, queries=val_head_queries, qrels=val_head_dctr_qrels, scoreddocs=val_head_scoreddocs,
        citation=CITATION,
        desc='TripClick validation queries, head slice, with click-through-rate-binned '
             '("dctr") relevance instead of raw clicked/not-clicked.')
    val_torso = Benchmark('tripclick-val-torso',
        docs=docs, queries=val_torso_queries, qrels=val_torso_qrels, scoreddocs=val_torso_scoreddocs,
        citation=CITATION, desc='TripClick validation queries, torso (mid-frequency) slice.')
    val_tail = Benchmark('tripclick-val-tail',
        docs=docs, queries=val_tail_queries, qrels=val_tail_qrels, scoreddocs=val_tail_scoreddocs,
        citation=CITATION, desc='TripClick validation queries, tail (low-frequency) slice.')
    val = Benchmark('tripclick-val',
        docs=docs, queries=val_queries, qrels=val_qrels, scoreddocs=val_scoreddocs,
        citation=CITATION, desc='TripClick validation queries, head+torso+tail combined.')

    test_head = Benchmark('tripclick-test-head',
        docs=docs, queries=test_head_queries, scoreddocs=test_head_scoreddocs,
        citation=CITATION,
        desc='TripClick test queries, head (high-frequency) slice. Unjudged (no qrels).')
    test_torso = Benchmark('tripclick-test-torso',
        docs=docs, queries=test_torso_queries, scoreddocs=test_torso_scoreddocs,
        citation=CITATION,
        desc='TripClick test queries, torso (mid-frequency) slice. Unjudged (no qrels).')
    test_tail = Benchmark('tripclick-test-tail',
        docs=docs, queries=test_tail_queries, scoreddocs=test_tail_scoreddocs,
        citation=CITATION,
        desc='TripClick test queries, tail (low-frequency) slice. Unjudged (no qrels).')
    test = Benchmark('tripclick-test',
        docs=docs, queries=test_queries, scoreddocs=test_scoreddocs,
        citation=CITATION,
        desc='TripClick test queries, head+torso+tail combined. Unjudged (no qrels).')


# Registration
# -----------------------------------------
irds.register(
    docs,
    train_head, train_head_dctr, train_torso, train_tail, train, train_hofstaetter_triples,
    val_head, val_head_dctr, val_torso, val_tail, val,
    test_head, test_torso, test_tail, test,
    logs_docs, logs_qlogs,
)
