"""TREC Common Core -- a v2 dataset family, covering both years the track
has run: 2017 (judged over ``nyt-docs``) and 2018 (judged over
``wapo-v2-docs``) -- all in one file since it's the same track throughout,
just switching corpora each year, same "track spans multiple corpora"
reasoning as trec_web.py/trec_genomics.py/trec_pm.py.

Docs are imported by reference from ``nyt.py``/``wapo.py`` -- same
cross-file pattern as ``trec_adhoc.py`` importing ``docs`` from
``disks45.py``.

Previously bundled in their respective corpus files as ``nyt-trec-core-2017``
and ``wapo-trec-core-2018`` (the latter alongside ``wapo-trec-news-*``, now
split out separately into ``trec_news.py``) -- moved here and renamed to drop
those corpus prefixes, matching the unprefixed ``trec-web-2002``-style
convention every other multi-corpus track file uses. This overturns the
single-consumer bundling convention those two files' docstrings previously
argued for: a TREC track's home shouldn't depend on how many corpora happen
to be migrated for it yet, only on whether the track name itself is
corpus-agnostic (see todo-v2.txt).
"""
from ir_datasets.datasets.nyt import CORE_QREL_DEFS as NYT_CORE_QREL_DEFS
from ir_datasets.datasets.wapo import CORE_QREL_DEFS as WAPO_CORE_QREL_DEFS
from ir_datasets.v2 import Benchmark, Resource, Source, TrecQrels, TrecQueries, irds
from ir_datasets.v2.datasets.nyt import docs as nyt_docs
from ir_datasets.v2.datasets.wapo import docs_v2 as wapo_v2_docs

CITATION_2017 = 'Allan2017TrecCore; Sandhaus2008Nyt'

with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    core_2017_queries_file = Resource('trec-core-2017-queries.txt',
        sources=['https://trec.nist.gov/data/core/core_nist.txt', Source.irds()],
        md5='821f8eaaf11ae3ce9657d1442749480a',
        size=24_444,
    )
    core_2017_qrels_file = Resource('trec-core-2017-qrels.txt',
        sources=['https://trec.nist.gov/data/core/qrels.txt', Source.irds()],
        md5='8cf8dcafba6557e5ee62a28a44b0314d',
        size=462_387,
    )
    core_2018_queries_file = Resource('trec-core-2018-queries.txt',
        sources=['https://trec.nist.gov/data/core/topics2018.txt', Source.irds()],
        md5='1b11276f0e1badd68347884664816654',
        size=24_079,
    )
    core_2018_qrels_file = Resource('trec-core-2018-qrels.txt',
        sources=['https://trec.nist.gov/data/core/qrels2018.txt', Source.irds()],
        md5='7a982cd110f8bb30da4141f0f639f2e1',
        size=1_121_301,
    )

    # Tables
    # -----------------------------------------
    core_2017_queries = TrecQueries('trec-core-2017-queries',
        source=core_2017_queries_file, count_hint=50)
    core_2017_qrels = TrecQrels('trec-core-2017-qrels',
        source=core_2017_qrels_file, defs=NYT_CORE_QREL_DEFS, count_hint=30_030)

    core_2018_queries = TrecQueries('trec-core-2018-queries',
        source=core_2018_queries_file, count_hint=50)
    core_2018_qrels = TrecQrels('trec-core-2018-qrels',
        source=core_2018_qrels_file, defs=WAPO_CORE_QREL_DEFS, count_hint=26_233)

    # Benchmarks
    # -----------------------------------------
    core_2017 = Benchmark('trec-core-2017',
        docs=nyt_docs, queries=core_2017_queries, qrels=core_2017_qrels,
        citation=CITATION_2017,
        desc='TREC Common Core 2017 (50 queries assessed by NIST).')
    core_2018 = Benchmark('trec-core-2018',
        docs=wapo_v2_docs, queries=core_2018_queries, qrels=core_2018_qrels,
        desc='TREC Common Core 2018 (50 queries assessed by NIST).')


# Registration
# -----------------------------------------
irds.register(core_2017, core_2018)
