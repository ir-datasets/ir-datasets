"""Touché -- a v2 dataset family (the CLEF Touché shared tasks on argument
retrieval, comparative argument retrieval, argument quality, and
controversial image retrieval), 2020-2022.

Same shape as ``trec_web.py``: a single track spanning multiple
already-migrated corpora, judged from one file -- ``argsme.py`` (three
different doc tables: ``docs_1_0``, ``docs_2020_04_01``,
``docs_2020_04_01_processed``, all imported by reference), ``clueweb12.py``
(``docs``, imported by reference), and ``touche_image.py`` (``docs``,
imported by reference). There is no separate "touche corpus" of its own, so
this track gets its own file rather than being folded into one of those
(unlike ``csl.py``/``cord19.py``/``nyt.py``/``pmc.py``, whose corpora each
feed only a single TREC family).

Every query/qrels handler here (``ToucheQueries``, ``ToucheTitleQueries``,
``ToucheComparativeQueries``, ``ToucheQrels``, ``ToucheQualityQrels``,
``ToucheQualityCoherenceQrels``, ``ToucheQualityComparativeStanceQrels``,
``ToucheControversialStanceQrels``, ``TouchePassageDocs``) lives in
``ir_datasets/formats/touche.py``, not ``formats/trec.py`` -- narrow,
task-specific classes with no general-purpose v2 format node, same shape as
``codec.py``'s ``CodecQueries`` -- so each is reused unmodified via a thin
local ``Parser``, same "thin layer over v1 machinery" pattern as everywhere
else. Multi-file qrels handlers (``ToucheQualityQrels`` and friends) take
their extra files (quality/coherence/stance) as extra constructor args, not
through the source pipeline -- so their ``QrelTable``'s ``source=`` is a
*list* of the underlying Resources (relevance, [quality], [coherence/
stance]), and the parser unpacks that list positionally before handing each
piece to the v1 class. This matters for more than tidiness: ``source=`` is
what ``Table`` walks (via ``source_resources``) to build its ``derived_from``
edges, so every file actually read must appear there or the resulting qrels
table would silently claim to derive from fewer Resources than it really
reads (the same bug a previous migration this session, ``istella22.py``,
had to be caught and fixed for).

Task-2 2022 (comparative argument retrieval, ``touche-2022-task-2``) is
judged over passage-level text, not the raw ClueWeb12 doc collection --
v1's ``TouchePassageDocs`` builds a standalone doc collection from a
downloaded/extracted JSONL(.gz) of passages (``.gunzip()`` on the Resource),
so its ``docs=`` table (``touche-2022-task-2-docs``) is its own new
``DocTable`` here, distinct from ``clueweb12-docs``. Its
``expanded-doc-t5-query`` sub-dataset shares the exact same ``queries=``/
``qrels=`` nodes (imported by reference, not rebuilt) but reads a different
passages file into its own ``docs=`` table. Doc count for both
(``868655``) is hardcoded here exactly as v1 hardcodes it in its own
``_init()`` -- a real known value, not read from ``etc/metadata.json``.

Two "uncorrected" sub-datasets of ``touche-2020-task-1`` re-judge args.me
1.0 and args.me 2020-04-01 respectively with an earlier, uncorrected set of
crowdworker labels (``ToucheQrels(..., allow_float_score=True)``) -- both
reuse ``touche-2020-task-1``'s own ``queries=`` table by reference (same
node, not a rebuilt copy), same as v1's
``registry[...].queries_handler()``. Being over two different corpora they
need two distinct flat names, chosen here as
``touche-2020-task-1-argsme-1.0-uncorrected`` and
``touche-2020-task-1-argsme-2020-04-01-uncorrected``.

**Observation on 2022 task-1 vs task-3 queries**: v1's ``downloads.json``
entries for ``2022/task-1/queries`` and ``2022/task-3/queries`` point at two
different Zenodo records (6873574 vs 6873575) but both name the file
``topics.xml`` and declare the *identical* md5
(``83a9cb2290f867199d6de9c73eeacf43``) and size (31089) -- i.e. Touché
republished the same topics file verbatim alongside both tasks' own
records. Since they are two distinct published URLs (their own DOI-bearing
records), this module keeps them as two separate ``Resource``/``QueryTable``
nodes (``touche-2022-task-1-queries`` / ``touche-2022-task-3-queries``)
rather than merging them into one shared node -- each benchmark's
provenance should point at the record it actually cites, even though the
bytes happen to be identical.

Node names are flat throughout (``touche-2020-task-1``,
``touche-2020-task-1-queries``, ``touche-2022-task-2-docs``, ...), per the
v2 flat-naming convention.
"""
from ir_datasets.formats import touche as _v1_touche
from ir_datasets.v2 import Benchmark, DocTable, QrelTable, QueryTable, Resource, irds
from ir_datasets.v2.datasets.argsme import docs_1_0 as argsme_docs_1_0
from ir_datasets.v2.datasets.argsme import docs_2020_04_01 as argsme_docs_2020_04_01
from ir_datasets.v2.datasets.argsme import docs_2020_04_01_processed as argsme_docs_2020_04_01_processed
from ir_datasets.v2.datasets.clueweb12 import docs as clueweb12_docs
from ir_datasets.v2.datasets.touche_image import docs as touche_image_docs
from ir_datasets.v2.formats import Parser

QRELS_DEFS_2020_TASK_1 = {
    -2: "spam, non-argument",
    1: "very low relevance",
    2: "low relevance",
    3: "moderate relevance",
    4: "high relevance",
    5: "very high relevance",
}
QRELS_DEFS_2020_TASK_2 = {
    0: "not relevant",
    1: "relevant",
    2: "highly relevant",
}
QRELS_DEFS_2021_TASK_1 = {
    -2: "spam",
    0: "not relevant",
    1: "relevant",
    2: "highly relevant",
}
QRELS_DEFS_2021_TASK_2 = {
    0: "not relevant",
    1: "relevant",
    2: "highly relevant",
}
QRELS_DEFS_2022_TASK_1 = {
    0: "not relevant",
    1: "relevant",
    2: "highly relevant",
}
QRELS_DEFS_2022_TASK_2 = {
    0: "not relevant",
    1: "relevant",
    2: "highly relevant",
}
QRELS_DEFS_2022_TASK_3 = {
    0: "not relevant",
    1: "relevant",
}

CITATION_TASK_1_2020 = 'dblp:conf/clef/BondarenkoFBGAP20; dblp:conf/eacl/WachsmuthSHPBHN17'
CITATION_TASK_2_2020 = 'dblp:conf/clef/BondarenkoFBGAP20; dblp:conf/ecir/BraunstainKCSS16; dblp:conf/www/RafalakAW14'
CITATION_2021 = 'dblp:conf/clef/BondarenkoGFBAP21a'
CITATION_2022 = 'dblp:conf/clef/BondarenkoFKSGB22a'
CITATION_TASK_3_2022 = 'dblp:conf/clef/BondarenkoFKSGB22a; dblp:conf/argmining/KieselRSP21; dblp:conf/semeval/DimitrovASASFNM21; dblp:conf/www/Yanai07'


class _ToucheQueriesParser(Parser):
    name = 'ToucheQueries'

    def build(self, source, node):
        return _v1_touche.ToucheQueries(source, language=node.lang)


class _ToucheTitleQueriesParser(Parser):
    name = 'ToucheTitleQueries'

    def build(self, source, node):
        return _v1_touche.ToucheTitleQueries(source, language=node.lang)


class _ToucheComparativeQueriesParser(Parser):
    name = 'ToucheComparativeQueries'

    def build(self, source, node):
        return _v1_touche.ToucheComparativeQueries(source, language=node.lang)


class _ToucheQrelsParser(Parser):
    name = 'ToucheQrels'

    def __init__(self, allow_float_score=False):
        self.allow_float_score = allow_float_score

    def build(self, source, node):
        return _v1_touche.ToucheQrels(source, node.defs or {}, allow_float_score=self.allow_float_score)


class _ToucheQualityQrelsParser(Parser):
    """``source=`` is ``[relevance_file, quality_file]`` -- see module
    docstring for why both must be threaded through ``source=`` rather than
    passed around it."""
    name = 'ToucheQualityQrels'

    def build(self, source, node):
        relevance, quality = source
        return _v1_touche.ToucheQualityQrels(relevance, quality, node.defs or {})


class _ToucheQualityCoherenceQrelsParser(Parser):
    """``source=`` is ``[relevance_file, quality_file, coherence_file]``."""
    name = 'ToucheQualityCoherenceQrels'

    def build(self, source, node):
        relevance, quality, coherence = source
        return _v1_touche.ToucheQualityCoherenceQrels(relevance, quality, coherence, node.defs or {})


class _ToucheQualityComparativeStanceQrelsParser(Parser):
    """``source=`` is ``[relevance_file, quality_file, stance_file]``."""
    name = 'ToucheQualityComparativeStanceQrels'

    def build(self, source, node):
        relevance, quality, stance = source
        return _v1_touche.ToucheQualityComparativeStanceQrels(relevance, quality, stance, node.defs or {})


class _ToucheControversialStanceQrelsParser(Parser):
    name = 'ToucheControversialStanceQrels'

    def build(self, source, node):
        return _v1_touche.ToucheControversialStanceQrels(source, node.defs or {})


class _TouchePassageDocsParser(Parser):
    name = 'TouchePassageDocs'

    def build(self, source, node):
        return _v1_touche.TouchePassageDocs(source, language=node.lang, count_hint=node.count_hint)


# license verified 2026-09-30: Zenodo records 6862281, 6797876, 6798216, 6798217, 6873574, 6873567, 6873575 (Touche
# topics/qrels/passages) all list CC-BY-4.0 (e.g. https://zenodo.org/api/records/6862281)
with irds.defaults(lang='en', license='CC-BY-4.0'):
    # Files
    # -----------------------------------------
    task1_2020_queries_file = Resource('touche-2020-task-1-queries.zip',
        sources=['https://zenodo.org/record/6862281/files/topics-task-1.zip'],
        hash='md5:9605104435165a6b01b737464596eba4',
        size=8_768,
    ).zip_member('topics-task-1.xml')
    task1_2020_qrels_file = Resource('touche-2020-task-1-qrels.qrels',
        sources=['https://zenodo.org/record/6862281/files/touche2020-task1-relevance-args-me-corpus-version-2020-04-01-corrected.qrels'],
        hash='md5:6a645e2ebd4f1d6c44da4d9509624598',
        size=62_058,
    )
    task1_2020_qrels_argsme_1_0_uncorrected_file = Resource('touche-2020-task-1-qrels-argsme-1.0-uncorrected.qrels',
        sources=['https://zenodo.org/record/6862281/files/touche2020-task1-relevance-args-me-corpus-version-1.qrels'],
        hash='md5:10f043e086818f9159ac37a9ebe5ce5d',
        size=145_201,
    )
    task1_2020_qrels_argsme_2020_04_01_uncorrected_file = Resource('touche-2020-task-1-qrels-argsme-2020-04-01-uncorrected.qrels',
        sources=['https://zenodo.org/record/6862281/files/touche2020-task1-relevance-args-me-corpus-version-2020-04-01.qrels'],
        hash='md5:6a27d7123423540664ccfe0391e4e417',
        size=66_283,
    )

    task2_2020_queries_file = Resource('touche-2020-task-2-queries.zip',
        sources=['https://zenodo.org/record/6797876/files/topics-task-2.zip'],
        hash='md5:8de387d753ee8289a9f02346b63e12e4',
        size=17_279,
    ).zip_member('topics-task-2.xml')
    task2_2020_qrels_file = Resource('touche-2020-task-2-qrels.qrels',
        sources=['https://zenodo.org/record/6797876/files/touche2020-task2-relevance-withbaseline.qrels'],
        hash='md5:b230436beb3a9eecbeb19c84ee6c855c',
        size=58_522,
    )

    task1_2021_queries_file = Resource('touche-2021-task-1-queries.zip',
        sources=['https://zenodo.org/record/6798216/files/topics-task-1-only-titles-2021.zip'],
        hash='md5:61bad9cf6bc713a81297cd95cf9e156f',
        size=1_350,
    ).zip_member('topics-task-1-only-titles.xml')
    task1_2021_qrels_relevance_file = Resource('touche-2021-task-1-qrels-relevance.qrels',
        sources=['https://zenodo.org/record/6798216/files/touche-task1-51-100-relevance.qrels'],
        hash='md5:76b4e8348bde353167ce52ffa598a6b1',
        size=99_736,
    )
    task1_2021_qrels_quality_file = Resource('touche-2021-task-1-qrels-quality.qrels',
        sources=['https://zenodo.org/record/6798216/files/touche-task1-51-100-quality.qrels'],
        hash='md5:c899bcab9b00fdd28f77d08a1d26298a',
        size=100_087,
    )

    task2_2021_queries_file = Resource('touche-2021-task-2-queries.zip',
        sources=['https://zenodo.org/record/6798217/files/topics-task-2-2021.zip'],
        hash='md5:0c06079b327ecd2b4c5971bd17bd0aa3',
        size=15_532,
    ).zip_member('topics-task2-51-100.xml')
    task2_2021_qrels_relevance_file = Resource('touche-2021-task-2-qrels-relevance.qrels',
        sources=['https://zenodo.org/record/6798217/files/touche-task2-51-100-relevance.qrels'],
        hash='md5:970b48e0c057afaee17a7832100ab67c',
        size=68_548,
    )
    task2_2021_qrels_quality_file = Resource('touche-2021-task-2-qrels-quality.qrels',
        sources=['https://zenodo.org/record/6798217/files/touche-task2-51-100-quality.qrels'],
        hash='md5:062bea3b8307ae481876fe31253092b5',
        size=68_548,
    )

    task1_2022_queries_file = Resource('touche-2022-task-1-queries.xml',
        sources=['https://zenodo.org/record/6873574/files/topics.xml'],
        hash='md5:83a9cb2290f867199d6de9c73eeacf43',
        size=31_089,
    )
    task1_2022_qrels_relevance_file = Resource('touche-2022-task-1-qrels-relevance.qrels',
        sources=['https://zenodo.org/record/6873574/files/touche-task1-2022-relevance-dedup.qrels'],
        hash='md5:658e5c13d8e5a80371d73ee1b04499bc',
        size=482_162,
    )
    task1_2022_qrels_quality_file = Resource('touche-2022-task-1-qrels-quality.qrels',
        sources=['https://zenodo.org/record/6873574/files/touche-task1-2022-quality-dedup.qrels'],
        hash='md5:8d12dac4ca8dfc0693ab34c926e87c1b',
        size=482_162,
    )
    task1_2022_qrels_coherence_file = Resource('touche-2022-task-1-qrels-coherence.qrels',
        sources=['https://zenodo.org/record/6873574/files/touche-task1-2022-coherence-dedup.qrels'],
        hash='md5:5389bb1df02d3ba54a458eb969cbc5e9',
        size=482_162,
    )

    task2_2022_queries_file = Resource('touche-2022-task-2-queries.zip',
        sources=['https://zenodo.org/record/6873567/files/topics-task2-2022.zip'],
        hash='md5:fafbb6352be108419535aaed83fc5762',
        size=17_203,
    ).zip_member('topics-task2.xml')
    task2_2022_qrels_relevance_file = Resource('touche-2022-task-2-qrels-relevance.qrels',
        sources=['https://zenodo.org/record/6873567/files/touche-task2-2022-relevance.qrels'],
        hash='md5:dcdd1031ce2e0830ae76d7b21fca2579',
        size=78_379,
    )
    task2_2022_qrels_quality_file = Resource('touche-2022-task-2-qrels-quality.qrels',
        sources=['https://zenodo.org/record/6873567/files/touche-task2-2022-quality.qrels'],
        hash='md5:adf28c877efcd94954263b392714cb60',
        size=78_379,
    )
    task2_2022_qrels_stance_file = Resource('touche-2022-task-2-qrels-stance.qrels',
        sources=['https://zenodo.org/record/6873567/files/touche-task2-2022-stance.qrels'],
        hash='md5:ac9afec4eb590877df9c94dd8d931a37',
        size=84_891,
    )
    task2_2022_passages_file = Resource('touche-2022-task-2-passages.jsonl.gz',
        sources=['https://zenodo.org/record/6873567/files/touche-task2-passages-version-002.jsonl.gz'],
        hash='md5:ed4d6104b78986849c59bbc470464cec',
        size=285_743_369,
    ).gunzip()
    task2_2022_passages_expanded_file = Resource('touche-2022-task-2-passages-expanded-doc-t5-query.jsonl.gz',
        sources=['https://zenodo.org/record/6873567/files/touche-task2-passages-version-002-expanded-with-doc-t5-query.jsonl.gz'],
        hash='md5:062452996389a320b83ce274df82cf4b',
        size=300_736_532,
    ).gunzip()

    task3_2022_queries_file = Resource('touche-2022-task-3-queries.xml',
        sources=['https://zenodo.org/record/6873575/files/topics.xml'],
        hash='md5:83a9cb2290f867199d6de9c73eeacf43',
        size=31_089,
    )
    task3_2022_qrels_file = Resource('touche-2022-task-3-qrels.qrels',
        sources=['https://zenodo.org/record/6873575/files/touche-task3-001-050-relevance.qrels'],
        hash='md5:83ec2d715d0205b68b9b63f9c30da784',
        size=558_028,
    )

    # Tables
    # -----------------------------------------
    task1_2020_queries = QueryTable('touche-2020-task-1-queries',
        source=task1_2020_queries_file, parser=_ToucheQueriesParser(), count_hint=49)
    task1_2020_qrels = QrelTable('touche-2020-task-1-qrels',
        source=task1_2020_qrels_file, parser=_ToucheQrelsParser(),
        defs=QRELS_DEFS_2020_TASK_1, count_hint=2_298)
    task1_2020_argsme_1_0_uncorrected_qrels = QrelTable('touche-2020-task-1-argsme-1.0-uncorrected-qrels',
        source=task1_2020_qrels_argsme_1_0_uncorrected_file,
        parser=_ToucheQrelsParser(allow_float_score=True),
        defs=QRELS_DEFS_2020_TASK_1, count_hint=2_964)
    task1_2020_argsme_2020_04_01_uncorrected_qrels = QrelTable('touche-2020-task-1-argsme-2020-04-01-uncorrected-qrels',
        source=task1_2020_qrels_argsme_2020_04_01_uncorrected_file,
        parser=_ToucheQrelsParser(allow_float_score=True),
        defs=QRELS_DEFS_2020_TASK_1, count_hint=2_298)

    task2_2020_queries = QueryTable('touche-2020-task-2-queries',
        source=task2_2020_queries_file, parser=_ToucheQueriesParser(), count_hint=50)
    task2_2020_qrels = QrelTable('touche-2020-task-2-qrels',
        source=task2_2020_qrels_file, parser=_ToucheQrelsParser(),
        defs=QRELS_DEFS_2020_TASK_2, count_hint=1_783)

    task1_2021_queries = QueryTable('touche-2021-task-1-queries',
        source=task1_2021_queries_file, parser=_ToucheTitleQueriesParser(), count_hint=50)
    task1_2021_qrels = QrelTable('touche-2021-task-1-qrels',
        source=[task1_2021_qrels_relevance_file, task1_2021_qrels_quality_file],
        parser=_ToucheQualityQrelsParser(),
        defs=QRELS_DEFS_2021_TASK_1, count_hint=3_711)

    task2_2021_queries = QueryTable('touche-2021-task-2-queries',
        source=task2_2021_queries_file, parser=_ToucheQueriesParser(), count_hint=50)
    task2_2021_qrels = QrelTable('touche-2021-task-2-qrels',
        source=[task2_2021_qrels_relevance_file, task2_2021_qrels_quality_file],
        parser=_ToucheQualityQrelsParser(),
        defs=QRELS_DEFS_2021_TASK_2, count_hint=2_076)

    task1_2022_queries = QueryTable('touche-2022-task-1-queries',
        source=task1_2022_queries_file, parser=_ToucheQueriesParser(), count_hint=50)
    task1_2022_qrels = QrelTable('touche-2022-task-1-qrels',
        source=[task1_2022_qrels_relevance_file, task1_2022_qrels_quality_file, task1_2022_qrels_coherence_file],
        parser=_ToucheQualityCoherenceQrelsParser(),
        defs=QRELS_DEFS_2022_TASK_1, count_hint=6_841)

    task2_2022_docs = DocTable('touche-2022-task-2-docs',
        source=task2_2022_passages_file, parser=_TouchePassageDocsParser(),
        count_hint=868_655)
    task2_2022_expanded_docs = DocTable('touche-2022-task-2-expanded-doc-t5-query-docs',
        source=task2_2022_passages_expanded_file, parser=_TouchePassageDocsParser(),
        count_hint=868_655)
    task2_2022_queries = QueryTable('touche-2022-task-2-queries',
        source=task2_2022_queries_file, parser=_ToucheComparativeQueriesParser(), count_hint=50)
    task2_2022_qrels = QrelTable('touche-2022-task-2-qrels',
        source=[task2_2022_qrels_relevance_file, task2_2022_qrels_quality_file, task2_2022_qrels_stance_file],
        parser=_ToucheQualityComparativeStanceQrelsParser(),
        defs=QRELS_DEFS_2022_TASK_2, count_hint=2_107)

    task3_2022_queries = QueryTable('touche-2022-task-3-queries',
        source=task3_2022_queries_file, parser=_ToucheQueriesParser(), count_hint=50)
    task3_2022_qrels = QrelTable('touche-2022-task-3-qrels',
        source=task3_2022_qrels_file, parser=_ToucheControversialStanceQrelsParser(),
        defs=QRELS_DEFS_2022_TASK_3, count_hint=19_821)

    # Benchmarks
    # -----------------------------------------
    touche_2020_task_1 = Benchmark('touche-2020-task-1',
        docs=argsme_docs_2020_04_01, queries=task1_2020_queries, qrels=task1_2020_qrels,
        citation=CITATION_TASK_1_2020,
        desc='Touché 2020 Task 1: Argument Retrieval for Controversial Questions.')
    touche_2020_task_1_argsme_1_0_uncorrected = Benchmark('touche-2020-task-1-argsme-1.0-uncorrected',
        docs=argsme_docs_1_0, queries=task1_2020_queries, qrels=task1_2020_argsme_1_0_uncorrected_qrels,
        citation=CITATION_TASK_1_2020,
        desc='Touché 2020 Task 1, over the args.me 1.0 corpus with uncorrected '
             'crowdworker relevance judgements. Should not be used without preprocessing.')
    touche_2020_task_1_argsme_2020_04_01_uncorrected = Benchmark('touche-2020-task-1-argsme-2020-04-01-uncorrected',
        docs=argsme_docs_2020_04_01, queries=task1_2020_queries, qrels=task1_2020_argsme_2020_04_01_uncorrected_qrels,
        citation=CITATION_TASK_1_2020,
        desc='Touché 2020 Task 1, with uncorrected crowdworker relevance '
             'judgements. Should not be used without preprocessing.')

    touche_2020_task_2 = Benchmark('touche-2020-task-2',
        docs=clueweb12_docs, queries=task2_2020_queries, qrels=task2_2020_qrels,
        citation=CITATION_TASK_2_2020,
        desc='Touché 2020 Task 2: Argument Retrieval for Comparative Questions.')

    touche_2021_task_1 = Benchmark('touche-2021-task-1',
        docs=argsme_docs_2020_04_01, queries=task1_2021_queries, qrels=task1_2021_qrels,
        citation=CITATION_2021,
        desc='Touché 2021 Task 1: Argument Retrieval for Controversial Questions.')

    touche_2021_task_2 = Benchmark('touche-2021-task-2',
        docs=clueweb12_docs, queries=task2_2021_queries, qrels=task2_2021_qrels,
        citation=CITATION_2021,
        desc='Touché 2021 Task 2: Argument Retrieval for Comparative Questions.')

    touche_2022_task_1 = Benchmark('touche-2022-task-1',
        docs=argsme_docs_2020_04_01_processed, queries=task1_2022_queries, qrels=task1_2022_qrels,
        citation=CITATION_2022,
        desc='Touché 2022 Task 1: Argument Retrieval for Controversial Questions.')

    touche_2022_task_2 = Benchmark('touche-2022-task-2',
        docs=task2_2022_docs, queries=task2_2022_queries, qrels=task2_2022_qrels,
        citation=CITATION_2022,
        desc='Touché 2022 Task 2: Argument Retrieval for Comparative Questions.')
    touche_2022_task_2_expanded_doc_t5_query = Benchmark('touche-2022-task-2-expanded-doc-t5-query',
        docs=task2_2022_expanded_docs, queries=task2_2022_queries, qrels=task2_2022_qrels,
        citation=CITATION_2022,
        desc='Touché 2022 Task 2, over passages expanded with DocT5Query.')

    touche_2022_task_3 = Benchmark('touche-2022-task-3',
        docs=touche_image_docs, queries=task3_2022_queries, qrels=task3_2022_qrels,
        citation=CITATION_TASK_3_2022,
        desc='Touché 2022 Task 3: Controversial Image Retrieval.')


# Registration
# -----------------------------------------
irds.register(
    touche_2020_task_1, touche_2020_task_1_argsme_1_0_uncorrected, touche_2020_task_1_argsme_2020_04_01_uncorrected,
    touche_2020_task_2,
    touche_2021_task_1, touche_2021_task_2,
    touche_2022_task_1,
    touche_2022_task_2, touche_2022_task_2_expanded_doc_t5_query,
    touche_2022_task_3,
)
