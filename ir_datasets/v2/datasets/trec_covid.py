"""TREC-COVID -- a v2 dataset family, judged entirely over CORD-19 snapshots
(imported by reference from ``cord19.py``, by date -- same cross-file
pattern as ``trec_adhoc.py`` importing ``docs`` from ``disks45.py``).

Previously bundled in ``cord19.py`` as ``cord19-trec-covid``/
``cord19-fulltext-trec-covid``/``cord19-trec-covid-round1..5`` (CORD-19 had
no other consumer at the time) -- moved here and renamed to drop the corpus
prefix, matching the unprefixed ``trec-web-2002``-style convention every
other multi-corpus track file uses. This overturns the single-consumer
bundling convention ``cord19.py``'s docstring previously argued for: a TREC
track's home shouldn't depend on how many corpora happen to be migrated for
it yet (see ``trec_core.py``'s docstring, and todo-v2.txt).

v1's own ``round1``-``round5`` subset names are a TREC-COVID judging concept,
not a property of the corpus itself, so the "round" numbering now lives only
here (as benchmark ids); ``cord19.py``'s doc tables are named by snapshot
date instead. Round 5 and the main TREC-COVID benchmark share the *same*
docs table (the 2020-07-16 snapshot) and the *same* queries table -- only
round 5's qrels file is its own; that reuse is preserved here exactly as it
was in ``cord19.py`` (see that module's git history for the v1 provenance).
"""
from ir_datasets.v2 import Benchmark, Resource, TrecQrels, TrecQueries, irds
from ir_datasets.v2.datasets.cord19 import docs_2020_04_10, docs_2020_05_01, docs_2020_05_19
from ir_datasets.v2.datasets.cord19 import docs_2020_06_19, docs_2020_07_16, docs_2020_07_16_fulltext

CITATION = 'Voorhees2020TrecCovid'

QRELS_DEFS = {
    2: 'Relevant: the article is fully responsive to the information need as expressed by the topic, i.e. answers the Question in the topic. The article need not contain all information on the topic, but must, on its own, provide an answer to the question.',
    1: 'Partially Relevant: the article answers part of the question but would need to be combined with other information to get a complete answer.',
    0: 'Not Relevant: everything else.',
}

QTYPE_MAP = {
    'query': 'title',
    'question': 'description',
    'narrative': 'narrative',
}

with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    queries_file = Resource('trec-covid-queries.xml',
        sources=['https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml'],
        md5='0307a37b6b9f1a5f233340a769d538ea',
        size=18_707,
    )
    qrels_file = Resource('trec-covid-qrels.txt',
        sources=['https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt'],
        md5='8138424a59daea0aba751c8a891e5f54',
        size=1_142_244,
    )
    round1_queries_file = Resource('trec-covid-round1-queries.xml',
        sources=['https://ir.nist.gov/covidSubmit/data/topics-rnd1.xml'],
        md5='cf1b605222f45f7dbc90ca8e4d9b2c31',
        size=10_348,
    )
    round1_qrels_file = Resource('trec-covid-round1-qrels.txt',
        sources=['https://ir.nist.gov/covidSubmit/data/qrels-rnd1.txt'],
        md5='d58586df5823e7d1d0b3619a73b31518',
        size=150_110,
    )
    round2_queries_file = Resource('trec-covid-round2-queries.xml',
        sources=['https://ir.nist.gov/covidSubmit/data/topics-rnd2.xml'],
        md5='550129e71c83de3fb4d6d29a172c5842',
        size=12_291,
    )
    round2_qrels_file = Resource('trec-covid-round2-qrels.txt',
        sources=['https://ir.nist.gov/covidSubmit/data/qrels-rnd2.txt'],
        md5='157df01d5a084b09be089407f41cf51b',
        size=212_662,
    )
    round3_queries_file = Resource('trec-covid-round3-queries.xml',
        sources=['https://ir.nist.gov/covidSubmit/data/topics-rnd3.xml'],
        md5='aa42a15c107e74488c8189a16a311358',
        size=14_271,
    )
    round3_qrels_file = Resource('trec-covid-round3-qrels.txt',
        sources=['https://ir.nist.gov/covidSubmit/data/qrels-covid_d3_j2.5-3.txt'],
        md5='2a534a42b5b6b43dd8ae7d9433249006',
        size=223_360,
    )
    round4_queries_file = Resource('trec-covid-round4-queries.xml',
        sources=['https://ir.nist.gov/covidSubmit/data/topics-rnd4.xml'],
        md5='202ba3155b1e390115ae13f34d80d4fc',
        size=16_327,
    )
    round4_qrels_file = Resource('trec-covid-round4-qrels.txt',
        sources=['https://ir.nist.gov/covidSubmit/data/qrels-covid_d4_j3.5-4.txt'],
        md5='b86dd338d6b0a41e62f18e566e541b96',
        size=232_379,
    )
    round5_qrels_file = Resource('trec-covid-round5-qrels.txt',
        sources=['https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j4.5-5.txt'],
        md5='6111c00ac9adac774f5b51e4f9a2a25b',
        size=402_401,
    )

    # Tables
    # -----------------------------------------
    queries = TrecQueries('trec-covid-queries',
        source=queries_file, qtype_map=QTYPE_MAP,
        count_hint=50,
        citation=CITATION,
    )
    qrels = TrecQrels('trec-covid-qrels',
        source=qrels_file, defs=QRELS_DEFS, count_hint=69_318)

    round1_queries = TrecQueries('trec-covid-round1-queries',
        source=round1_queries_file, qtype_map=QTYPE_MAP,
        count_hint=30,
        citation=CITATION,
    )
    round1_qrels = TrecQrels('trec-covid-round1-qrels',
        source=round1_qrels_file, defs=QRELS_DEFS, count_hint=8_691)

    round2_queries = TrecQueries('trec-covid-round2-queries',
        source=round2_queries_file, qtype_map=QTYPE_MAP,
        count_hint=35,
        citation=CITATION,
    )
    round2_qrels = TrecQrels('trec-covid-round2-qrels',
        source=round2_qrels_file, defs=QRELS_DEFS, count_hint=12_037)

    round3_queries = TrecQueries('trec-covid-round3-queries',
        source=round3_queries_file, qtype_map=QTYPE_MAP,
        count_hint=40,
        citation=CITATION,
    )
    round3_qrels = TrecQrels('trec-covid-round3-qrels',
        source=round3_qrels_file, defs=QRELS_DEFS, count_hint=12_713)

    round4_queries = TrecQueries('trec-covid-round4-queries',
        source=round4_queries_file, qtype_map=QTYPE_MAP,
        count_hint=45,
        citation=CITATION,
    )
    round4_qrels = TrecQrels('trec-covid-round4-qrels',
        source=round4_qrels_file, defs=QRELS_DEFS, count_hint=13_262)

    round5_qrels = TrecQrels('trec-covid-round5-qrels',
        source=round5_qrels_file, defs=QRELS_DEFS, count_hint=23_151)

    # Benchmarks
    # -----------------------------------------
    trec_covid = Benchmark('trec-covid',
        docs=docs_2020_07_16, queries=queries, qrels=qrels,
        citation=CITATION,
        desc='The complete TREC-COVID collection: the 2020-07-16 (metadata-only) '
             'CORD-19 snapshot with deep, cumulative relevance judgments.')
    fulltext_trec_covid = Benchmark('trec-covid-fulltext',
        docs=docs_2020_07_16_fulltext, queries=queries, qrels=qrels,
        citation=CITATION,
        desc='Same queries/qrels as trec-covid, but over the full-text '
             '(rather than metadata-only) 2020-07-16 CORD-19 snapshot.')
    trec_covid_round1 = Benchmark('trec-covid-round1',
        docs=docs_2020_04_10, queries=round1_queries, qrels=round1_qrels,
        citation=CITATION,
        desc='Round 1 of the TREC-COVID task, over the 2020-04-10 CORD-19 snapshot.')
    trec_covid_round2 = Benchmark('trec-covid-round2',
        docs=docs_2020_05_01, queries=round2_queries, qrels=round2_qrels,
        citation=CITATION,
        desc='Round 2 of the TREC-COVID task, over the 2020-05-01 CORD-19 snapshot.')
    trec_covid_round3 = Benchmark('trec-covid-round3',
        docs=docs_2020_05_19, queries=round3_queries, qrels=round3_qrels,
        citation=CITATION,
        desc='Round 3 of the TREC-COVID task, over the 2020-05-19 CORD-19 snapshot.')
    trec_covid_round4 = Benchmark('trec-covid-round4',
        docs=docs_2020_06_19, queries=round4_queries, qrels=round4_qrels,
        citation=CITATION,
        desc='Round 4 of the TREC-COVID task, over the 2020-06-19 CORD-19 snapshot.')
    trec_covid_round5 = Benchmark('trec-covid-round5',
        docs=docs_2020_07_16, queries=queries, qrels=round5_qrels,
        citation=CITATION,
        desc='Round 5 of the TREC-COVID task: same docs/queries as trec-covid '
             '(the 2020-07-16 snapshot), but with round-5-only relevance judgments.')


# Registration
# -----------------------------------------
irds.register(trec_covid, fulltext_trec_covid,
              trec_covid_round1, trec_covid_round2, trec_covid_round3, trec_covid_round4,
              trec_covid_round5)
