"""TREC Deep Learning (2019-2023) -- a v2 dataset family.

TREC-DL judges the *same* query set against two different corpora (MS MARCO
passage and MS MARCO document), so it isn't naturally "part of" either
corpus's own file -- it's its own benchmark identity, living in its own
file, that reuses each corpus family's ``docs`` Table (imported below, not
re-declared) as its corpus. 2019/2020 judge the v1 corpora
(``msmarco_passage.py``/``msmarco_document.py``); 2021 onward switched to
the "v2" (bigger, re-segmented) corpora (``msmarco_passage_v2.py``/
``msmarco_document_v2.py``) -- MS MARCO's own corpus-revision naming, not
this package's, same caveat those two files note.

Queries are shared between the passage and document tasks for a given year
(same file, same md5, per v1's own download config): declared once here and
referenced by both sets of benchmarks, so the graph shows one QueryTable
node with two independent Benchmarks pointing at it. Qrels/scoreddocs are
corpus-specific and carry "-passage"/"-document" in their names.

Not covered here (left for a future pass, same as the corpus files above):
``trec-dl-hard`` and its 5 folds -- needs an external fold-id list
(``DL_HARD_QIDS_BYFOLD``) transcribed from v1.
"""
from ir_datasets.v2 import Benchmark, Filter, Resource, Source, TrecQrels, TrecScoredDocs, TsvQueries, irds
from ir_datasets.v2.datasets.msmarco_document import CITATION as DOCUMENT_CITATION
from ir_datasets.v2.datasets.msmarco_document import DUA
from ir_datasets.v2.datasets.msmarco_document import docs as document_docs
from ir_datasets.v2.datasets.msmarco_document_v2 import docs as document_v2_docs
from ir_datasets.v2.datasets.msmarco_passage import CITATION as PASSAGE_CITATION
from ir_datasets.v2.datasets.msmarco_passage import docs as passage_docs
from ir_datasets.v2.datasets.msmarco_passage import extract_qid_pid
from ir_datasets.v2.datasets.msmarco_passage_v2 import docs as passage_v2_docs

MEASURES = ['nDCG@10', 'RR(rel=2)', 'AP(rel=2)']

PASSAGE_QRELS_DEFS = {
    3: "Perfectly relevant: The passage is dedicated to the query and contains the exact answer.",
    2: "Highly relevant: The passage has some answer for the query, but the answer may be a bit "
       "unclear, or hidden amongst extraneous information.",
    1: "Related: The passage seems related to the query but does not answer it.",
    0: "Irrelevant: The passage has nothing to do with the query.",
}

DOCUMENT_QRELS_DEFS = {
    3: "Perfectly relevant: Document is dedicated to the query, it is worthy of being a top result "
       "in a search engine.",
    2: "Highly relevant: The content of this document provides substantial information on the query.",
    1: "Relevant: Document provides some information relevant to the query, which may be minimal.",
    0: "Irrelevant: Document does not provide any useful information about the query",
}

with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    queries_2019_file = Resource('trec-dl-2019-queries.tsv.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz'],
        md5='eda71eccbe4d251af83150abe065368c',
        size=4_276,
    )
    queries_2020_file = Resource('trec-dl-2020-queries.tsv.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz'],
        md5='00a406fb0d14ed3752d70d1e4eb98600',
        size=4_131,
    )

    passage_2019_qrels_file = Resource('trec-dl-2019-passage-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2019qrels-pass.txt', Source.irds()],
        md5='2f4be390198da108f6845c822e5ada14',
        size=187_092,
    )
    passage_2019_scoreddocs_file = Resource('trec-dl-2019-passage-scoreddocs.tsv.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz'],
        md5='ec9e012746aa9763c7ff10b3336a3ce1',
        size=26_634_062,
    )
    passage_2020_qrels_file = Resource('trec-dl-2020-passage-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2020qrels-pass.txt', Source.irds()],
        md5='0355ccee7509ac0463e8278186cdd8d1',
        size=218_617,
    )
    passage_2020_scoreddocs_file = Resource('trec-dl-2020-passage-scoreddocs.tsv.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2020-top1000.tsv.gz'],
        md5='aa6fbc51d66bd1dc745964c0e140a727',
        size=26_230_218,
    )

    document_2019_qrels_file = Resource('trec-dl-2019-document-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2019qrels-docs.txt'],
        md5='d7ef53b995ef7e01676ea85d7ec01dda',
        size=339_438,
    )
    document_2019_scoreddocs_file = Resource('trec-dl-2019-document-scoreddocs.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz'],
        md5='91071b89dd52124057a87d53cd22028d',
        size=220_457,
    )
    document_2020_qrels_file = Resource('trec-dl-2020-document-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2020qrels-docs.txt'],
        md5='e10f3545583b124a4ed5e7992293e15a',
        size=182_852,
    )
    document_2020_scoreddocs_file = Resource('trec-dl-2020-document-scoreddocs.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctest2020-top100.gz'],
        md5='96f39dae3443736bd6393bd09a5a0a20',
        size=208_679,
    )

    queries_2021_file = Resource('trec-dl-2021-queries.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_queries.tsv'],
        md5='46d863434dda18300f5af33ee29c4b28',
        size=24_585,
    )
    queries_2022_file = Resource('trec-dl-2022-queries.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_queries.tsv'],
        md5='f1bfd53d80e81e58207ce557fd2211a0',
        size=21_508,
    )
    queries_2023_file = Resource('trec-dl-2023-queries.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_queries.tsv'],
        md5='7df9e17b47cc9aa5d1c9fd5b313e273c',
        size=38_128,
    )

    passage_2021_qrels_file = Resource('trec-dl-2021-passage-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2021.qrels.pass.final.txt', Source.irds()],
        md5='c5b76ec95b589732edc9040302e22a2b',
        size=433_887,
    )
    passage_2021_scoreddocs_file = Resource('trec-dl-2021-passage-scoreddocs.tsv.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_passage_top100.txt.gz'],
        md5='e2be2d307da26d1a3f76eb95507672a3',
        size=604_533,
    )
    passage_2022_qrels_file = Resource('trec-dl-2022-passage-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2022.qrels.pass.withDupes.txt', Source.irds()],
        md5='b36484d6cfd039664a570a4bf04f0eeb',
        size=15_800_539,
    )
    passage_2022_scoreddocs_file = Resource('trec-dl-2022-passage-scoreddocs.tsv.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_passage_top100.txt.gz'],
        md5='36004dfad64826167aeecddff1d490a6',
        size=630_095,
    )
    passage_2023_qrels_file = Resource('trec-dl-2023-passage-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt', Source.irds()],
        md5='3a742d51ae65da2ece9c09b304b9e358',
        size=912_450,
    )
    passage_2023_scoreddocs_file = Resource('trec-dl-2023-passage-scoreddocs.tsv.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_passage_top100.txt.gz'],
        md5='c339ed75e1556cacb387899f34cadad1',
        size=888_898,
    )

    document_2021_qrels_file = Resource('trec-dl-2021-document-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2021.qrels.docs.final.txt', Source.irds()],
        md5='3b266fdaf27f3775e04028765a4839d3',
        size=478_328,
    )
    document_2021_scoreddocs_file = Resource('trec-dl-2021-document-scoreddocs.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_document_top100.txt.gz'],
        md5='0bc85e3f2a6f798b91e18f0cd4a6bc6b',
        size=618_228,
    )
    document_2022_qrels_file = Resource('trec-dl-2022-document-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2022.qrels.docs.inferred.txt', Source.irds()],
        md5='cca2e4db9d842e6262500532809bd571',
        size=13_808_681,
    )
    document_2022_scoreddocs_file = Resource('trec-dl-2022-document-scoreddocs.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_document_top100.txt.gz'],
        md5='93f70329ce1b9ce913a5f87008736ff2',
        size=642_721,
    )
    document_2023_qrels_file = Resource('trec-dl-2023-document-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2023.qrels.docs.withDupes.txt', Source.irds()],
        md5='1e9c540b3cb03bcc975a583586c04090',
        size=675_015,
    )
    document_2023_scoreddocs_file = Resource('trec-dl-2023-document-scoreddocs.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_document_top100.txt.gz'],
        md5='0f5d548e53afb9e319c837ad67f9046a',
        size=902_168,
    )

    # Tables
    # -----------------------------------------
    # Shared by both the passage and document benchmarks below.
    queries_2019 = TsvQueries('trec-dl-2019-queries', source=queries_2019_file.gunzip())
    queries_2020 = TsvQueries('trec-dl-2020-queries', source=queries_2020_file.gunzip())

    passage_2019_qrels = TrecQrels('trec-dl-2019-passage-qrels',
        source=passage_2019_qrels_file, defs=PASSAGE_QRELS_DEFS)
    passage_2019_scoreddocs = TrecScoredDocs('trec-dl-2019-passage-scoreddocs',
        source=passage_2019_scoreddocs_file.gunzip().pipe(extract_qid_pid))
    passage_2020_qrels = TrecQrels('trec-dl-2020-passage-qrels',
        source=passage_2020_qrels_file, defs=PASSAGE_QRELS_DEFS)
    passage_2020_scoreddocs = TrecScoredDocs('trec-dl-2020-passage-scoreddocs',
        source=passage_2020_scoreddocs_file.gunzip().pipe(extract_qid_pid))

    document_2019_qrels = TrecQrels('trec-dl-2019-document-qrels',
        source=document_2019_qrels_file, defs=DOCUMENT_QRELS_DEFS)
    document_2019_scoreddocs = TrecScoredDocs('trec-dl-2019-document-scoreddocs',
        source=document_2019_scoreddocs_file.gunzip())
    document_2020_qrels = TrecQrels('trec-dl-2020-document-qrels',
        source=document_2020_qrels_file, defs=DOCUMENT_QRELS_DEFS)
    document_2020_scoreddocs = TrecScoredDocs('trec-dl-2020-document-scoreddocs',
        source=document_2020_scoreddocs_file.gunzip())

    # Shared by both the passage-v2 and document-v2 benchmarks below.
    queries_2021 = TsvQueries('trec-dl-2021-queries', source=queries_2021_file)
    queries_2022 = TsvQueries('trec-dl-2022-queries', source=queries_2022_file)
    queries_2023 = TsvQueries('trec-dl-2023-queries', source=queries_2023_file)

    passage_2021_qrels = TrecQrels('trec-dl-2021-passage-qrels',
        source=passage_2021_qrels_file, defs=PASSAGE_QRELS_DEFS)
    passage_2021_scoreddocs = TrecScoredDocs('trec-dl-2021-passage-scoreddocs',
        source=passage_2021_scoreddocs_file.gunzip())
    passage_2022_qrels = TrecQrels('trec-dl-2022-passage-qrels',
        source=passage_2022_qrels_file, defs=PASSAGE_QRELS_DEFS)
    passage_2022_scoreddocs = TrecScoredDocs('trec-dl-2022-passage-scoreddocs',
        source=passage_2022_scoreddocs_file.gunzip())
    passage_2023_qrels = TrecQrels('trec-dl-2023-passage-qrels',
        source=passage_2023_qrels_file, defs=PASSAGE_QRELS_DEFS)
    passage_2023_scoreddocs = TrecScoredDocs('trec-dl-2023-passage-scoreddocs',
        source=passage_2023_scoreddocs_file.gunzip())

    document_2021_qrels = TrecQrels('trec-dl-2021-document-qrels',
        source=document_2021_qrels_file, defs=DOCUMENT_QRELS_DEFS)
    document_2021_scoreddocs = TrecScoredDocs('trec-dl-2021-document-scoreddocs',
        source=document_2021_scoreddocs_file.gunzip())
    document_2022_qrels = TrecQrels('trec-dl-2022-document-qrels',
        source=document_2022_qrels_file, defs=DOCUMENT_QRELS_DEFS)
    document_2022_scoreddocs = TrecScoredDocs('trec-dl-2022-document-scoreddocs',
        source=document_2022_scoreddocs_file.gunzip())
    document_2023_qrels = TrecQrels('trec-dl-2023-document-qrels',
        source=document_2023_qrels_file, defs=DOCUMENT_QRELS_DEFS)
    document_2023_scoreddocs = TrecScoredDocs('trec-dl-2023-document-scoreddocs',
        source=document_2023_scoreddocs_file.gunzip())

    # Benchmarks
    # -----------------------------------------
    passage_2019 = Benchmark('trec-dl-2019-passage',
        docs=passage_docs, queries=queries_2019, qrels=passage_2019_qrels,
        scoreddocs=passage_2019_scoreddocs,
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2019 passage ranking task.')
    passage_2019_judged = Benchmark('trec-dl-2019-passage-judged',
        derived_from=passage_2019, filter=Filter(queries_with_qrels=True),
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='trec-dl-2019-passage restricted to queries with >= 1 qrel.')

    passage_2020 = Benchmark('trec-dl-2020-passage',
        docs=passage_docs, queries=queries_2020, qrels=passage_2020_qrels,
        scoreddocs=passage_2020_scoreddocs,
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2020 passage ranking task.')
    passage_2020_judged = Benchmark('trec-dl-2020-passage-judged',
        derived_from=passage_2020, filter=Filter(queries_with_qrels=True),
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='trec-dl-2020-passage restricted to queries with >= 1 qrel.')

    document_2019 = Benchmark('trec-dl-2019-document',
        docs=document_docs, queries=queries_2019, qrels=document_2019_qrels,
        scoreddocs=document_2019_scoreddocs,
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2019 document ranking task.')
    document_2019_judged = Benchmark('trec-dl-2019-document-judged',
        derived_from=document_2019, filter=Filter(queries_with_qrels=True),
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='trec-dl-2019-document restricted to queries with >= 1 qrel.')

    document_2020 = Benchmark('trec-dl-2020-document',
        docs=document_docs, queries=queries_2020, qrels=document_2020_qrels,
        scoreddocs=document_2020_scoreddocs,
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2020 document ranking task.')
    document_2020_judged = Benchmark('trec-dl-2020-document-judged',
        derived_from=document_2020, filter=Filter(queries_with_qrels=True),
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='trec-dl-2020-document restricted to queries with >= 1 qrel.')

    passage_2021 = Benchmark('trec-dl-2021-passage',
        docs=passage_v2_docs, queries=queries_2021, qrels=passage_2021_qrels,
        scoreddocs=passage_2021_scoreddocs,
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2021 passage ranking task (MS MARCO v2 passage corpus).')
    passage_2021_judged = Benchmark('trec-dl-2021-passage-judged',
        derived_from=passage_2021, filter=Filter(queries_with_qrels=True),
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='trec-dl-2021-passage restricted to queries with >= 1 qrel.')

    passage_2022 = Benchmark('trec-dl-2022-passage',
        docs=passage_v2_docs, queries=queries_2022, qrels=passage_2022_qrels,
        scoreddocs=passage_2022_scoreddocs,
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2022 passage ranking task (MS MARCO v2 passage corpus).')
    passage_2022_judged = Benchmark('trec-dl-2022-passage-judged',
        derived_from=passage_2022, filter=Filter(queries_with_qrels=True),
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='trec-dl-2022-passage restricted to queries with >= 1 qrel.')

    passage_2023 = Benchmark('trec-dl-2023-passage',
        docs=passage_v2_docs, queries=queries_2023, qrels=passage_2023_qrels,
        scoreddocs=passage_2023_scoreddocs,
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2023 passage ranking task (MS MARCO v2 passage corpus).')
    passage_2023_judged = Benchmark('trec-dl-2023-passage-judged',
        derived_from=passage_2023, filter=Filter(queries_with_qrels=True),
        citation=PASSAGE_CITATION, metrics=MEASURES,
        desc='trec-dl-2023-passage restricted to queries with >= 1 qrel.')

    document_2021 = Benchmark('trec-dl-2021-document',
        docs=document_v2_docs, queries=queries_2021, qrels=document_2021_qrels,
        scoreddocs=document_2021_scoreddocs,
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2021 document ranking task (MS MARCO v2 document corpus).')
    document_2021_judged = Benchmark('trec-dl-2021-document-judged',
        derived_from=document_2021, filter=Filter(queries_with_qrels=True),
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='trec-dl-2021-document restricted to queries with >= 1 qrel.')

    document_2022 = Benchmark('trec-dl-2022-document',
        docs=document_v2_docs, queries=queries_2022, qrels=document_2022_qrels,
        scoreddocs=document_2022_scoreddocs,
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2022 document ranking task (MS MARCO v2 document corpus).')
    document_2022_judged = Benchmark('trec-dl-2022-document-judged',
        derived_from=document_2022, filter=Filter(queries_with_qrels=True),
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='trec-dl-2022-document restricted to queries with >= 1 qrel.')

    document_2023 = Benchmark('trec-dl-2023-document',
        docs=document_v2_docs, queries=queries_2023, qrels=document_2023_qrels,
        scoreddocs=document_2023_scoreddocs,
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='TREC Deep Learning 2023 document ranking task (MS MARCO v2 document corpus).')
    document_2023_judged = Benchmark('trec-dl-2023-document-judged',
        derived_from=document_2023, filter=Filter(queries_with_qrels=True),
        citation=DOCUMENT_CITATION, metrics=MEASURES,
        desc='trec-dl-2023-document restricted to queries with >= 1 qrel.')


# Registration
# -----------------------------------------
irds.register(
    passage_2019, passage_2019_judged,
    passage_2020, passage_2020_judged,
    document_2019, document_2019_judged,
    document_2020, document_2020_judged,
    passage_2021, passage_2021_judged,
    passage_2022, passage_2022_judged,
    passage_2023, passage_2023_judged,
    document_2021, document_2021_judged,
    document_2022, document_2022_judged,
    document_2023, document_2023_judged,
)
