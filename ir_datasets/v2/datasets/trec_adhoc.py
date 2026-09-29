"""TREC 7 & TREC 8 ad hoc tracks -- a v2 dataset family.

Both years judge the same ``disks45-nocr`` corpus (imported by reference
from ``disks45.py``, not re-declared -- same cross-file pattern
``trec_dl.py`` uses for MS MARCO's docs Tables) against their own topics and
qrels. TREC 7's qrels are published as five separately gzip'd parts inside
one tar (``source=[...]``, a list -- see ``nodes.source_resources``, which
walks a list of pipelines the same as a single one); TREC 8's qrels are one
plain member of its own tar, no further gzip.
"""
from ir_datasets.v2 import Benchmark, Resource, Source, TrecQrels, TrecQueries, irds
from ir_datasets.v2.datasets.disks45 import DUA, docs

CITATION_TREC7 = 'Voorhees1998Trec7'
CITATION_TREC8 = 'dblp:conf/trec/VoorheesH99'

QREL_DEFS = {
    1: 'relevant',
    0: 'not relevant',
}

with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    trec7_queries_file = Resource('trec-adhoc-7-queries.gz',
        sources=['https://trec.nist.gov/data/topics_eng/topics.351-400.gz', Source.mirror()],
        md5='fdee3f7e37e173fd6fcdc00fbe1fc671',
        size=7_400,
    )
    trec7_qrels_file = Resource('trec-adhoc-7-qrels.tar.gz',
        sources=['https://trec.nist.gov/data/qrels_eng/qrels.trec7.adhoc.parts1-5.tar.gz', Source.mirror()],
        md5='43def30d4f4b33a830ae67e3dce19023',
        size=307_120,
    )
    trec8_queries_file = Resource('trec-adhoc-8-queries.gz',
        sources=['https://trec.nist.gov/data/topics_eng/topics.401-450.gz', Source.mirror()],
        md5='daaafb700eed76f61a6e9e4b0dcc40c8',
        size=6_946,
    )
    trec8_qrels_file = Resource('trec-adhoc-8-qrels.tar.gz',
        sources=['https://trec.nist.gov/data/qrels_eng/qrels.trec8.adhoc.parts1-5.tar.gz', Source.mirror()],
        md5='ce1cfa80b29746d2a5eeddab268d4f6a',
        size=325_935,
    )

    # Tables
    # -----------------------------------------
    trec7_queries = TrecQueries('trec-adhoc-7-queries', source=trec7_queries_file.gunzip(), count_hint=50,
        citation=CITATION_TREC7)
    trec7_qrels = TrecQrels('trec-adhoc-7-qrels',
        source=[
            trec7_qrels_file.member('qrels.trec7.adhoc.part1.gz').gunzip(),
            trec7_qrels_file.member('qrels.trec7.adhoc.part2.gz').gunzip(),
            trec7_qrels_file.member('qrels.trec7.adhoc.part3.gz').gunzip(),
            trec7_qrels_file.member('qrels.trec7.adhoc.part4.gz').gunzip(),
            trec7_qrels_file.member('qrels.trec7.adhoc.part5.gz').gunzip(),
        ], defs=QREL_DEFS, count_hint=80_345, citation=CITATION_TREC7)

    trec8_queries = TrecQueries('trec-adhoc-8-queries', source=trec8_queries_file.gunzip(), count_hint=50,
        citation=CITATION_TREC8)
    trec8_qrels = TrecQrels('trec-adhoc-8-qrels',
        source=trec8_qrels_file.member('qrels.trec8.adhoc.parts1-5'),
        defs=QREL_DEFS, count_hint=86_830, citation=CITATION_TREC8)

    # Benchmarks
    # -----------------------------------------
    trec7 = Benchmark('trec-adhoc-7',
        docs=docs, queries=trec7_queries, qrels=trec7_qrels,
        citation=CITATION_TREC7,
        desc='TREC 7 (1998) ad hoc retrieval track.')
    trec8 = Benchmark('trec-adhoc-8',
        docs=docs, queries=trec8_queries, qrels=trec8_qrels,
        citation=CITATION_TREC8,
        desc='TREC 8 (1999) ad hoc retrieval track.')


# Registration
# -----------------------------------------
irds.register(trec7, trec8)
