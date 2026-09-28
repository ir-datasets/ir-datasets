"""TREC Million Query track (2007-2009) -- a v2 dataset family.

2007-2008 judge ``gov2`` (imported by reference from ``gov2.py``, same
cross-file pattern as ``trec_adhoc.py`` importing ``docs`` from
``disks45.py``); 2009 judges ClueWeb09 (imported by reference from
``clueweb09.py``, the same full-multilingual ``docs`` collection, not
``docs_en`` -- v1's own ``clueweb09.py`` judges ``trec-mq-2009`` against
``collection``, not ``collection_en``). All three years use per-query
judgments (``TrecPrels``, not ``TrecQrels`` -- includes the sampling method
and inclusion probability used to build unbiased effectiveness estimates)
rather than exhaustive ones. ``TrecPrels`` has no general-purpose v2 format
node (as narrow as ``CodecQueries``), so it goes through a local ``Parser``
wrapper around the unmodified v1 class, same pattern as ``codec.py``. The
2007/2008 query files share ``TrecColonQueries`` (``qid:text`` per line)
with the Terabyte track's efficiency tasks -- imported from ``trec_tb.py``
by reference rather than duplicated, same cross-file pattern as ``docs``
above; 2009's queries are ``TrecColonQueries`` too, but latin1-encoded like
the others, so it reuses the same parser class.
"""
from ir_datasets.datasets.gov2 import QREL_DEFS
from ir_datasets.formats import TrecPrels as _V1TrecPrels
from ir_datasets.v2 import Benchmark, QrelTable, QueryTable, Resource, Source, irds
from ir_datasets.v2.datasets.clueweb09 import docs as cw09_docs
from ir_datasets.v2.datasets.clueweb09 import DUA as CW09_DUA
from ir_datasets.v2.datasets.gov2 import DUA, docs
from ir_datasets.v2.datasets.trec_tb import _TrecColonQueriesParser
from ir_datasets.v2.formats import Parser

CITATION_MQ_2007 = 'Allen2007MQ'
CITATION_MQ_2008 = 'Allen2008MQ'
CITATION_MQ_2009 = 'Carterette2009MQ'

# 2009's prels use a 3-level scale (v1's clueweb09.py QREL_DEFS_09), unlike
# 2007/2008's gov2-scale QREL_DEFS above.
QREL_DEFS_09 = {
    2: 'highly relevant',
    1: 'relevant',
    0: 'not relevant',
}


class _TrecPrelsParser(Parser):
    name = 'TrecPrels'

    def build(self, source, node):
        return _V1TrecPrels(source, node.defs or {})


with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    mq2007_queries_file = Resource('trec-mq-2007-queries.gz',
        sources=['https://trec.nist.gov/data/million.query/07/07-million-query-topics.1-10000.gz', Source.irds()],
        md5='db64470f08450c6b15a9ee2c7eac2f9b',
        size=143_691,
    )
    mq2007_qrels_file = Resource('trec-mq-2007-qrels.txt',
        sources=['https://trec.nist.gov/data/million.query/07/07.prels', Source.irds()],
        md5='4f930d85442ac74cc00b56a4252f1f4a',
        size=2_749_725,
    )
    mq2008_queries_file = Resource('trec-mq-2008-queries.gz',
        sources=['https://trec.nist.gov/data/million.query/08/08.million-query-topics.10001-20000.gz', Source.irds()],
        md5='fc8fc0e92ae9bc1d16756534ac682058',
        size=162_791,
    )
    mq2008_qrels_file = Resource('trec-mq-2008-qrels.tar.gz',
        sources=['https://trec.nist.gov/data/million.query/08/2008.RC1.tgz', Source.irds()],
        md5='dae403e1834e87cba1babbef71b73714',
        size=459_172,
    )

    # Tables
    # -----------------------------------------
    mq2007_queries = QueryTable('trec-mq-2007-queries',
        source=mq2007_queries_file.gunzip(),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=10_000)
    mq2007_qrels = QrelTable('trec-mq-2007-qrels',
        source=mq2007_qrels_file,
        parser=_TrecPrelsParser(),
        defs=QREL_DEFS, count_hint=73_015)

    mq2008_queries = QueryTable('trec-mq-2008-queries',
        source=mq2008_queries_file.gunzip(),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=10_000)
    mq2008_qrels = QrelTable('trec-mq-2008-qrels',
        source=mq2008_qrels_file.member('2008.RC1/prels'),
        parser=_TrecPrelsParser(),
        defs=QREL_DEFS, count_hint=15_211)

    # Benchmarks
    # -----------------------------------------
    trec_mq_2007 = Benchmark('trec-mq-2007',
        docs=docs, queries=mq2007_queries, qrels=mq2007_qrels,
        citation=CITATION_MQ_2007,
        desc='TREC 2007 Million Query Track: a large query sample with shallow, per-query '
             'probabilistic judgments (prels) for unbiased effectiveness estimation.')
    trec_mq_2008 = Benchmark('trec-mq-2008',
        docs=docs, queries=mq2008_queries, qrels=mq2008_qrels,
        citation=CITATION_MQ_2008,
        desc='TREC 2008 Million Query Track: a large query sample with shallow, per-query '
             'probabilistic judgments (prels) for unbiased effectiveness estimation.')

with irds.defaults(dua=CW09_DUA, lang='en'):
    # Files
    # -----------------------------------------
    mq2009_queries_file = Resource('trec-mq-2009-queries.gz',
        sources=['https://trec.nist.gov/data/million.query/09/09.mq.topics.20001-60000.gz', Source.irds()],
        md5='6347147d4d6c847f0423709140a7b10d',
        size=437_150,
    )
    mq2009_qrels_file = Resource('trec-mq-2009-qrels.gz',
        sources=['https://trec.nist.gov/data/million.query/09/prels.20001-60000.gz', Source.irds()],
        md5='e67f45d3060e20667596f37e34d696c8',
        size=323_113,
    )

    # Tables
    # -----------------------------------------
    mq2009_queries = QueryTable('trec-mq-2009-queries',
        source=mq2009_queries_file.gunzip(),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=40_000)
    mq2009_qrels = QrelTable('trec-mq-2009-qrels',
        source=mq2009_qrels_file.gunzip(),
        parser=_TrecPrelsParser(),
        defs=QREL_DEFS_09, count_hint=34_534)

    # Benchmarks
    # -----------------------------------------
    trec_mq_2009 = Benchmark('trec-mq-2009',
        docs=cw09_docs, queries=mq2009_queries, qrels=mq2009_qrels,
        citation=CITATION_MQ_2009,
        desc='TREC 2009 Million Query Track: a large query sample with shallow, per-query '
             'probabilistic judgments (prels) for unbiased effectiveness estimation, over '
             'the full multilingual ClueWeb09 collection.')


# Registration
# -----------------------------------------
irds.register(
    trec_mq_2007, trec_mq_2008, trec_mq_2009,
)
