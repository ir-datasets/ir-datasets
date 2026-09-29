"""TREC Terabyte track -- a v2 dataset family.

Judges ``gov2`` (imported by reference from ``gov2.py``, same cross-file
pattern as ``trec_adhoc.py`` importing ``docs`` from ``disks45.py``).

Three TREC Terabyte years: 2004 (ad hoc only), 2005 (ad hoc, named-page, and
an efficiency task whose 50,000 queries are a search engine log that happens
to embed the 50 ad hoc topics -- ``EFF_MAP_05`` maps the ad hoc topic ids
onto their ids within that log, and the qrels are the same ad hoc qrels
re-keyed through that map via v1's ``RewriteQids``, not a separate judgment
file), and 2006 (ad hoc, named-page, and a larger efficiency task with
several named streams, only some of which carry judged queries via
``EFF_MAP_06``). ``TrecColonQueries`` (``qid:text`` per line, used by the
efficiency tasks) has no general-purpose v2 format node (as narrow as
``CodecQueries``), so it goes through a local ``Parser`` wrapper around the
unmodified v1 class, same pattern as ``codec.py`` -- also imported into
``trec_mq.py``, which judges this same corpus and shares that query file
shape for its own logs.

Benchmarks are flat-named off v1's ``gov2/trec-tb-2005/named-page`` etc.
ids (``trec-tb-2005-named-page``, ...) per the flat-naming rule -- the
corpus relationship is already the ``derived_from`` edge, not the name.
"""
from ir_datasets.datasets.gov2 import (
    EFF_MAP_05, EFF_MAP_06, NAMED_PAGE_QREL_DEFS, NAMED_PAGE_QTYPE_MAP,
    QREL_DEFS, RewriteQids as _V1RewriteQids,
)
from ir_datasets.formats import GenericQuery
from ir_datasets.formats import TrecColonQueries as _V1TrecColonQueries
from ir_datasets.formats import TrecQrels as _V1TrecQrels
from ir_datasets.v2 import (
    Benchmark, QrelTable, QueryTable, Resource, Source, TrecQrels, TrecQueries, irds,
)
from ir_datasets.v2.datasets.gov2 import DUA, docs
from ir_datasets.v2.formats import Parser

CITATION_TB_2004 = 'dblp:conf/trec/ClarkeCS04'
CITATION_TB_2005 = 'dblp:conf/trec/ClarkeSS05'
CITATION_TB_2006 = 'dblp:conf/trec/ButtcherCS06'


class _TrecColonQueriesParser(Parser):
    name = 'TrecColonQueries'

    def __init__(self, encoding=None):
        self.encoding = encoding

    def build(self, source, node):
        return _V1TrecColonQueries(source, encoding=self.encoding, lang=node.lang)


class _RewriteQidsParser(Parser):
    """A qrels table that is another qrels file's records, re-keyed through a
    fixed query-id map -- the efficiency tasks' judgments are the ad hoc
    qrels for the same year, re-expressed under the query ids the ad hoc
    topics happen to have within the much larger efficiency query log (see
    ``EFF_MAP_05``/``EFF_MAP_06`` and the module docstring)."""
    name = 'Gov2RewriteQids'

    def __init__(self, qid_map):
        self.qid_map = qid_map

    def build(self, source, node):
        return _V1RewriteQids(_V1TrecQrels(source, node.defs or {}), self.qid_map)


with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    tb2004_queries_file = Resource('trec-tb-2004-queries.txt',
        sources=['https://trec.nist.gov/data/terabyte/04/04topics.701-750.txt', Source.irds()],
        md5='18b390335e440d099f3d64bef81708be',
        size=21_236,
    )
    tb2004_qrels_file = Resource('trec-tb-2004-qrels.txt',
        sources=['https://trec.nist.gov/data/terabyte/04/04.qrels.12-Nov-04', Source.irds()],
        md5='228e4b0c466b1778a01b3337f8774fb6',
        size=1_475_219,
    )
    tb2005_queries_file = Resource('trec-tb-2005-queries.txt',
        sources=['https://trec.nist.gov/data/terabyte/05/05.topics.751-800.txt', Source.irds()],
        md5='f0fb2603c7d89425965e5aaa104ddca6',
        size=24_822,
    )
    tb2005_qrels_file = Resource('trec-tb-2005-qrels.txt',
        sources=['https://trec.nist.gov/data/terabyte/05/05.adhoc_qrels', Source.irds()],
        md5='87f2af26215f092c948249771c8607f6',
        size=1_150_486,
    )
    tb2005_np_queries_file = Resource('trec-tb-2005-named-page-queries.txt',
        sources=['https://trec.nist.gov/data/terabyte/05/05.np_topics.601-872.final.txt', Source.irds()],
        md5='266444c58e3567250f56df5c6a79670d',
        size=20_987,
    )
    tb2005_np_qrels_file = Resource('trec-tb-2005-named-page-qrels.txt',
        sources=['https://trec.nist.gov/data/terabyte/05/05.np_qrels', Source.irds()],
        md5='0b0f73650d1297a7e5572576a4b93d28',
        size=297_947,
    )
    tb2005_eff_queries_file = Resource('trec-tb-2005-efficiency-queries.gz',
        sources=['https://trec.nist.gov/data/terabyte/05/05.efficiency_topics.gz', Source.irds()],
        md5='034a21c9dd956f3b7fb4f162782c9909',
        size=554_590,
    )
    tb2006_queries_file = Resource('trec-tb-2006-queries.txt',
        sources=['https://trec.nist.gov/data/terabyte/06/06.topics.801-850.txt', Source.irds()],
        md5='6e23a748c060ef5be64dbcc65245072f',
        size=27_791,
    )
    tb2006_qrels_file = Resource('trec-tb-2006-qrels.txt',
        sources=['https://trec.nist.gov/data/terabyte/06/qrels.tb06.top50', Source.irds()],
        md5='1b1dfd769ff00d9e8ec4530c64221543',
        size=812_484,
    )
    tb2006_np_queries_file = Resource('trec-tb-2006-named-page-queries.txt',
        sources=['https://trec.nist.gov/data/terabyte/06/06.np_topics.901-1081.txt', Source.irds()],
        md5='811a53107b4445a9955e7376d90a1eec',
        size=13_224,
    )
    tb2006_np_qrels_file = Resource('trec-tb-2006-named-page-qrels.txt',
        sources=['https://trec.nist.gov/data/terabyte/06/qrels.tb06.np', Source.irds()],
        md5='f9f7d07de3070eafc08989dd98d1fab8',
        size=60_528,
    )
    tb2006_eff_queries_file = Resource('trec-tb-2006-efficiency-queries.tar.gz',
        sources=['https://trec.nist.gov/data/terabyte/06/06.efficiency_topics.tar.gz', Source.irds()],
        md5='e8599a08af5b3f036c203957f5b82de8',
        size=3_015_007,
    )

    # Tables
    # -----------------------------------------
    tb2004_queries = TrecQueries('trec-tb-2004-queries', source=tb2004_queries_file, count_hint=50)
    tb2004_qrels = TrecQrels('trec-tb-2004-qrels', source=tb2004_qrels_file, defs=QREL_DEFS, count_hint=58_077)

    tb2005_queries = TrecQueries('trec-tb-2005-queries', source=tb2005_queries_file, count_hint=50)
    tb2005_qrels = TrecQrels('trec-tb-2005-qrels', source=tb2005_qrels_file, defs=QREL_DEFS, count_hint=45_291)
    tb2005_np_queries = TrecQueries('trec-tb-2005-named-page-queries',
        source=tb2005_np_queries_file, qtype=GenericQuery, qtype_map=NAMED_PAGE_QTYPE_MAP, count_hint=252)
    tb2005_np_qrels = TrecQrels('trec-tb-2005-named-page-qrels',
        source=tb2005_np_qrels_file, defs=NAMED_PAGE_QREL_DEFS, count_hint=11_729)
    tb2005_eff_queries = QueryTable('trec-tb-2005-efficiency-queries',
        source=tb2005_eff_queries_file.gunzip(),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=50_000)
    tb2005_eff_qrels = QrelTable('trec-tb-2005-efficiency-qrels',
        source=tb2005_qrels_file,
        parser=_RewriteQidsParser(EFF_MAP_05),
        defs=QREL_DEFS, count_hint=45_291)

    tb2006_queries = TrecQueries('trec-tb-2006-queries', source=tb2006_queries_file, count_hint=50)
    tb2006_qrels = TrecQrels('trec-tb-2006-qrels', source=tb2006_qrels_file, defs=QREL_DEFS, count_hint=31_984)
    tb2006_np_queries = TrecQueries('trec-tb-2006-named-page-queries',
        source=tb2006_np_queries_file, qtype=GenericQuery, qtype_map=NAMED_PAGE_QTYPE_MAP, count_hint=181)
    tb2006_np_qrels = TrecQrels('trec-tb-2006-named-page-qrels',
        source=tb2006_np_qrels_file, defs=NAMED_PAGE_QREL_DEFS, count_hint=2_361)

    tb2006_eff_all_queries = QueryTable('trec-tb-2006-efficiency-queries',
        source=tb2006_eff_queries_file.member('06.efficiency_topics.all'),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=100_000)
    tb2006_eff_all_qrels = QrelTable('trec-tb-2006-efficiency-qrels',
        source=tb2006_qrels_file,
        parser=_RewriteQidsParser(EFF_MAP_06),
        defs=QREL_DEFS, count_hint=31_984)
    tb2006_eff_10k_queries = QueryTable('trec-tb-2006-efficiency-10k-queries',
        source=tb2006_eff_queries_file.member('06.efficiency_topics.10k'),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=10_000)
    tb2006_eff_stream1_queries = QueryTable('trec-tb-2006-efficiency-stream1-queries',
        source=tb2006_eff_queries_file.member('06.efficiency_topics.stream-1'),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=25_000)
    tb2006_eff_stream2_queries = QueryTable('trec-tb-2006-efficiency-stream2-queries',
        source=tb2006_eff_queries_file.member('06.efficiency_topics.stream-2'),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=25_000)
    tb2006_eff_stream3_queries = QueryTable('trec-tb-2006-efficiency-stream3-queries',
        source=tb2006_eff_queries_file.member('06.efficiency_topics.stream-3'),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=25_000)
    tb2006_eff_stream3_qrels = QrelTable('trec-tb-2006-efficiency-stream3-qrels',
        source=tb2006_qrels_file,
        parser=_RewriteQidsParser(EFF_MAP_06),
        defs=QREL_DEFS, count_hint=31_984)
    tb2006_eff_stream4_queries = QueryTable('trec-tb-2006-efficiency-stream4-queries',
        source=tb2006_eff_queries_file.member('06.efficiency_topics.stream-4'),
        parser=_TrecColonQueriesParser(encoding='latin1'),
        count_hint=25_000)

    # Benchmarks
    # -----------------------------------------
    trec_tb_2004 = Benchmark('trec-tb-2004',
        docs=docs, queries=tb2004_queries, qrels=tb2004_qrels,
        citation=CITATION_TB_2004,
        desc='TREC Terabyte Track 2004 ad hoc ranking benchmark: 50 queries with deep relevance judgments.')

    trec_tb_2005 = Benchmark('trec-tb-2005',
        docs=docs, queries=tb2005_queries, qrels=tb2005_qrels,
        citation=CITATION_TB_2005,
        desc='TREC Terabyte Track 2005 ad hoc ranking benchmark: 50 queries with deep relevance judgments.')
    trec_tb_2005_np = Benchmark('trec-tb-2005-named-page',
        docs=docs, queries=tb2005_np_queries, qrels=tb2005_np_qrels,
        citation=CITATION_TB_2005,
        desc='TREC Terabyte Track 2005 named-page ranking benchmark: 252 queries with titles that '
             'resemble bookmark labels; judgments include near-duplicate pages and other pages that '
             'may satisfy the bookmark label.')
    trec_tb_2005_eff = Benchmark('trec-tb-2005-efficiency',
        docs=docs, queries=tb2005_eff_queries, qrels=tb2005_eff_qrels,
        citation=CITATION_TB_2005,
        desc='TREC Terabyte Track 2005 efficiency ranking benchmark: 50,000 queries from a search '
             'engine log, including the 50 topics from trec-tb-2005 (only those have judgments).')

    trec_tb_2006 = Benchmark('trec-tb-2006',
        docs=docs, queries=tb2006_queries, qrels=tb2006_qrels,
        citation=CITATION_TB_2006,
        desc='TREC Terabyte Track 2006 ad hoc ranking benchmark: 50 queries with deep relevance judgments.')
    trec_tb_2006_np = Benchmark('trec-tb-2006-named-page',
        docs=docs, queries=tb2006_np_queries, qrels=tb2006_np_qrels,
        citation=CITATION_TB_2006,
        desc='TREC Terabyte Track 2006 named-page ranking benchmark: 181 queries with titles that '
             'resemble bookmark labels; judgments include near-duplicate pages and other pages that '
             'may satisfy the bookmark label.')
    trec_tb_2006_eff = Benchmark('trec-tb-2006-efficiency',
        docs=docs, queries=tb2006_eff_all_queries, qrels=tb2006_eff_all_qrels,
        citation=CITATION_TB_2006,
        desc='TREC Terabyte Track 2006 efficiency ranking benchmark: 100,000 queries from a search '
             'engine log, including the 50 topics from trec-tb-2006 (only those have judgments).')
    trec_tb_2006_eff_10k = Benchmark('trec-tb-2006-efficiency-10k',
        docs=docs, queries=tb2006_eff_10k_queries,
        citation=CITATION_TB_2006,
        desc='Small 10,000-query stream from trec-tb-2006-efficiency (unjudged).')
    trec_tb_2006_eff_stream1 = Benchmark('trec-tb-2006-efficiency-stream1',
        docs=docs, queries=tb2006_eff_stream1_queries,
        citation=CITATION_TB_2006,
        desc='Stream 1 of trec-tb-2006-efficiency (25,000 queries, unjudged).')
    trec_tb_2006_eff_stream2 = Benchmark('trec-tb-2006-efficiency-stream2',
        docs=docs, queries=tb2006_eff_stream2_queries,
        citation=CITATION_TB_2006,
        desc='Stream 2 of trec-tb-2006-efficiency (25,000 queries, unjudged).')
    trec_tb_2006_eff_stream3 = Benchmark('trec-tb-2006-efficiency-stream3',
        docs=docs, queries=tb2006_eff_stream3_queries, qrels=tb2006_eff_stream3_qrels,
        citation=CITATION_TB_2006,
        desc='Stream 3 of trec-tb-2006-efficiency (25,000 queries, including the judged ad hoc topics).')
    trec_tb_2006_eff_stream4 = Benchmark('trec-tb-2006-efficiency-stream4',
        docs=docs, queries=tb2006_eff_stream4_queries,
        citation=CITATION_TB_2006,
        desc='Stream 4 of trec-tb-2006-efficiency (25,000 queries, unjudged).')


# Registration
# -----------------------------------------
irds.register(
    trec_tb_2004,
    trec_tb_2005, trec_tb_2005_np, trec_tb_2005_eff,
    trec_tb_2006, trec_tb_2006_np, trec_tb_2006_eff,
    trec_tb_2006_eff_10k, trec_tb_2006_eff_stream1, trec_tb_2006_eff_stream2,
    trec_tb_2006_eff_stream3, trec_tb_2006_eff_stream4,
)
