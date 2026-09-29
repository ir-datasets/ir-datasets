"""TREC Web Track -- a v2 dataset family, covering all three corpora the
track has run over: gov (2002-2004), ClueWeb09 (2009-2012), and ClueWeb12
(2013-2014) -- all in one file since it's the same track throughout, just
switching to a newer corpus each generation; splitting by corpus would cut
across the actual track/benchmark boundary the "split TREC tasks into their
own file" convention cares about.

2002-2004 judge ``gov-docs`` (imported by reference from ``gov.py``, same
cross-file pattern as ``trec_adhoc.py`` importing ``docs`` from
``disks45.py``):

- 2002 and 2003 each additionally define a named-page finding task, over a
  disjoint topic/qrel set from that year's ad hoc task (not a filtered view
  of it) -- ``trec-web-2002-named-page``/``trec-web-2003-named-page``.
- 2004's single benchmark mixes topic distillation, homepage finding and
  named page finding queries in one topic file (v1's ``WEB04_QTYPE_MAP``),
  with no separate named-page split.
- 2002's queries/qrels are gzip'd (``.gunzip()``); 2003's and 2004's are
  published as plain text, same split v1's code shows (``GzipExtract`` used
  only for the 2002 downloads).

2009-2012 judge ClueWeb09 (imported by reference from ``clueweb09.py``):
each year has an ad hoc benchmark and a diversity (subtopic-level) one, each
run against both the full English subset (``docs_en``) and the smaller
``catb`` subset (``docs_catb``) -- v1's ``CatBQrelFilter`` restricts the same
ad hoc/diversity qrels to just the segments present in catb, rather than
being a separately judged qrel set, so the catb variants share one
``Resource``/query set with their non-catb counterpart and only wrap the
qrels parser. 2009's qrels are ``TrecPrels``/``TrecSubQrels`` over a 3-level
scale (``QREL_DEFS_09``/``SQREL_DEFS_09``); 2010-2012 use a 6-level scale
(``QREL_DEFS_CW``, imported from ``clueweb12.py`` -- the same scale
ClueWeb12's own 2013-2014 web-track years use below, and the one that module
declares as the vocabulary's source of truth).

2013-2014 judge ClueWeb12 (imported by reference from ``clueweb12.py``),
with no catb-style corpus subsetting -- just an ad hoc and a diversity
benchmark per year, same 6-level ``QREL_DEFS_CW`` scale as 2010-2012.

All of the above use ``TrecWebTrackQuery`` (title + description + type +
subtopics), imported from v1's ``clueweb09`` module (also reused unmodified
by v1's own ``clueweb12.py``, so importing it from one place here is no
behavior change).

Benchmarks are flat-named (``trec-web-2002``, ``trec-web-2009-catb``, ...),
never nested under a corpus name, per the flat-naming rule -- same as
``trec_adhoc.py``'s ``trec-adhoc-7``/``trec-adhoc-8``.
"""
from ir_datasets.datasets.clueweb09 import CatBQrelFilter as _V1CatBQrelFilter
from ir_datasets.datasets.clueweb09 import TrecWebTrackQuery
from ir_datasets.datasets.clueweb12 import QREL_DEFS as QREL_DEFS_CW
from ir_datasets.datasets.gov import GovWeb02Query
from ir_datasets.formats import GenericQuery
from ir_datasets.formats import TrecPrels as _V1TrecPrels
from ir_datasets.formats import TrecSubQrels as _V1TrecSubQrels
from ir_datasets.v2 import Benchmark, QrelTable, QueryTable, Resource, Source, TrecQrels, TrecQueries, irds
from ir_datasets.v2.datasets.clueweb09 import docs_catb as cw09_docs_catb
from ir_datasets.v2.datasets.clueweb09 import docs_en as cw09_docs_en
from ir_datasets.v2.datasets.clueweb09 import DUA as CW09_DUA
from ir_datasets.v2.datasets.clueweb12 import docs as cw12_docs
from ir_datasets.v2.datasets.clueweb12 import DUA as CW12_DUA
from ir_datasets.v2.datasets.gov import DUA, docs
from ir_datasets.v2.formats import Parser

QREL_DEFS = {
    1: 'Relevant',
    0: 'Not Relevant',
}

NAMED_PAGE_QREL_DEFS = {
    1: 'Name refers to this page',
}

QREL_DEFS_09 = {
    2: 'highly relevant',
    1: 'relevant',
    0: 'not relevant',
}
SQREL_DEFS_09 = {
    1: 'relevant',
    0: 'not relevant',
}
# Same 6-level scale ClueWeb09's later years and ClueWeb12's own web-track
# years use -- imported above as QREL_DEFS_CW from v1's clueweb12 module
# (which happens to declare the identical dict clueweb09.py's v1 module
# does; either would do, clueweb12's is picked since that's the newer
# corpus's own web-track years below).


class _TrecXmlQueriesParser(Parser):
    name = 'TrecXmlQueries'

    def build(self, source, node):
        from ir_datasets.formats import TrecXmlQueries
        return TrecXmlQueries(source, qtype=TrecWebTrackQuery, lang=node.lang)


class _TrecPrelsParser(Parser):
    name = 'TrecPrels'

    def build(self, source, node):
        return _V1TrecPrels(source, node.defs or {})


class _TrecSubQrelsParser(Parser):
    name = 'TrecSubQrels'

    def build(self, source, node):
        return _V1TrecSubQrels(source, node.defs or {})


class _TrecQrelsParser(Parser):
    """Same shape as the v2 ``TrecQrels`` format node, but as a local
    ``Parser`` instance so it can be composed with ``_CatBQrelsParser``
    below (that wrapper needs an inner *parser*, not a pre-built table)."""
    name = 'TrecQrels'

    def build(self, source, node):
        from ir_datasets.formats import TrecQrels as _V1TrecQrels
        return _V1TrecQrels(source, node.defs or {})


class _CatBQrelsParser(Parser):
    """Wraps another qrels parser's v1 handler in v1's ``CatBQrelFilter`` --
    the catb benchmarks are a segment-level *filter* over the same qrels as
    their non-catb counterpart, not a separately judged qrel set (see the
    module docstring)."""
    name = 'ClueWeb09CatBQrels'

    def __init__(self, inner):
        self.inner = inner

    def build(self, source, node):
        return _V1CatBQrelFilter(self.inner.build(source, node))

NAMED_PAGE_QTYPE_MAP = {
    '<num> *(Number:)? *NP': 'query_id', # Remove NP prefix from QIDs
    '<desc> *(Description:)?': 'text',
}

WEB03_QTYPE_MAP = {
    '<num> *(Number:)? *TD': 'query_id', # Remove TD prefix from QIDs
    '<title>': 'title',
    '<desc> *(Description:)?': 'description',
}

WEB04_QTYPE_MAP = {
    '<num> *(Number:)? *WT04-': 'query_id',
    '<title>': 'text',
}

with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    web2002_queries_file = Resource('trec-web-2002-queries.gz',
        sources=['https://trec.nist.gov/data/topics_eng/webtopics_551-600.txt.gz', Source.mirror()],
        md5='133e5d1628684f7a044df86ad08907f0',
        size=6_528,
    )
    web2002_qrels_file = Resource('trec-web-2002-qrels.gz',
        sources=['https://trec.nist.gov/data/qrels_eng/qrels.distillation.txt.gz', Source.mirror()],
        md5='313d1cab9a37aa9b76b6c647cf7151a8',
        size=402_641,
    )
    web2002_np_queries_file = Resource('trec-web-2002-named-page-queries.gz',
        sources=['https://trec.nist.gov/data/topics_eng/webnamed_page_topics.1-150.txt.gz', Source.mirror()],
        md5='00422f1c1f5109d7f609708de071e527',
        size=3_168,
    )
    web2002_np_qrels_file = Resource('trec-web-2002-named-page-qrels.gz',
        sources=['https://trec.nist.gov/data/qrels_eng/qrels.named-page.txt.gz', Source.mirror()],
        md5='ed7e69528faddd1baece4cae41c6f613',
        size=1_649,
    )

    web2003_queries_file = Resource('trec-web-2003-queries.txt',
        sources=['https://trec.nist.gov/data/topics_eng/2003.distillation_topics.1-50.txt', Source.mirror()],
        md5='409e5d16eb8c795945715850c7d26a8e',
        size=8_221,
    )
    web2003_qrels_file = Resource('trec-web-2003-qrels.txt',
        sources=['https://trec.nist.gov/data/qrels_eng/qrels.distillation.2003.txt', Source.mirror()],
        md5='ce11fa22c6f7f5d8048bdc0d104986e5',
        size=1_113_881,
    )
    web2003_np_queries_file = Resource('trec-web-2003-named-page-queries.txt',
        sources=['https://trec.nist.gov/data/topics_eng/2003.named_page_topics.151-450.txt', Source.mirror()],
        md5='0b9bbe2bce309c5bf5754536abaaa0b6',
        size=26_337,
    )
    web2003_np_qrels_file = Resource('trec-web-2003-named-page-qrels.txt',
        sources=['https://trec.nist.gov/data/qrels_eng/qrels.named-page.2003.txt', Source.mirror()],
        md5='e7b05e05fab39862d5f8ad6ebc0c36fd',
        size=8_096,
    )

    web2004_queries_file = Resource('trec-web-2004-queries.txt',
        sources=['https://trec.nist.gov/data/web/Web2004.query.stream.trecformat.txt', Source.mirror()],
        md5='10821f7a000b8bec058097ede39570be',
        size=15_657,
    )
    web2004_qrels_file = Resource('trec-web-2004-qrels.txt',
        sources=['https://trec.nist.gov/data/web/04.qrels.web.mixed.txt', Source.mirror()],
        md5='93daa0e4b4190c84e30d2cce78a0f674',
        size=1_996_931,
    )

    # Tables
    # -----------------------------------------
    web2002_queries = TrecQueries('trec-web-2002-queries', source=web2002_queries_file.gunzip(), count_hint=50, citation='dblp:conf/trec/CraswellH02')
    web2002_qrels = TrecQrels('trec-web-2002-qrels', source=web2002_qrels_file.gunzip(), defs=QREL_DEFS, count_hint=56_650, citation='dblp:conf/trec/CraswellH02')
    web2002_np_queries = TrecQueries('trec-web-2002-named-page-queries',
        source=web2002_np_queries_file.gunzip(),
        qtype=GenericQuery, qtype_map=NAMED_PAGE_QTYPE_MAP,
        count_hint=150, citation='dblp:conf/trec/CraswellH02')
    web2002_np_qrels = TrecQrels('trec-web-2002-named-page-qrels',
        source=web2002_np_qrels_file.gunzip(), defs=NAMED_PAGE_QREL_DEFS, count_hint=170, citation='dblp:conf/trec/CraswellH02')

    web2003_queries = TrecQueries('trec-web-2003-queries',
        source=web2003_queries_file,
        qtype=GovWeb02Query, qtype_map=WEB03_QTYPE_MAP,
        count_hint=50, citation='dblp:conf/trec/CraswellHWW03')
    web2003_qrels = TrecQrels('trec-web-2003-qrels', source=web2003_qrels_file, defs=QREL_DEFS, count_hint=51_062, citation='dblp:conf/trec/CraswellHWW03')
    web2003_np_queries = TrecQueries('trec-web-2003-named-page-queries',
        source=web2003_np_queries_file,
        qtype=GenericQuery, qtype_map=NAMED_PAGE_QTYPE_MAP,
        count_hint=300, citation='dblp:conf/trec/CraswellHWW03')
    web2003_np_qrels = TrecQrels('trec-web-2003-named-page-qrels',
        source=web2003_np_qrels_file, defs=NAMED_PAGE_QREL_DEFS, count_hint=352, citation='dblp:conf/trec/CraswellHWW03')

    web2004_queries = TrecQueries('trec-web-2004-queries',
        source=web2004_queries_file,
        qtype=GenericQuery, qtype_map=WEB04_QTYPE_MAP,
        count_hint=225, citation='dblp:conf/trec/CraswellH04')
    web2004_qrels = TrecQrels('trec-web-2004-qrels', source=web2004_qrels_file, defs=QREL_DEFS, count_hint=88_566, citation='dblp:conf/trec/CraswellH04')

    # Benchmarks
    # -----------------------------------------
    web2002 = Benchmark('trec-web-2002',
        docs=docs, queries=web2002_queries, qrels=web2002_qrels,
        citation='dblp:conf/trec/CraswellH02',
        desc='TREC Web Track 2002 ad hoc (topic distillation) ranking benchmark.')
    web2002_named_page = Benchmark('trec-web-2002-named-page',
        docs=docs, queries=web2002_np_queries, qrels=web2002_np_qrels,
        citation='dblp:conf/trec/CraswellH02',
        desc='TREC Web Track 2002 named page finding benchmark.')

    web2003 = Benchmark('trec-web-2003',
        docs=docs, queries=web2003_queries, qrels=web2003_qrels,
        citation='dblp:conf/trec/CraswellHWW03',
        desc='TREC Web Track 2003 ad hoc (topic distillation) ranking benchmark.')
    web2003_named_page = Benchmark('trec-web-2003-named-page',
        docs=docs, queries=web2003_np_queries, qrels=web2003_np_qrels,
        citation='dblp:conf/trec/CraswellHWW03',
        desc='TREC Web Track 2003 named page finding benchmark.')

    web2004 = Benchmark('trec-web-2004',
        docs=docs, queries=web2004_queries, qrels=web2004_qrels,
        citation='dblp:conf/trec/CraswellH04',
        desc='TREC Web Track 2004 ad hoc ranking benchmark: a mix of topic '
             'distillation, homepage finding, and named page finding queries.')

with irds.defaults(dua=CW09_DUA, lang='en'):
    # Files (ClueWeb09)
    # -----------------------------------------
    web2009_queries_file = Resource('trec-web-2009-queries.xml',
        sources=['https://trec.nist.gov/data/web/09/wt09.topics.full.xml', Source.mirror()],
        md5='52e4a03d32718fa11290286e8e8dff47',
        size=35_853,
    )
    web2009_qrels_adhoc_file = Resource('trec-web-2009-qrels-adhoc.gz',
        sources=['https://trec.nist.gov/data/web/09/prels.1-50.gz', Source.mirror()],
        md5='3afdef86adf3211629182e3380f9e751',
        size=171_396,
    )
    web2009_qrels_all_file = Resource('trec-web-2009-qrels-all.gz',
        sources=['https://trec.nist.gov/data/web/09/qrels.diversity.gz', Source.mirror()],
        md5='0a3fb04bfdaa1551d8960d862e925c9e',
        size=166_538,
    )

    web2010_queries_file = Resource('trec-web-2010-queries.xml',
        sources=['https://trec.nist.gov/data/web/10/wt2010-topics.xml', Source.mirror()],
        md5='8f084cc90c13e4cd66192d3a9585235e',
        size=32_661,
    )
    web2010_qrels_adhoc_file = Resource('trec-web-2010-qrels-adhoc.txt',
        sources=['https://trec.nist.gov/data/web/10/10.adhoc-qrels.final', Source.mirror()],
        md5='8a22083b0370d6ac799e1e779110de06',
        size=837_288,
    )
    web2010_qrels_all_file = Resource('trec-web-2010-qrels-all.txt',
        sources=['https://trec.nist.gov/data/web/10/10.diversity-qrels.final', Source.mirror()],
        md5='0a78c8bf7a809039a1fc9013a4bfe4eb',
        size=297_198,
    )

    web2011_queries_file = Resource('trec-web-2011-queries.xml',
        sources=['https://trec.nist.gov/data/web/11/full-topics.xml', Source.mirror()],
        md5='23914875d80a5d24571d4f458a83c7fa',
        size=29_693,
    )
    web2011_qrels_adhoc_file = Resource('trec-web-2011-qrels-adhoc.txt',
        sources=['https://trec.nist.gov/data/web/11/qrels.adhoc', Source.mirror()],
        md5='7844c8a9cc3a4b6f740d45e56013693d',
        size=659_973,
    )
    web2011_qrels_all_file = Resource('trec-web-2011-qrels-all.txt',
        sources=['https://trec.nist.gov/data/web/11/qrels.diversity', Source.mirror()],
        md5='b88c1d42afbbbc5a4a776dd3f0b905c2',
        size=2_208_947,
    )

    web2012_queries_file = Resource('trec-web-2012-queries.xml',
        sources=['https://trec.nist.gov/data/web/12/full-topics.xml', Source.mirror()],
        md5='a0b8ee33da312a284fda379582b0bc2a',
        size=29_353,
    )
    web2012_qrels_adhoc_file = Resource('trec-web-2012-qrels-adhoc.txt',
        sources=['https://trec.nist.gov/data/web/12/qrels.adhoc', Source.mirror()],
        md5='079723ba3e955269f0de6254c4bec180',
        size=610_948,
    )
    web2012_qrels_all_file = Resource('trec-web-2012-qrels-all.txt',
        sources=['https://trec.nist.gov/data/web/12/qrels.diversity', Source.mirror()],
        md5='bbfde42fc4bc502b19aec5dcc6922faa',
        size=2_124_769,
    )

    # Tables (ClueWeb09)
    # -----------------------------------------
    web2009_queries = QueryTable('trec-web-2009-queries',
        source=web2009_queries_file, parser=_TrecXmlQueriesParser(), count_hint=50, citation='dblp:conf/trec/ClarkeCS09')
    web2009_qrels_adhoc = QrelTable('trec-web-2009-qrels-adhoc',
        source=web2009_qrels_adhoc_file.gunzip(), parser=_TrecPrelsParser(),
        defs=QREL_DEFS_09, count_hint=23_601, citation='dblp:conf/trec/ClarkeCS09')
    web2009_qrels_all = QrelTable('trec-web-2009-qrels-all',
        source=web2009_qrels_all_file.gunzip(), parser=_TrecSubQrelsParser(),
        defs=SQREL_DEFS_09, count_hint=27_964, citation='dblp:conf/trec/ClarkeCS09')
    web2009_catb_qrels_adhoc = QrelTable('trec-web-2009-catb-qrels-adhoc',
        source=web2009_qrels_adhoc_file.gunzip(),
        parser=_CatBQrelsParser(_TrecPrelsParser()),
        defs=QREL_DEFS_09, count_hint=13_118, citation='dblp:conf/trec/ClarkeCS09')
    web2009_catb_qrels_all = QrelTable('trec-web-2009-catb-qrels-all',
        source=web2009_qrels_all_file.gunzip(),
        parser=_CatBQrelsParser(_TrecSubQrelsParser()),
        defs=SQREL_DEFS_09, count_hint=16_347, citation='dblp:conf/trec/ClarkeCS09')

    web2010_queries = QueryTable('trec-web-2010-queries',
        source=web2010_queries_file, parser=_TrecXmlQueriesParser(), count_hint=50, citation='dblp:conf/trec/ClarkeCSC10')
    web2010_qrels_adhoc = QrelTable('trec-web-2010-qrels-adhoc',
        source=web2010_qrels_adhoc_file, parser=_TrecQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=25_329, citation='dblp:conf/trec/ClarkeCSC10')
    web2010_qrels_all = QrelTable('trec-web-2010-qrels-all',
        source=web2010_qrels_all_file, parser=_TrecSubQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=9_006, citation='dblp:conf/trec/ClarkeCSC10')
    web2010_catb_qrels_adhoc = QrelTable('trec-web-2010-catb-qrels-adhoc',
        source=web2010_qrels_adhoc_file,
        parser=_CatBQrelsParser(_TrecQrelsParser()),
        defs=QREL_DEFS_CW, count_hint=15_845, citation='dblp:conf/trec/ClarkeCSC10')
    web2010_catb_qrels_all = QrelTable('trec-web-2010-catb-qrels-all',
        source=web2010_qrels_all_file,
        parser=_CatBQrelsParser(_TrecSubQrelsParser()),
        defs=QREL_DEFS_CW, count_hint=5_522, citation='dblp:conf/trec/ClarkeCSC10')

    web2011_queries = QueryTable('trec-web-2011-queries',
        source=web2011_queries_file, parser=_TrecXmlQueriesParser(), count_hint=50, citation='dblp:conf/trec/ClarkeCSV11')
    web2011_qrels_adhoc = QrelTable('trec-web-2011-qrels-adhoc',
        source=web2011_qrels_adhoc_file, parser=_TrecQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=19_381, citation='dblp:conf/trec/ClarkeCSV11')
    web2011_qrels_all = QrelTable('trec-web-2011-qrels-all',
        source=web2011_qrels_all_file, parser=_TrecSubQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=64_868, citation='dblp:conf/trec/ClarkeCSV11')
    web2011_catb_qrels_adhoc = QrelTable('trec-web-2011-catb-qrels-adhoc',
        source=web2011_qrels_adhoc_file,
        parser=_CatBQrelsParser(_TrecQrelsParser()),
        defs=QREL_DEFS_CW, count_hint=13_081, citation='dblp:conf/trec/ClarkeCSV11')
    web2011_catb_qrels_all = QrelTable('trec-web-2011-catb-qrels-all',
        source=web2011_qrels_all_file,
        parser=_CatBQrelsParser(_TrecSubQrelsParser()),
        defs=QREL_DEFS_CW, count_hint=43_889, citation='dblp:conf/trec/ClarkeCSV11')

    web2012_queries = QueryTable('trec-web-2012-queries',
        source=web2012_queries_file, parser=_TrecXmlQueriesParser(), count_hint=50, citation='dblp:conf/trec/ClarkeCV12')
    web2012_qrels_adhoc = QrelTable('trec-web-2012-qrels-adhoc',
        source=web2012_qrels_adhoc_file, parser=_TrecQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=16_055, citation='dblp:conf/trec/ClarkeCV12')
    web2012_qrels_all = QrelTable('trec-web-2012-qrels-all',
        source=web2012_qrels_all_file, parser=_TrecSubQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=62_394, citation='dblp:conf/trec/ClarkeCV12')
    web2012_catb_qrels_adhoc = QrelTable('trec-web-2012-catb-qrels-adhoc',
        source=web2012_qrels_adhoc_file,
        parser=_CatBQrelsParser(_TrecQrelsParser()),
        defs=QREL_DEFS_CW, count_hint=10_022, citation='dblp:conf/trec/ClarkeCV12')
    web2012_catb_qrels_all = QrelTable('trec-web-2012-catb-qrels-all',
        source=web2012_qrels_all_file,
        parser=_CatBQrelsParser(_TrecSubQrelsParser()),
        defs=QREL_DEFS_CW, count_hint=38_992, citation='dblp:conf/trec/ClarkeCV12')

    # Benchmarks (ClueWeb09)
    # -----------------------------------------
    web2009 = Benchmark('trec-web-2009',
        docs=cw09_docs_en, queries=web2009_queries, qrels=web2009_qrels_adhoc,
        citation='dblp:conf/trec/ClarkeCS09',
        desc='TREC Web Track 2009 ad hoc ranking benchmark (over the full ClueWeb09 English subset).')
    web2009_diversity = Benchmark('trec-web-2009-diversity',
        docs=cw09_docs_en, queries=web2009_queries, qrels=web2009_qrels_all,
        citation='dblp:conf/trec/ClarkeCS09',
        desc='TREC Web Track 2009 diversity (subtopic-level) ranking benchmark '
             '(over the full ClueWeb09 English subset).')
    web2009_catb = Benchmark('trec-web-2009-catb',
        docs=cw09_docs_catb, queries=web2009_queries, qrels=web2009_catb_qrels_adhoc,
        citation='dblp:conf/trec/ClarkeCS09',
        desc='TREC Web Track 2009 ad hoc ranking benchmark, restricted to the ClueWeb09 Category B subset.')
    web2009_catb_diversity = Benchmark('trec-web-2009-catb-diversity',
        docs=cw09_docs_catb, queries=web2009_queries, qrels=web2009_catb_qrels_all,
        citation='dblp:conf/trec/ClarkeCS09',
        desc='TREC Web Track 2009 diversity ranking benchmark, restricted to the ClueWeb09 Category B subset.')

    web2010 = Benchmark('trec-web-2010',
        docs=cw09_docs_en, queries=web2010_queries, qrels=web2010_qrels_adhoc,
        citation='dblp:conf/trec/ClarkeCSC10',
        desc='TREC Web Track 2010 ad hoc ranking benchmark (over the full ClueWeb09 English subset).')
    web2010_diversity = Benchmark('trec-web-2010-diversity',
        docs=cw09_docs_en, queries=web2010_queries, qrels=web2010_qrels_all,
        citation='dblp:conf/trec/ClarkeCSC10',
        desc='TREC Web Track 2010 diversity ranking benchmark (over the full ClueWeb09 English subset).')
    web2010_catb = Benchmark('trec-web-2010-catb',
        docs=cw09_docs_catb, queries=web2010_queries, qrels=web2010_catb_qrels_adhoc,
        citation='dblp:conf/trec/ClarkeCSC10',
        desc='TREC Web Track 2010 ad hoc ranking benchmark, restricted to the ClueWeb09 Category B subset.')
    web2010_catb_diversity = Benchmark('trec-web-2010-catb-diversity',
        docs=cw09_docs_catb, queries=web2010_queries, qrels=web2010_catb_qrels_all,
        citation='dblp:conf/trec/ClarkeCSC10',
        desc='TREC Web Track 2010 diversity ranking benchmark, restricted to the ClueWeb09 Category B subset.')

    web2011 = Benchmark('trec-web-2011',
        docs=cw09_docs_en, queries=web2011_queries, qrels=web2011_qrels_adhoc,
        citation='dblp:conf/trec/ClarkeCSV11',
        desc='TREC Web Track 2011 ad hoc ranking benchmark (over the full ClueWeb09 English subset).')
    web2011_diversity = Benchmark('trec-web-2011-diversity',
        docs=cw09_docs_en, queries=web2011_queries, qrels=web2011_qrels_all,
        citation='dblp:conf/trec/ClarkeCSV11',
        desc='TREC Web Track 2011 diversity ranking benchmark (over the full ClueWeb09 English subset).')
    web2011_catb = Benchmark('trec-web-2011-catb',
        docs=cw09_docs_catb, queries=web2011_queries, qrels=web2011_catb_qrels_adhoc,
        citation='dblp:conf/trec/ClarkeCSV11',
        desc='TREC Web Track 2011 ad hoc ranking benchmark, restricted to the ClueWeb09 Category B subset.')
    web2011_catb_diversity = Benchmark('trec-web-2011-catb-diversity',
        docs=cw09_docs_catb, queries=web2011_queries, qrels=web2011_catb_qrels_all,
        citation='dblp:conf/trec/ClarkeCSV11',
        desc='TREC Web Track 2011 diversity ranking benchmark, restricted to the ClueWeb09 Category B subset.')

    web2012 = Benchmark('trec-web-2012',
        docs=cw09_docs_en, queries=web2012_queries, qrels=web2012_qrels_adhoc,
        citation='dblp:conf/trec/ClarkeCV12',
        desc='TREC Web Track 2012 ad hoc ranking benchmark (over the full ClueWeb09 English subset).')
    web2012_diversity = Benchmark('trec-web-2012-diversity',
        docs=cw09_docs_en, queries=web2012_queries, qrels=web2012_qrels_all,
        citation='dblp:conf/trec/ClarkeCV12',
        desc='TREC Web Track 2012 diversity ranking benchmark (over the full ClueWeb09 English subset).')
    web2012_catb = Benchmark('trec-web-2012-catb',
        docs=cw09_docs_catb, queries=web2012_queries, qrels=web2012_catb_qrels_adhoc,
        citation='dblp:conf/trec/ClarkeCV12',
        desc='TREC Web Track 2012 ad hoc ranking benchmark, restricted to the ClueWeb09 Category B subset.')
    web2012_catb_diversity = Benchmark('trec-web-2012-catb-diversity',
        docs=cw09_docs_catb, queries=web2012_queries, qrels=web2012_catb_qrels_all,
        citation='dblp:conf/trec/ClarkeCV12',
        desc='TREC Web Track 2012 diversity ranking benchmark, restricted to the ClueWeb09 Category B subset.')

with irds.defaults(dua=CW12_DUA, lang='en'):
    # Files (ClueWeb12)
    # -----------------------------------------
    web2013_queries_file = Resource('trec-web-2013-queries.xml',
        sources=['https://trec.nist.gov/data/web/2013/trec2013-topics.xml', Source.mirror()],
        md5='4c0ecdddc8632d3fa8fecb507f19801d',
        size=23_143,
    )
    web2013_qrels_adhoc_file = Resource('trec-web-2013-qrels-adhoc.txt',
        sources=['https://trec.nist.gov/data/web/2013/qrels.adhoc.txt', Source.mirror()],
        md5='44aa6300f9df4a77f7205c574afb9c2d',
        size=492_350,
    )
    web2013_qrels_all_file = Resource('trec-web-2013-qrels-all.txt',
        sources=['https://trec.nist.gov/data/web/2013/qrels.all.txt', Source.mirror()],
        md5='741e76258543ad47ae75030363be13a9',
        size=1_598_265,
    )
    web2014_queries_file = Resource('trec-web-2014-queries.xml',
        sources=['https://trec.nist.gov/data/web/2014/trec2014-topics.xml', Source.mirror()],
        md5='b1bf5c7aa9f6e7026e1558686330744f',
        size=22_873,
    )
    web2014_qrels_adhoc_file = Resource('trec-web-2014-qrels-adhoc.txt',
        sources=['https://trec.nist.gov/data/web/2014/qrels.adhoc.txt', Source.mirror()],
        md5='afa1db71680acf71283adc7846282a44',
        size=491_247,
    )
    web2014_qrels_all_file = Resource('trec-web-2014-qrels-all.txt',
        sources=['https://trec.nist.gov/data/web/2014/qrels.all.txt', Source.mirror()],
        md5='085256d18544cd3e34b9fa9cc29ae513',
        size=1_492_061,
    )

    # Tables (ClueWeb12)
    # -----------------------------------------
    web2013_queries = QueryTable('trec-web-2013-queries',
        source=web2013_queries_file, parser=_TrecXmlQueriesParser(), count_hint=50, citation='dblp:conf/trec/Collins-Thompson13')
    web2013_qrels_adhoc = QrelTable('trec-web-2013-qrels-adhoc',
        source=web2013_qrels_adhoc_file, parser=_TrecQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=14_474, citation='dblp:conf/trec/Collins-Thompson13')
    web2013_qrels_all = QrelTable('trec-web-2013-qrels-all',
        source=web2013_qrels_all_file, parser=_TrecSubQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=46_985, citation='dblp:conf/trec/Collins-Thompson13')

    web2014_queries = QueryTable('trec-web-2014-queries',
        source=web2014_queries_file, parser=_TrecXmlQueriesParser(), count_hint=50, citation='dblp:conf/trec/Collins-Thompson14')
    web2014_qrels_adhoc = QrelTable('trec-web-2014-qrels-adhoc',
        source=web2014_qrels_adhoc_file, parser=_TrecQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=14_432, citation='dblp:conf/trec/Collins-Thompson14')
    web2014_qrels_all = QrelTable('trec-web-2014-qrels-all',
        source=web2014_qrels_all_file, parser=_TrecSubQrelsParser(),
        defs=QREL_DEFS_CW, count_hint=43_840, citation='dblp:conf/trec/Collins-Thompson14')

    # Benchmarks (ClueWeb12)
    # -----------------------------------------
    web2013 = Benchmark('trec-web-2013',
        docs=cw12_docs, queries=web2013_queries, qrels=web2013_qrels_adhoc,
        citation='dblp:conf/trec/Collins-Thompson13',
        desc='TREC Web Track 2013 ad hoc ranking benchmark.')
    web2013_diversity = Benchmark('trec-web-2013-diversity',
        docs=cw12_docs, queries=web2013_queries, qrels=web2013_qrels_all,
        citation='dblp:conf/trec/Collins-Thompson13',
        desc='TREC Web Track 2013 diversity ranking benchmark.')

    web2014 = Benchmark('trec-web-2014',
        docs=cw12_docs, queries=web2014_queries, qrels=web2014_qrels_adhoc,
        citation='dblp:conf/trec/Collins-Thompson14',
        desc='TREC Web Track 2014 ad hoc ranking benchmark.')
    web2014_diversity = Benchmark('trec-web-2014-diversity',
        docs=cw12_docs, queries=web2014_queries, qrels=web2014_qrels_all,
        citation='dblp:conf/trec/Collins-Thompson14',
        desc='TREC Web Track 2014 diversity ranking benchmark.')


# Registration
# -----------------------------------------
irds.register(
    web2002, web2002_named_page, web2003, web2003_named_page, web2004,
    web2009, web2009_diversity, web2009_catb, web2009_catb_diversity,
    web2010, web2010_diversity, web2010_catb, web2010_catb_diversity,
    web2011, web2011_diversity, web2011_catb, web2011_catb_diversity,
    web2012, web2012_diversity, web2012_catb, web2012_catb_diversity,
    web2013, web2013_diversity, web2014, web2014_diversity,
)
