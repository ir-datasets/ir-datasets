"""TREC Genomics Track -- a v2 dataset family, covering all four years the
track has run over: 2004/2005 (MEDLINE abstracts) and 2006/2007 (Highwire
full-text) -- all in one file since it's the same track throughout, just
switching corpora each generation, same "track spans multiple corpora"
reasoning as trec_web.py/trec_tb.py/trec_mq.py/trec_misinfo.py.

2004/2005 judge ``medline-2004-docs`` (imported by reference from
``medline.py``); 2006/2007 judge ``highwire`` (imported by reference
from ``highwire.py``) -- same cross-file pattern as ``trec_adhoc.py``
importing ``docs`` from ``disks45.py``.

Queries/qrels for all four years were previously defined inline in
``medline.py``/``highwire.py`` themselves, with the Benchmark id prefixed by
the corpus module name (``medline-trec-genomics-2004``,
``highwire-trec-genomics-2007``, ...) -- moved here and renamed to drop that
prefix, matching the unprefixed ``trec-web-2002``-style convention every
other multi-corpus track file uses (node names shouldn't encode lineage;
``derived_from`` edges already do -- see the "v2 naming" note in
todo-v2.txt).
"""
from ir_datasets.datasets.highwire import HighwireQrels as _V1HighwireQrels
from ir_datasets.datasets.highwire import TrecGenomicsQueries as _V1TrecGenomicsQueries
from ir_datasets.datasets.medline import TREC04_XML_MAP, TrecGenomicsQuery
from ir_datasets.v2 import Benchmark, QrelTable, QueryTable, Resource, TrecQrels, irds
from ir_datasets.v2.datasets.highwire import docs as highwire_docs
from ir_datasets.v2.datasets.medline import docs_2004 as medline_2004_docs
from ir_datasets.v2.formats import Parser

QREL_DEFS_04_05 = {
    0: 'not relevant',
    1: 'possibly relevant',
    2: 'definitely relevant',
}
QREL_DEFS_2006 = {
    0: 'NOT',
    1: 'POSSIBLY',
    2: 'DEFINITELY',
}
QREL_DEFS_2007 = {
    0: 'NOT_RELEVANT',
    1: 'RELEVANT',
}


class _TrecXmlQueriesParser(Parser):
    name = 'TrecXmlQueries'

    def __init__(self, qtype, qtype_map=None):
        self.qtype = qtype
        self.qtype_map = qtype_map

    def build(self, source, node):
        from ir_datasets.formats import TrecXmlQueries
        return TrecXmlQueries(source, qtype=self.qtype, qtype_map=self.qtype_map, lang=node.lang)


class _TrecGenomicsQueriesParser(Parser):
    name = 'TrecGenomicsQueries'

    def build(self, source, node):
        return _V1TrecGenomicsQueries(source)


class _HighwireQrelsParser(Parser):
    name = 'HighwireQrels'

    def __init__(self, defs):
        self.defs = defs

    def build(self, source, node):
        return _V1HighwireQrels(source, self.defs)


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    genomics_2004_queries_file = Resource('trec-genomics-2004-queries.zip',
        sources=['https://dmice.ohsu.edu/trec-gen/data/2004/rest.zip'],
        md5='3f252c59774fe8e74337637d73f8afc6',
        size=227_203,
    )
    genomics_2004_qrels_file = Resource('trec-genomics-2004-qrels.txt',
        sources=['https://dmice.ohsu.edu/trec-gen/data/2004/04.qrels.txt'],
        md5='1cb017045d7909102476bcb17fb19878',
        size=128_056,
    )
    genomics_2005_queries_file = Resource('trec-genomics-2005-queries.txt',
        sources=['https://dmice.ohsu.edu/trec-gen/data/2005/adhoc2005narrative.txt'],
        md5='71e8044cb65458731f4496fdc2aad94a',
        size=5_551,
    )
    genomics_2005_qrels_file = Resource('trec-genomics-2005-qrels.txt',
        sources=['https://dmice.ohsu.edu/trec-gen/data/2005/genomics.qrels.large.txt'],
        md5='fd6ac71dcd337c0c0cddf0ffc0528cc6',
        size=661_626,
    )
    genomics_2006_queries_file = Resource('trec-genomics-2006-queries.txt',
        sources=['https://dmice.ohsu.edu/trec-gen/data/2006/topics/2006topics.txt'],
        md5='fd458f5398350e59831745e51854b2b0',
        size=2_056,
    )
    genomics_2006_qrels_file = Resource('trec-genomics-2006-qrels.txt',
        sources=['https://dmice.ohsu.edu/trec-gen/data/2006/trec2006.raw.relevance.tsv.txt'],
        md5='a133e38bcd03c8b6509f46506cae753b',
        size=1_323_494,
    )
    genomics_2007_queries_file = Resource('trec-genomics-2007-queries.txt',
        sources=['https://dmice.ohsu.edu/trec-gen/data/2007/2007topics.txt'],
        md5='be5fc2d4e984003da6aa9dfab9eb67a3',
        size=2_576,
    )
    genomics_2007_qrels_file = Resource('trec-genomics-2007-qrels.txt',
        sources=['https://dmice.ohsu.edu/trec-gen/data/2007/trecgen2007.all.judgments.tsv.txt'],
        md5='5be6b6eea10d8ec0dac25bbe21af38a0',
        size=1_288_196,
    )

    # Tables
    # -----------------------------------------
    genomics_2004_queries = QueryTable('trec-genomics-2004-queries',
        source=genomics_2004_queries_file.zip_member('Official.xml'),
        parser=_TrecXmlQueriesParser(TrecGenomicsQuery, qtype_map=TREC04_XML_MAP),
        count_hint=50,
        citation='dblp:conf/trec/HershBRCKJ04',
    )
    genomics_2004_qrels = TrecQrels('trec-genomics-2004-qrels',
        source=genomics_2004_qrels_file, defs=QREL_DEFS_04_05, count_hint=8_268)

    genomics_2005_queries = QueryTable('trec-genomics-2005-queries',
        source=genomics_2005_queries_file,
        parser=_TrecGenomicsQueriesParser(),
        count_hint=50,
        citation='dblp:conf/trec/HershCYBRH05',
    )
    genomics_2005_qrels = TrecQrels('trec-genomics-2005-qrels',
        source=genomics_2005_qrels_file, defs=QREL_DEFS_04_05, count_hint=39_958)

    genomics_2006_queries = QueryTable('trec-genomics-2006-queries',
        source=genomics_2006_queries_file,
        parser=_TrecGenomicsQueriesParser(),
        count_hint=28,
        citation='dblp:conf/trec/HershCRR06',
    )
    genomics_2006_qrels = QrelTable('trec-genomics-2006-qrels',
        source=genomics_2006_qrels_file,
        parser=_HighwireQrelsParser(QREL_DEFS_2006),
        defs=QREL_DEFS_2006,
        count_hint=27_999,
    )
    genomics_2007_queries = QueryTable('trec-genomics-2007-queries',
        source=genomics_2007_queries_file,
        parser=_TrecGenomicsQueriesParser(),
        count_hint=36,
        citation='dblp:conf/trec/HershCRR07',
    )
    genomics_2007_qrels = QrelTable('trec-genomics-2007-qrels',
        source=genomics_2007_qrels_file,
        parser=_HighwireQrelsParser(QREL_DEFS_2007),
        defs=QREL_DEFS_2007,
        count_hint=35_996,
    )

    # Benchmarks
    # -----------------------------------------
    genomics_2004 = Benchmark('trec-genomics-2004',
        docs=medline_2004_docs, queries=genomics_2004_queries, qrels=genomics_2004_qrels,
        desc='TREC Genomics Track 2004.',
        citation='dblp:conf/trec/HershBRCKJ04')
    genomics_2005 = Benchmark('trec-genomics-2005',
        docs=medline_2004_docs, queries=genomics_2005_queries, qrels=genomics_2005_qrels,
        desc='TREC Genomics Track 2005.',
        citation='dblp:conf/trec/HershCYBRH05')
    genomics_2006 = Benchmark('trec-genomics-2006',
        docs=highwire_docs, queries=genomics_2006_queries, qrels=genomics_2006_qrels,
        desc='TREC Genomics Track 2006.',
        citation='dblp:conf/trec/HershCRR06')
    genomics_2007 = Benchmark('trec-genomics-2007',
        docs=highwire_docs, queries=genomics_2007_queries, qrels=genomics_2007_qrels,
        desc='TREC Genomics Track 2007.',
        citation='dblp:conf/trec/HershCRR07')


# Registration
# -----------------------------------------
irds.register(genomics_2004, genomics_2005, genomics_2006, genomics_2007)
