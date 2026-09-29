"""TREC Clinical Decision Support -- a v2 dataset family, covering the three
years it ran: 2014/2015 (judged over ``pmc-v1-docs``) and 2016 (judged over
``pmc-v2-docs``), all in one file since it's the same track throughout, just
switching corpus versions each generation, same "track spans multiple
corpora" reasoning as trec_web.py/trec_genomics.py -- even though, for now,
both versions are PMC's own (see ``pmc.py``'s docstring; a track's home
shouldn't depend on how many corpora happen to be migrated for it yet).

Docs are imported by reference from ``pmc.py`` -- same cross-file pattern as
``trec_adhoc.py`` importing ``docs`` from ``disks45.py``.

Previously bundled in ``pmc.py`` as ``pmc-trec-cds-2014/2015/2016`` -- moved
here and renamed to drop the corpus prefix, matching the unprefixed
``trec-web-2002``-style convention every other multi-corpus track file uses.
"""
from ir_datasets.datasets.pmc import TrecCdsQuery, TrecCds2016Query
from ir_datasets.v2 import Benchmark, QueryTable, Resource, TrecQrels, irds
from ir_datasets.v2.datasets.pmc import v1_docs as pmc_v1_docs
from ir_datasets.v2.datasets.pmc import v2_docs as pmc_v2_docs
from ir_datasets.v2.formats import Parser

QREL_DEFS = {
    0: 'not relevant',
    1: 'possibly relevant',
    2: 'definitely relevant',
}

QUERY_FILE_MAP = {
    'number': 'query_id',
    'type': 'type',
    'description': 'description',
    'summary': 'summary',
    'note': 'note',
}


class _TrecXmlQueriesParser(Parser):
    name = 'TrecXmlQueries'

    def __init__(self, qtype, qtype_map=None):
        self.qtype = qtype
        self.qtype_map = qtype_map

    def build(self, source, node):
        from ir_datasets.formats import TrecXmlQueries
        return TrecXmlQueries(source, qtype=self.qtype, qtype_map=self.qtype_map,
            namespace=node.name, lang=node.lang)


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    cds_2014_queries_file = Resource('trec-cds-2014-queries.xml',
        sources=['http://www.trec-cds.org/topics2014.xml'],
        md5='4924e2a3bb539feac6cbb967f4875926',
        size=22_514)
    cds_2014_qrels_file = Resource('trec-cds-2014-qrels.txt',
        sources=['https://trec.nist.gov/data/clinical/qrels-treceval-2014.txt'],
        md5='07c8f85a7b7bcfd4211301ecaa3b4769',
        size=556_628)
    cds_2015_queries_file = Resource('trec-cds-2015-queries.xml',
        sources=['https://trec.nist.gov/data/clinical/topics-2015-A.xml'],
        md5='462d9804257e7ad0128f16324c9ec06d',
        size=22_491)
    cds_2015_qrels_file = Resource('trec-cds-2015-qrels.txt',
        sources=['https://trec.nist.gov/data/clinical/qrels-treceval-2015.txt'],
        md5='7bbe901cfa36df56dd13cce0275c1a2b',
        size=554_312)
    cds_2016_queries_file = Resource('trec-cds-2016-queries.xml',
        sources=['https://trec.nist.gov/data/clinical/topics2016.xml'],
        md5='22ccb3412931efe1ea084330737e41bc',
        size=79_966)
    cds_2016_qrels_file = Resource('trec-cds-2016-qrels.txt',
        sources=['https://trec.nist.gov/data/clinical/qrels-treceval-2016.txt'],
        md5='1a450d38137082e214c1201a3023a6d1',
        size=553_709)

    # Tables
    # -----------------------------------------
    cds_2014_queries = QueryTable('trec-cds-2014-queries',
        source=cds_2014_queries_file,
        parser=_TrecXmlQueriesParser(TrecCdsQuery, QUERY_FILE_MAP),
        count_hint=30,
        citation='dblp:conf/trec/SimpsonVH14',
    )
    cds_2014_qrels = TrecQrels('trec-cds-2014-qrels',
        source=cds_2014_qrels_file,
        defs=QREL_DEFS,
        count_hint=37_949,
    )
    cds_2015_queries = QueryTable('trec-cds-2015-queries',
        source=cds_2015_queries_file,
        parser=_TrecXmlQueriesParser(TrecCdsQuery, QUERY_FILE_MAP),
        count_hint=30,
        citation='dblp:conf/trec/RobertsSVH15',
    )
    cds_2015_qrels = TrecQrels('trec-cds-2015-qrels',
        source=cds_2015_qrels_file,
        defs=QREL_DEFS,
        count_hint=37_807,
    )
    cds_2016_queries = QueryTable('trec-cds-2016-queries',
        source=cds_2016_queries_file,
        parser=_TrecXmlQueriesParser(TrecCds2016Query, QUERY_FILE_MAP),
        count_hint=30,
        citation='dblp:conf/trec/RobertsDVH16',
    )
    cds_2016_qrels = TrecQrels('trec-cds-2016-qrels',
        source=cds_2016_qrels_file,
        defs=QREL_DEFS,
        count_hint=37_707,
    )

    # Benchmarks
    # -----------------------------------------
    cds_2014 = Benchmark('trec-cds-2014',
        docs=pmc_v1_docs, queries=cds_2014_queries, qrels=cds_2014_qrels,
        desc='TREC Clinical Decision Support track 2014.',
        citation='dblp:conf/trec/SimpsonVH14')
    cds_2015 = Benchmark('trec-cds-2015',
        docs=pmc_v1_docs, queries=cds_2015_queries, qrels=cds_2015_qrels,
        desc='TREC Clinical Decision Support track 2015.',
        citation='dblp:conf/trec/RobertsSVH15')
    cds_2016 = Benchmark('trec-cds-2016',
        docs=pmc_v2_docs, queries=cds_2016_queries, qrels=cds_2016_qrels,
        desc='TREC Clinical Decision Support track 2016.',
        citation='dblp:conf/trec/RobertsDVH16')


# Registration
# -----------------------------------------
irds.register(cds_2014, cds_2015, cds_2016)
