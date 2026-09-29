"""NTCIR-WWW -- a v2 dataset family (three NTCIR "web search" shared-task
benchmarks, all judging ``clueweb12-b13-docs``, imported by reference from
``clueweb12.py`` -- same cross-file pattern as ``disks45.py``/``trec_adhoc.py``).

NTCIR-WWW is not a TREC track, so it does not belong in ``trec_web.py`` (the
TREC Web Track file) -- it gets its own file here, same as
``trec_misinfo.py``/``clef_ehealth.py`` split out of legacy ``clueweb12.py``.

``NtcirQrels`` (a v1 ``formats`` class, a 3-column ``qid did score`` parser
with no general-purpose v2 format node) is reused unmodified via a local
``Parser``, same "thin layer over v1 machinery" pattern as everywhere else.
``NtcirQuery``/``ntcir_map``/``NTCIR_QREL_DEFS`` are still defined in legacy
``ir_datasets.datasets.clueweb12`` -- type/constant imports, not behavioral
ones, same as ``clinicaltrials.py`` importing record types from
``medline.py``.

* ``ntcir-www-1``: queries are a zip member (``eng.queries.xml``), parsed as
  a plain ``GenericQuery`` with a custom ``qtype_map`` (v1 never defined a
  dedicated record type for this one). No ``.cache()`` needed around the zip
  member -- ``TrecXmlQueries`` only ever calls ``.stream()``, same reasoning
  as ``argsme.py``'s zip members.
* ``ntcir-www-2``: queries are a zip member (``qEng.xml``), ``NtcirQuery``/
  ``ntcir_map``.
* ``ntcir-www-3``: queries only, plain XML (no zip), ``NtcirQuery``/
  ``ntcir_map`` -- v1's own ``_init()`` registers no qrels for this one.
"""
from ir_datasets.datasets.clueweb12 import NTCIR_QREL_DEFS, NtcirQuery, ntcir_map
from ir_datasets.formats import GenericQuery
from ir_datasets.formats import NtcirQrels as _V1NtcirQrels
from ir_datasets.v2 import Benchmark, QrelTable, QueryTable, Resource, irds
from ir_datasets.v2.datasets.clueweb12 import docs_b13
from ir_datasets.v2.formats import Parser


class _TrecXmlQueriesParser(Parser):
    name = 'TrecXmlQueries'

    def __init__(self, qtype, qtype_map=None):
        self.qtype = qtype
        self.qtype_map = qtype_map

    def build(self, source, node):
        from ir_datasets.formats import TrecXmlQueries
        return TrecXmlQueries(source, qtype=self.qtype, qtype_map=self.qtype_map, lang=node.lang)


class _NtcirQrelsParser(Parser):
    name = 'NtcirQrels'

    def build(self, source, node):
        return _V1NtcirQrels(source, node.defs or {})


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    ntcir_www_1_queries_file = Resource('ntcir-www-1-queries.zip',
        sources=['http://www.thuir.cn/ntcirwww/files/eng.queries.xml.zip'],
        hash='md5:ed43ba82791bb20776c049421525a055',
        size=1_611,
    ).zip_member('eng.queries.xml')
    ntcir_www_1_qrels_file = Resource('ntcir-www-1-qrels.txt',
        sources=['https://macavaney.us/misc/ntcir-www-1.qrels'],
        hash='md5:634464456437bf378725958822910242',
        size=865_810,
    )
    ntcir_www_2_queries_file = Resource('ntcir-www-2-queries.zip',
        sources=['http://www.thuir.cn/ntcirwww2/qEng.zip'],
        hash='md5:d4b108b52b2e2c8bedc2e12540414735',
        size=3_466,
    ).zip_member('qEng.xml')
    ntcir_www_2_qrels_file = Resource('ntcir-www-2-qrels.txt',
        sources=['http://www.thuir.cn/ntcirwww3/www2e.qrels'],
        hash='md5:155b515dd9fc05e1aeb9c116c9147bb0',
        size=939_318,
    )
    ntcir_www_3_queries_file = Resource('ntcir-www-3-queries.xml',
        sources=['http://www.thuir.cn/ntcirwww3/www2www3topics-E.xml'],
        hash='md5:1ecd1380b20894014a54eb6cb8064587',
        size=28_823,
    )

    # Tables
    # -----------------------------------------
    ntcir_www_1_queries = QueryTable('ntcir-www-1-queries',
        source=ntcir_www_1_queries_file,
        parser=_TrecXmlQueriesParser(GenericQuery, qtype_map={'qid': 'query_id', 'content': 'text'}),
        count_hint=100,
    citation='dblp:conf/ntcir/0001S0DXX17')
    ntcir_www_1_qrels = QrelTable('ntcir-www-1-qrels',
        source=ntcir_www_1_qrels_file,
        parser=_NtcirQrelsParser(),
        defs=NTCIR_QREL_DEFS,
        count_hint=25_465,
    citation='dblp:conf/ntcir/0001S0DXX17')
    ntcir_www_2_queries = QueryTable('ntcir-www-2-queries',
        source=ntcir_www_2_queries_file,
        parser=_TrecXmlQueriesParser(NtcirQuery, qtype_map=ntcir_map),
        count_hint=80,
    citation='dblp:conf/ntcir/MaoS0X0D19')
    ntcir_www_2_qrels = QrelTable('ntcir-www-2-qrels',
        source=ntcir_www_2_qrels_file,
        parser=_NtcirQrelsParser(),
        defs=NTCIR_QREL_DEFS,
        count_hint=27_627,
    citation='dblp:conf/ntcir/MaoS0X0D19')
    ntcir_www_3_queries = QueryTable('ntcir-www-3-queries',
        source=ntcir_www_3_queries_file,
        parser=_TrecXmlQueriesParser(NtcirQuery, qtype_map=ntcir_map),
        count_hint=160,
    )

    # Benchmarks
    # -----------------------------------------
    ntcir_www_1 = Benchmark('ntcir-www-1',
        docs=docs_b13, queries=ntcir_www_1_queries, qrels=ntcir_www_1_qrels,
        citation='dblp:conf/ntcir/0001S0DXX17',
        desc='NTCIR-13 WWW-1.')
    ntcir_www_2 = Benchmark('ntcir-www-2',
        docs=docs_b13, queries=ntcir_www_2_queries, qrels=ntcir_www_2_qrels,
        citation='dblp:conf/ntcir/MaoS0X0D19',
        desc='NTCIR-14 WWW-2.')
    ntcir_www_3 = Benchmark('ntcir-www-3',
        docs=docs_b13, queries=ntcir_www_3_queries,
        desc='NTCIR-15 WWW-3 (queries only; v1 does not wire up qrels for this dataset).')


# Registration
# -----------------------------------------
irds.register(ntcir_www_1, ntcir_www_2, ntcir_www_3)
