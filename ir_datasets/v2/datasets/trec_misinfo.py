"""TREC Misinfo 2019 -- a v2 dataset family (one benchmark, judging
``clueweb12-b13-docs``, imported by reference from ``clueweb12.py`` -- same
cross-file pattern as ``disks45.py``/``trec_adhoc.py``).

TREC Misinfo *is* a TREC track, so -- per the "split TREC tasks out of the
shared corpus file into their own module" convention -- it gets its own file
here rather than living in ``clueweb12.py`` alongside the non-TREC NTCIR-WWW/
CLEF eHealth benchmarks. Note this file covers only the 2019 edition (over
ClueWeb12): TREC C4's own, later ``trec-misinfo-2021`` (a related but
distinct TREC track/year, over the *C4* corpus) is intentionally not merged
in here -- it is already migrated independently, in ``c4.py``.

``MsinfoQrels`` (note the typo in the v1 class name -- kept as-is, it's the
real v1 identifier) is a custom ``TrecQrels`` subclass with a 6-column parse
(``qid, it, did, rel, eff, cred`` -> ``MisinfoQrel``), reused unmodified via a
local ``Parser``. ``MisinfoQuery``/``misinfo_map``/``MISINFO_QREL_DEFS`` are
still defined in legacy ``ir_datasets.datasets.clueweb12`` -- type/constant
imports, not behavioral ones.
"""
from ir_datasets.datasets.clueweb12 import MISINFO_QREL_DEFS, MisinfoQuery, misinfo_map
from ir_datasets.datasets.clueweb12 import MsinfoQrels as _V1MsinfoQrels
from ir_datasets.v2 import Benchmark, QrelTable, QueryTable, Resource, Source, irds
from ir_datasets.v2.datasets.clueweb12 import docs_b13
from ir_datasets.v2.formats import Parser

CITATION = 'Abualsaud2019TrecDecision'


class _TrecXmlQueriesParser(Parser):
    name = 'TrecXmlQueries'

    def __init__(self, qtype, qtype_map=None):
        self.qtype = qtype
        self.qtype_map = qtype_map

    def build(self, source, node):
        from ir_datasets.formats import TrecXmlQueries
        return TrecXmlQueries(source, qtype=self.qtype, qtype_map=self.qtype_map, lang=node.lang)


class _MsinfoQrelsParser(Parser):
    name = 'MsinfoQrels'

    def build(self, source, node):
        return _V1MsinfoQrels(source, node.defs or {})


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    trec_misinfo_2019_queries_file = Resource('trec-misinfo-2019-queries.xml',
        sources=['https://trec.nist.gov/data/misinfo/2019topics.xml', Source.mirror()],
        md5='e46bb8ff3058bbcc1bd73a0ecbda1621',
        size=30_028,
    )
    trec_misinfo_2019_qrels_file = Resource('trec-misinfo-2019-qrels.txt',
        sources=['https://trec.nist.gov/data/misinfo/2019qrels_raw.txt', Source.mirror()],
        md5='faf86b2ac5fcca52b189a3ad408fd019',
        size=878_952,
    )

    # Tables
    # -----------------------------------------
    trec_misinfo_2019_queries = QueryTable('trec-misinfo-2019-queries',
        source=trec_misinfo_2019_queries_file,
        parser=_TrecXmlQueriesParser(MisinfoQuery, qtype_map=misinfo_map),
        count_hint=51,
    citation=CITATION)
    trec_misinfo_2019_qrels = QrelTable('trec-misinfo-2019-qrels',
        source=trec_misinfo_2019_qrels_file,
        parser=_MsinfoQrelsParser(),
        defs=MISINFO_QREL_DEFS,
        count_hint=22_859,
    citation=CITATION)

    # Benchmarks
    # -----------------------------------------
    trec_misinfo_2019 = Benchmark('trec-misinfo-2019',
        docs=docs_b13, queries=trec_misinfo_2019_queries, qrels=trec_misinfo_2019_qrels,
        citation=CITATION,
        desc='TREC Misinfo 2019.')


# Registration
# -----------------------------------------
irds.register(trec_misinfo_2019)
