"""TREC Clinical Trials -- a v2 dataset family, covering the two years it
ran (2021/2022), both judged over ``clinicaltrials-2021-docs``, same "track
spans multiple corpora" reasoning as trec_web.py/trec_genomics.py -- even
though, for now, clinicaltrials-2021 is the only corpus this track judges
(see ``clinicaltrials.py``'s docstring; a track's home shouldn't depend on
how many corpora happen to be migrated for it yet).

Docs are imported by reference from ``clinicaltrials.py`` -- same
cross-file pattern as ``trec_adhoc.py`` importing ``docs`` from
``disks45.py``.

Previously bundled in ``clinicaltrials.py`` as
``clinicaltrials-trec-ct-2021/2022`` -- moved here and renamed to drop the
corpus prefix, matching the unprefixed ``trec-web-2002``-style convention
every other multi-corpus track file uses.
"""
from ir_datasets.formats import GenericQuery
from ir_datasets.v2 import Benchmark, QueryTable, Resource, TrecQrels, irds
from ir_datasets.v2.datasets.clinicaltrials import docs_2021 as clinicaltrials_2021_docs
from ir_datasets.v2.formats import Parser

QREL_DEFS_2021 = {
    0: 'Not Relevant',
    1: 'Excluded',
    2: 'Eligible',
}

CT_QMAP = {'topic': 'text'}


class _TrecXmlQueriesParser(Parser):
    name = 'TrecXmlQueries'

    def __init__(self, qtype, qtype_map=None):
        self.qtype = qtype
        self.qtype_map = qtype_map

    def build(self, source, node):
        from ir_datasets.formats import TrecXmlQueries
        return TrecXmlQueries(source, qtype=self.qtype, qtype_map=self.qtype_map, lang=node.lang)


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    ct_2021_queries_file = Resource('trec-ct-2021-queries.xml',
        sources=['http://www.trec-cds.org/topics2021.xml'],
        md5='6d842b40387d760274447c1f8d7396a8',
        size=64_618,
    )
    ct_2021_qrels_file = Resource('trec-ct-2021-qrels.txt',
        sources=['https://trec.nist.gov/data/trials/qrels2021.txt'],
        md5='0335d95c58d5f5fd9bc730bccb60ca90',
        size=676_496,
    )
    ct_2022_queries_file = Resource('trec-ct-2022-queries.xml',
        sources=['https://www.trec-cds.org/topics2022.xml'],
        md5='73bcb3985a17fd60d786f5d8f5a0bb2e',
        size=32_423,
    )

    # Tables
    # -----------------------------------------
    ct_2021_queries = QueryTable('trec-ct-2021-queries',
        source=ct_2021_queries_file,
        parser=_TrecXmlQueriesParser(GenericQuery, qtype_map=CT_QMAP),
        count_hint=75,
    )
    ct_2021_qrels = TrecQrels('trec-ct-2021-qrels',
        source=ct_2021_qrels_file, defs=QREL_DEFS_2021, count_hint=35_832)

    ct_2022_queries = QueryTable('trec-ct-2022-queries',
        source=ct_2022_queries_file,
        parser=_TrecXmlQueriesParser(GenericQuery, qtype_map=CT_QMAP),
        count_hint=50,
    )

    # Benchmarks
    # -----------------------------------------
    ct_2021 = Benchmark('trec-ct-2021',
        docs=clinicaltrials_2021_docs, queries=ct_2021_queries, qrels=ct_2021_qrels,
        desc='TREC Clinical Trials 2021.')
    ct_2022 = Benchmark('trec-ct-2022',
        docs=clinicaltrials_2021_docs, queries=ct_2022_queries,
        citation='dblp:conf/trec/RobertsDVBH22',
        desc='TREC Clinical Trials 2022 (queries only; qrels not yet published).')


# Registration
# -----------------------------------------
irds.register(ct_2021, ct_2022)
