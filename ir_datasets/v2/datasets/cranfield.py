"""Cranfield -- a v2 dataset family.

A small (1,400 abstract) test collection, single flat benchmark, no subsets.
Same shape as ``vaswani.py``: v1's ``CranfieldDocs``/``CranfieldQueries``/
``CranfieldQrels`` handlers wrapped as v2 ``parser=``s rather than
reimplemented, since their format is specific to this one corpus.

``CranfieldDocs``/``CranfieldQueries`` read through ``io.TextIOWrapper``, so
their tar members need ``.cache(path)`` (same reasoning as vaswani.py and
msmarco_passage.py); ``CranfieldQrels`` uses ``codecs.getreader`` instead and
wouldn't strictly need it, but gets one anyway to match v1's on-disk layout
byte-for-byte (an existing v1 copy is then reused in place, same file).
``docs_store()`` hardcodes v1's own path, so an existing v1 docstore is
likewise reused automatically.
"""
import ir_datasets
from ir_datasets.datasets.cranfield import (
    CranfieldDocs as _V1CranfieldDocs, CranfieldQrels as _V1CranfieldQrels,
    CranfieldQueries as _V1CranfieldQueries,
)

from ir_datasets.v2 import Benchmark, DocTable, Parser, QrelTable, QueryTable, Resource, Source, irds

QREL_DEFS = {
    -1: 'References of no interest.',
    1: 'References of minimum interest, for example, those that have been included from an historical viewpoint.',
    2: 'References which were useful, either as general background to the work or as suggesting methods of tackling certain aspects of the work.',
    3: 'References of a high degree of relevance, the lack of which either would have made the research impracticable or would have resulted in a considerable amount of extra work.',
    4: 'References which are a complete answer to the question.',
}

BASE = ir_datasets.util.home_path() / 'cranfield'


class _CranfieldDocsParser(Parser):
    name = 'CranfieldDocs'

    def build(self, source, node):
        return _V1CranfieldDocs(source)


class _CranfieldQueriesParser(Parser):
    name = 'CranfieldQueries'

    def build(self, source, node):
        return _V1CranfieldQueries(source)


class _CranfieldQrelsParser(Parser):
    name = 'CranfieldQrels'

    def build(self, source, node):
        return _V1CranfieldQrels(source)


# Files
# -----------------------------------------
main_file = Resource('cranfield.tar.gz',
    sources=['http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz', Source.mirror()],
    hash='md5:1730f7be572d95a5a4b56c59a7b900a5',
    size=506_960,
)

# Tables
# -----------------------------------------
docs = DocTable('cranfield-docs',
    source=main_file.member('cran.all.1400').cache(BASE / 'docs.txt'),
    parser=_CranfieldDocsParser(),
    lang='en',
    count_hint=1_400,
)
queries = QueryTable('cranfield-queries',
    source=main_file.member('cran.qry').cache(BASE / 'queries.txt'),
    parser=_CranfieldQueriesParser(),
    lang='en',
)
qrels = QrelTable('cranfield-qrels',
    source=main_file.member('cranqrel').cache(BASE / 'qrels.txt'),
    parser=_CranfieldQrelsParser(),
    defs=QREL_DEFS,
)

# Benchmark
# -----------------------------------------
cranfield = Benchmark('cranfield',
    docs=docs, queries=queries, qrels=qrels,
    desc='A small corpus of 1,400 scientific abstracts.')


# Registration
# -----------------------------------------
irds.register(cranfield)
