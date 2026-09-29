"""Vaswani (NPL) -- a v2 dataset family.

A small (~11k abstract) test collection, single flat benchmark, no subsets.
Reuses v1's ``VaswaniDocs``/``VaswaniQueries``/``VaswaniQrels`` handler
classes as v2 ``parser=`` wrappers -- their custom sentinel-delimited format
isn't a reusable ``formats.py`` type (it's specific to this one corpus), so
there is nothing to generalize; only ``Parser.build()`` needs writing.

All three handlers read their source through ``io.TextIOWrapper``, which (as
with msmarco-passage) is incompatible with a tar member in streaming mode --
so each is ``.member(...).cache(path)``, not bare ``.member(...)``. Each
handler's ``docs_store()`` hardcodes v1's own path
(``<home>/vaswani/docs.pklz4``), so an existing v1 docstore is reused in
place automatically, with no override needed (contrast ``formats.TsvDocs``,
which needed a ``_V1TsvDocs`` subclass for exactly this).
"""
import ir_datasets
from ir_datasets.datasets.vaswani import (
    VaswaniDocs as _V1VaswaniDocs, VaswaniQrels as _V1VaswaniQrels,
    VaswaniQueries as _V1VaswaniQueries,
)

from ir_datasets.v2 import Benchmark, DocTable, Parser, QrelTable, QueryTable, Resource, Source, irds

QREL_DEFS = {
    1: 'Relevant',
}

BASE = ir_datasets.util.home_path() / 'vaswani'


class _VaswaniDocsParser(Parser):
    name = 'VaswaniDocs'

    def build(self, source, node):
        return _V1VaswaniDocs(source)


class _VaswaniQueriesParser(Parser):
    name = 'VaswaniQueries'

    def build(self, source, node):
        return _V1VaswaniQueries(source)


class _VaswaniQrelsParser(Parser):
    name = 'VaswaniQrels'

    def build(self, source, node):
        return _V1VaswaniQrels(source)


# Files
# -----------------------------------------
main_file = Resource('vaswani.tar.gz',
    sources=['http://ir.dcs.gla.ac.uk/resources/test_collections/npl/npl.tar.gz', Source.mirror()],
    md5='23e5607081191b153738e81fbd834680',
    size=2_125_168,
)

# Tables
# -----------------------------------------
docs = DocTable('vaswani-docs',
    source=main_file.member('doc-text').cache(BASE / 'docs.txt'),
    parser=_VaswaniDocsParser(),
    lang='en',
    count_hint=11_429,
)
queries = QueryTable('vaswani-queries',
    source=main_file.member('query-text').cache(BASE / 'queries.txt'),
    parser=_VaswaniQueriesParser(),
    lang='en',
)
qrels = QrelTable('vaswani-qrels',
    source=main_file.member('rlv-ass').cache(BASE / 'qrels.txt'),
    parser=_VaswaniQrelsParser(),
    defs=QREL_DEFS,
)

# Benchmark
# -----------------------------------------
vaswani = Benchmark('vaswani',
    docs=docs, queries=queries, qrels=qrels,
    desc='A small corpus of roughly 11,000 scientific abstracts.')


# Registration
# -----------------------------------------
irds.register(vaswani)
