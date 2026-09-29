"""TREC Spanish -- a v2 dataset family.

Mandarin's sibling in the multilingual tracks (see ``trec_mandarin.py``): an
LDC-gated (LDC2000T51) Spanish-language news corpus, evaluated at TREC 3 and
TREC 4. ``Source.external()`` points at ``<home>/external/trec-spanish.tgz`` (v1's old
``<home>/trec-spanish/corpus.tgz`` still read as fallback).

TREC Spanish's queries have one extra wrinkle no other family here has: lines
starting with ``**`` are English translations interleaved with the Spanish
original, and v1 handles this with a small adapter,
``TrecSpanishTranslateQueries``, wrapped *around* a built ``TrecQueries``
handler (not a stream transform -- it works on already-parsed records). Reused
here via a local ``Parser`` that builds the inner v1 ``TrecQueries`` and wraps
it, the same "reuse, don't reimplement" principle as ``vaswani.py``'s and
``cranfield.py``'s one-off handler wrappers.
"""
from ir_datasets.formats import TrecQuery
from ir_datasets.datasets.trec_spanish import (
    TrecDescOnlyQuery, TrecSpanish3Query, TrecSpanish4Query,
    TrecSpanishTranslateQueries as _V1TrecSpanishTranslateQueries,
)

from ir_datasets.v2 import Benchmark, Parser, QrelTable, QueryTable, Resource, Source, TrecDocs, TrecQrels, irds
from ir_datasets.v2.formats import _v1

DATA_ACCESS = (
    "The TREC Spanish corpus is based on the LDC's Spanish news collection "
    "<https://catalog.ldc.upenn.edu/LDC2000T51> (LDC2000T51.tgz). Many "
    "organizations already have an LDC subscription; check with your library "
    "for access. Once obtained, place or symlink it at: {path}"
)

CITATION_TREC3 = 'dblp:conf/trec/Harman94'
CITATION_TREC4 = 'dblp:conf/trec/Harman95'

QREL_DEFS = {
    1: 'relevant',
    0: 'not relevant',
}

QTYPE_MAP_3 = {
    '<num> *(Number:)? *SP': 'query_id',  # Remove SP prefix from QIDs
    '<title> *(Topic:)?': 'title',
    '<desc> *(Description:)?': 'description',
    '<narr> *(Narrative:)?': 'narrative',
}

QTYPE_MAP_4 = {
    '<num> *(Number:)? *SP': 'query_id',  # Remove SP prefix from QIDs
    '<desc> *(Description:)?': 'description',
}

DOCS_MD5 = '4b8583c03149cf0c06c090fd230b33c6'
DOCS_LOCAL_PATH = 'trec-spanish.tgz'  # relative to <home>/external
DOCS_OLD_LOCATIONS = ['trec-spanish/corpus.tgz']  # relative to <home> (v1)


class _TrecSpanishQueriesParser(Parser):
    """Builds a plain TrecQueries handler, then splits its Spanish/English
    (``**``-prefixed translation) fields apart via v1's own adapter."""
    name = 'TrecSpanishQueries'

    def __init__(self, inner_qtype, qtype_map, target_qtype):
        self.inner_qtype = inner_qtype
        self.qtype_map = qtype_map
        self.target_qtype = target_qtype

    def build(self, source, node):
        inner = _v1.TrecQueries(source, qtype=self.inner_qtype, qtype_map=self.qtype_map,
                                encoding='ISO-8859-1', lang=node.lang)
        return _V1TrecSpanishTranslateQueries(inner, self.target_qtype)


# Files
# -----------------------------------------
docs_file = Resource('trec-spanish-docs.tgz',
    sources=[Source.external(DOCS_LOCAL_PATH, old_locations=DOCS_OLD_LOCATIONS, instructions=DATA_ACCESS)],
    md5=DOCS_MD5,
)
trec3_queries_file = Resource('trec-spanish-3-queries.gz',
    sources=['https://trec.nist.gov/data/topics_noneng/topics.SP1-SP25.spanish.english.gz', Source.mirror()],
    md5='22eea4a5c131db9cc4a431235f6a0573',
    size=9_029,
)
trec3_qrels_file = Resource('trec-spanish-3-qrels.gz',
    sources=['https://trec.nist.gov/data/qrels_noneng/qrels.1-25.spanish.gz', Source.mirror()],
    md5='e1703487f43fb7ea30b87a0f14ccb5ce',
    size=64_178,
)
trec4_queries_file = Resource('trec-spanish-4-queries.gz',
    sources=['https://trec.nist.gov/data/topics_noneng/topics.SP26-SP50.spanish.english.gz', Source.mirror()],
    md5='dfd9685cce559e33ab397c1878a6a1f8',
    size=2_091,
)
trec4_qrels_file = Resource('trec-spanish-4-qrels.gz',
    sources=['https://trec.nist.gov/data/qrels_noneng/qrels.26-50.spanish.gz', Source.mirror()],
    md5='f2540f9fb83433ca8ef9503671136498',
    size=46_394,
)

# Tables
# -----------------------------------------
docs = TrecDocs('trec-spanish-docs',
    source=docs_file,
    encoding='ISO-8859-1',
    path_globs=['**/afp_text/af*', '**/infosel_data/ism_*'],
    lang='es',
    count_hint=120_605,
    citation='Rogers2000Spanish',
)

# Query text mixes Spanish and English fields, so no single lang applies --
# same reasoning as trec_mandarin.py.
trec3_queries = QueryTable('trec-spanish-3-queries',
    source=trec3_queries_file.gunzip(),
    parser=_TrecSpanishQueriesParser(TrecQuery, QTYPE_MAP_3, TrecSpanish3Query),
    lang=None,
    citation=CITATION_TREC3,
)
trec3_qrels = TrecQrels('trec-spanish-3-qrels',
    source=trec3_qrels_file.gunzip(), defs=QREL_DEFS,
    citation=CITATION_TREC3)

trec4_queries = QueryTable('trec-spanish-4-queries',
    source=trec4_queries_file.gunzip(),
    parser=_TrecSpanishQueriesParser(TrecDescOnlyQuery, QTYPE_MAP_4, TrecSpanish4Query),
    lang=None,
    citation=CITATION_TREC4,
)
trec4_qrels = TrecQrels('trec-spanish-4-qrels',
    source=trec4_qrels_file.gunzip(), defs=QREL_DEFS,
    citation=CITATION_TREC4)

# Benchmarks
# -----------------------------------------
# Named trec-spanish-N, not N-spanish: unlike trec-dl-YYYY-passage (whose
# queries are genuinely shared with msmarco-document), nothing else reuses
# these tables, so there's no cross-family identity to hoist -- family name
# stays first, same as any other subset. v1's redundant "trec" in "trec3"/
# "trec4" is dropped since it's already in the family name.
trec3 = Benchmark('trec-spanish-3',
    docs=docs, queries=trec3_queries, qrels=trec3_qrels,
    citation=CITATION_TREC3,
    desc='Spanish-language benchmark from TREC 3.')
trec4 = Benchmark('trec-spanish-4',
    docs=docs, queries=trec4_queries, qrels=trec4_qrels,
    citation=CITATION_TREC4,
    desc='Spanish-language benchmark from TREC 4.')


# Registration
# -----------------------------------------
irds.register(trec3, trec4)
