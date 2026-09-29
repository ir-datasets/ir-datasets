"""TREC Mandarin — a v2 dataset family.

A collection of Mandarin (Simplified Chinese) news articles used for the
multilingual tracks at TREC 5 and TREC 6. The corpus itself is LDC-gated
(LDC2000T52) and not automatically downloadable -- ``Source.local()`` points
the user at the same well-known path v1 used (``<home>/trec-mandarin/corpus.tgz``),
so an existing v1 symlink/copy is picked up in place either way (its md5 is
already in ``legacy_cache.json``, generated from v1's ``downloads.json``).

Introduces two new v2 format nodes (``TrecDocs``/``TrecQueries``, in
``formats.py``), the first family here to need them: v1's ``TrecDocs`` already
globs matching members out of a tar in streaming mode internally (see
``formats.py``'s ``_TrecDocsParser``), so no new pipeline primitive was needed
for that -- unlike ``.member()``'s one-file case, path_globs stays inside the
parser, same as v1.
"""
from typing import NamedTuple

import ir_datasets
from ir_datasets.v2 import Benchmark, Resource, Source, TrecDocs, TrecQrels, TrecQueries, irds

DATA_ACCESS = (
    "The TREC Mandarin corpus is based on the LDC's Mandarin news collection "
    "<https://catalog.ldc.upenn.edu/LDC2000T52> (LDC2000T52.tgz). Many "
    "organizations already have an LDC subscription; check with your library "
    "for access. Once obtained, symlink or copy it here: {path}"
)

CITATION_TREC5 = 'dblp:conf/trec/Smeaton96'
CITATION_TREC6 = 'dblp:conf/trec/Wilkinson97'


class TrecMandarinQuery(NamedTuple):
    query_id: str
    title_en: str
    title_zh: str
    description_en: str
    description_zh: str
    narrative_en: str
    narrative_zh: str
    def default_text(self):
        return self.title_zh


QREL_DEFS = {
    1: 'relevant',
    0: 'not relevant',
}

QTYPE_MAP = {
    '<num> *(Number:)? *CH': 'query_id',  # Remove CH prefix from QIDs
    '<E-title> *(Topic:)?': 'title_en',
    '<C-title> *(Topic:)?': 'title_zh',
    '<E-desc> *(Description:)?': 'description_en',
    '<C-desc> *(Description:)?': 'description_zh',
    '<E-narr> *(Narrative:)?': 'narrative_en',
    '<C-narr> *(Narrative:)?': 'narrative_zh',
}

DOCS_MD5 = 'a847fa029a1356b8f396aa642c449e38'
# Same path v1 used (base_path/cache_path from downloads.json), so both an
# existing v1 copy (via legacy_path()) and a fresh manual placement (via this
# Source.local(), on cache-miss) land in one place.
DOCS_LOCAL_PATH = ir_datasets.util.home_path() / 'trec-mandarin' / 'corpus.tgz'

# Files
# -----------------------------------------
docs_file = Resource('trec-mandarin-docs.tgz',
    sources=[Source.local(DOCS_LOCAL_PATH, instructions=DATA_ACCESS)],
    md5=DOCS_MD5,
)
trec5_queries_file = Resource('trec-mandarin-5-queries.gz',
    sources=['https://trec.nist.gov/data/topics_noneng/topics.CH1-CH28.chinese.english.gz', Source.irds()],
    md5='9ce885d36e8642d4114f40e7008e5b8a',
    size=9_136,
)
trec5_qrels_file = Resource('trec-mandarin-5-qrels.gz',
    sources=['https://trec.nist.gov/data/qrels_noneng/qrels.1-28.chinese.gz', Source.irds()],
    md5='73693083d75ef323fca2a218604b41ac',
    size=76_063,
)
trec6_queries_file = Resource('trec-mandarin-6-queries.gz',
    sources=['https://trec.nist.gov/data/topics_noneng/topics.CH29-CH54.chinese.english.gz', Source.irds()],
    md5='c3a58ec59e55c162fdc3e3a9c5e9b8a7',
    size=8_920,
)
trec6_qrels_file = Resource('trec-mandarin-6-qrels.gz',
    sources=['https://trec.nist.gov/data/qrels_noneng/qrels.trec6.29-54.chinese.gz', Source.irds()],
    md5='675ab2f14fad9017d646d052c0b35c46',
    size=44_468,
)

# Tables
# -----------------------------------------
docs = TrecDocs('trec-mandarin-docs',
    source=docs_file,
    encoding='GB18030',
    path_globs=['**/xinhua/x*', '**/peoples-daily/pd*'],
    lang='zh',
    count_hint=164_789,
    citation='Rogers2000Mandarin',
)

# Query text mixes English and Chinese fields (title_en/title_zh/...), so no
# single lang applies -- same as v1.
trec5_queries = TrecQueries('trec-mandarin-5-queries',
    source=trec5_queries_file.gunzip(),
    qtype=TrecMandarinQuery, qtype_map=QTYPE_MAP, encoding='GBK',
    lang=None,
    citation=CITATION_TREC5,
)
trec5_qrels = TrecQrels('trec-mandarin-5-qrels',
    source=trec5_qrels_file.gunzip(), defs=QREL_DEFS,
    citation=CITATION_TREC5)

trec6_queries = TrecQueries('trec-mandarin-6-queries',
    source=trec6_queries_file.gunzip(),
    qtype=TrecMandarinQuery, qtype_map=QTYPE_MAP, encoding='GBK',
    lang=None,
    citation=CITATION_TREC6,
)
trec6_qrels = TrecQrels('trec-mandarin-6-qrels',
    source=trec6_qrels_file.gunzip(), defs=QREL_DEFS,
    citation=CITATION_TREC6)

# Benchmarks
# -----------------------------------------
# Named trec-mandarin-N, not N-mandarin: unlike trec-dl-YYYY-passage (whose
# queries are genuinely shared with msmarco-document), nothing else reuses
# these tables, so there's no cross-family identity to hoist -- family name
# stays first, same as any other subset. v1's redundant "trec" in "trec5"/
# "trec6" is dropped since it's already in the family name.
trec5 = Benchmark('trec-mandarin-5',
    docs=docs, queries=trec5_queries, qrels=trec5_qrels,
    citation=CITATION_TREC5,
    desc='Mandarin Chinese benchmark from TREC 5.')
trec6 = Benchmark('trec-mandarin-6',
    docs=docs, queries=trec6_queries, qrels=trec6_qrels,
    citation=CITATION_TREC6,
    desc='Mandarin Chinese benchmark from TREC 6.')


# Registration
# -----------------------------------------
irds.register(trec5, trec6)
