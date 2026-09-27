"""TREC Arabic -- a v2 dataset family.

An LDC-gated (LDC2001T55) Arabic newswire corpus, evaluated in the TREC
cross-lingual track in 2001 and 2002 (v1's ``ar2001``/``ar2002``; v2 drops
the redundant "ar" prefix since it's already in the family name --
``trec-arabic-2001``/``trec-arabic-2002``). Simpler
than ``trec_mandarin.py``/``trec_spanish.py``: a single query type, no
gzip'd query/qrels files, no custom record-splitting adapter -- just
``TrecDocs``/``TrecQueries``/``TrecQrels`` directly.
"""
import ir_datasets
from ir_datasets.v2 import Benchmark, Resource, Source, TrecDocs, TrecQrels, TrecQueries, irds

DATA_ACCESS = (
    "The TREC Arabic corpus is based on the LDC's Arabic Newswire collection "
    "<https://catalog.ldc.upenn.edu/LDC2001T55> (LDC2001T55.tgz). Many "
    "organizations already have an LDC subscription; check with your library "
    "for access. Once obtained, symlink or copy it here: {path}"
)

QREL_DEFS = {
    1: 'relevant',
    0: 'not relevant',
}

QTYPE_MAP = {
    '<num> *(Number:)? *AR': 'query_id',  # Remove AR prefix from QIDs
    '<title> *(Topic:)?': 'title',
    '<desc> *(Description:)?': 'description',
    '<narr> *(Narrative:)?': 'narrative',
}

DOCS_MD5 = 'b17f34a51dca8d19fae66c338c9ed73a'
DOCS_LOCAL_PATH = ir_datasets.util.home_path() / 'trec-arabic' / 'corpus.tgz'

# Files
# -----------------------------------------
docs_file = Resource('trec-arabic-docs.tgz',
    sources=[Source.local(DOCS_LOCAL_PATH, instructions=DATA_ACCESS)],
    md5=DOCS_MD5,
)
ar2001_queries_file = Resource('trec-arabic-2001-queries.txt',
    sources=['https://trec.nist.gov/data/topics_noneng/arabic_topics.txt', Source.irds()],
    md5='a3d78c379056a080fe40a59a341496b8',
    size=10_320,
)
ar2001_qrels_file = Resource('trec-arabic-2001-qrels.txt',
    sources=['https://trec.nist.gov/data/qrels_noneng/xlingual_t10qrels.txt', Source.irds()],
    md5='5951e2f0bf72df9f93fc32b93e3a7fde',
    size=650_331,
)
ar2002_queries_file = Resource('trec-arabic-2002-queries.txt',
    sources=['https://trec.nist.gov/data/topics_noneng/CL.topics.arabic.trec11.txt', Source.irds()],
    md5='f75a6164d794bab66509f1e818612363',
    size=15_873,
)
ar2002_qrels_file = Resource('trec-arabic-2002-qrels.txt',
    sources=['https://trec.nist.gov/data/qrels_noneng/qrels.trec11.xlingual.txt', Source.irds()],
    md5='40f25e1e98101e27d081685cbdc390ef',
    size=1_114_528,
)

# Tables
# -----------------------------------------
docs = TrecDocs('trec-arabic-docs',
    source=docs_file,
    encoding='utf8',
    path_globs=['arabic_newswire_a/transcripts/*/*.sgm.gz'],
    lang='ar',
    count_hint=383_872,
)

ar2001_queries = TrecQueries('trec-arabic-2001-queries',
    source=ar2001_queries_file,
    qtype_map=QTYPE_MAP, encoding='ISO-8859-6',
    lang='ar',
)
ar2001_qrels = TrecQrels('trec-arabic-2001-qrels',
    source=ar2001_qrels_file, defs=QREL_DEFS)

ar2002_queries = TrecQueries('trec-arabic-2002-queries',
    source=ar2002_queries_file,
    qtype_map=QTYPE_MAP, encoding='ISO-8859-6',
    lang='ar',
)
ar2002_qrels = TrecQrels('trec-arabic-2002-qrels',
    source=ar2002_qrels_file, defs=QREL_DEFS)

# Benchmarks
# -----------------------------------------
# Named trec-arabic-YYYY, not YYYY-arabic: unlike trec-dl-YYYY-passage (whose
# queries are genuinely shared with msmarco-document), nothing else reuses
# these tables, so there's no cross-family identity to hoist -- family name
# stays first, same as any other subset. v1's "ar" prefix (ar2001/ar2002) is
# dropped as redundant with "arabic" already in the family name.
ar2001 = Benchmark('trec-arabic-2001',
    docs=docs, queries=ar2001_queries, qrels=ar2001_qrels,
    desc='TREC cross-lingual Arabic benchmark, 2001 (TREC-10).')
ar2002 = Benchmark('trec-arabic-2002',
    docs=docs, queries=ar2002_queries, qrels=ar2002_qrels,
    desc='TREC cross-lingual Arabic benchmark, 2002 (TREC-11).')


# Registration
# -----------------------------------------
irds.register(ar2001, ar2002)


# Aliases (old ir-datasets ID mapping)
# -----------------------------------------
irds.alias({
    'trec-arabic': 'trec-arabic-docs',
    'trec-arabic/ar2001': 'trec-arabic-2001',
    'trec-arabic/ar2002': 'trec-arabic-2002',
})
