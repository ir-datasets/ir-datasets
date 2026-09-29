"""CODEC -- a v2 dataset family.

The corpus is JSONL, read through v1's ``JsonlDocs`` directly (same
"thin layer over v1 machinery" as every other v2 family -- see
``beir.py``/``car.py``), with the node's own ``docstore_path`` passed
through so its cache lives where v2 wants it rather than v1's default
``<source path>.pklz4``. The documents are one gzipped JSONL file hosted on the Hugging Face Hub
(``macavaney/codec``); the DocTable reads it through ``.gunzip()``.

Queries are a single JSON object (``{qid: {...}}``), not JSONL -- v1's own
small ``CodecQueries`` handler (not a general-purpose v2 format, so reused
as-is via a ``parser=`` wrapper rather than reimplemented) filters by a
domain prefix for the three per-domain subsets. Domain qrels are the shared
qrels table filtered down to that domain's own query ids (``ids_of``), same
shape as ``nfcorpus.py``'s nontopic/video variants -- not their own
separately-downloaded file.
"""
import ir_datasets
from ir_datasets.datasets.codec import CodecDoc, CodecQueries as _V1CodecQueries
from ir_datasets.v2 import Benchmark, DocTable, Filter, QueryTable, Resource, Source, TrecQrels, ids_of, irds
from ir_datasets.v2.formats import Parser

DOMAINS = ['economics', 'history', 'politics']

QREL_DEFS = {
    3: 'Very Valuable. Includes central topic-specific arguments, evidence, or knowledge. This does not include general definitions or background.',
    2: 'Somewhat Valuable. Includes valuable topic-specific arguments, evidence, or knowledge.',
    1: 'Not Valuable. Consists of definitions or background.',
    0: 'Not Relevant. Not useful or on topic.',
}


class _JsonlDocsParser(Parser):
    name = 'JsonlDocs'

    def build(self, source, node):
        from ir_datasets.formats import JsonlDocs
        return JsonlDocs(source, doc_cls=CodecDoc,
            mapping={'doc_id': 'id', 'title': 'title', 'text': 'contents', 'url': 'url'},
            lang=node.lang, count_hint=node.count_hint, docstore_path=str(node.docstore_path))


class _CodecQueriesParser(Parser):
    name = 'CodecQueries'

    def __init__(self, qid_filter=None):
        self.qid_filter = qid_filter

    def build(self, source, node):
        return _V1CodecQueries(source, qid_filter=self.qid_filter)


# Files
# -----------------------------------------
docs_file = Resource('codec-docs.jsonl.gz',
    sources=['https://huggingface.co/datasets/macavaney/codec/resolve/main/documents.jsonl.gz'],
    hashes=['sha256:c567900f432622637677060b5d24808bc0e4b6b38806b045d8815a0bedadab90'],
    size=1_150_819_496,
)
queries_file = Resource('codec-queries.json',
    sources=['https://raw.githubusercontent.com/grill-lab/CODEC/main/topics/topics.json', Source.mirror()],
    md5='f75e4733693588449f68f7fdceb02ec9',
    size=47_192,
)
qrels_file = Resource('codec-qrels.txt',
    sources=['https://raw.githubusercontent.com/grill-lab/CODEC/main/raw_judgments/raw_document_judgments.txt', Source.mirror()],
    md5='7200606d6dc573abe2dd93160d5a5ab5',
    size=306_976,
)

# Tables
# -----------------------------------------
docs = DocTable('codec-docs',
    source=docs_file.gunzip(),
    parser=_JsonlDocsParser(),
    lang='en',
    count_hint=729_824,
)
queries = QueryTable('codec-queries',
    source=queries_file,
    parser=_CodecQueriesParser(),
    lang='en',
    count_hint=42,
)
qrels = TrecQrels('codec-qrels', source=qrels_file, defs=QREL_DEFS, count_hint=6_186)

# Benchmarks
# -----------------------------------------
codec = Benchmark('codec',
    docs=docs, queries=queries, qrels=qrels,
    citation='dblp:conf/sigir/MackieOGFM022',
    desc='CODEC: a document-level test collection for complex, essay-style '
         'information needs across economics, history, and politics.')

domain_benchmarks = {}
_DOMAIN_COUNTS = {
    'economics': (14, 1_970),
    'history': (14, 2_024),
    'politics': (14, 2_192),
}
for _domain, (_qcount, _rcount) in _DOMAIN_COUNTS.items():
    _domain_queries = QueryTable(f'codec-{_domain}-queries',
        source=queries_file,
        parser=_CodecQueriesParser(qid_filter=_domain),
        lang='en',
        count_hint=_qcount,
    )
    domain_benchmarks[_domain] = Benchmark(f'codec-{_domain}',
        derived_from=codec,
        queries=_domain_queries,
        filter=Filter(query_ids=ids_of(f'irds:codec-{_domain}-queries'), mode='include'),
        desc=f'CODEC, {_domain} domain subset.')


# Registration
# -----------------------------------------
irds.register(codec, *domain_benchmarks.values())
