"""KILT -- a v2 dataset family.

KILT is a large Wikipedia-derived JSON corpus (~5.9M docs) used for various
"knowledge intensive language tasks". The only benchmark family here is
``kilt-codec`` and its three domain subsets: KILT's entity-ranking task that
judges *KILT's own* Wikipedia entities against **CODEC's own 42 queries**
(the same query set already migrated in ``codec.py``) -- reused here by
reference, not redefined (same cross-file pattern as ``trec_adhoc.py``
importing ``disks45.py``). The qrels, however, are KILT's own: a separately
downloaded judgments file (``raw_entity_judgments.txt``, judging Wikipedia
entities) with its own ``CODEC_QREL_DEFS`` -- textually close to but distinct
from ``codec.py``'s own ``QREL_DEFS`` (that one judges CODEC's document
collection), so the two are kept as genuinely separate constants rather than
deduplicated across files. The per-domain subsets reuse CODEC's own
per-domain query id sets (``ids_of('irds:codec-<domain>-queries')``) via
``Filter``, applied to KILT's qrels rather than CODEC's.

The docs corpus (``KiltDocs``, a v1 ``BaseDocs`` subclass with custom
newline-delimited-JSON parsing/markup-stripping logic) is reused completely
unmodified via a thin ``Parser`` wrapper -- including its own hardcoded
docstore path (``<home>/kilt/docs.pklz4``), same "reuse v1's own path
management unmodified" precedent as ``clinicaltrials.py``/``codec.py``.
"""
import ir_datasets
from ir_datasets.datasets.kilt import KiltDocs
from ir_datasets.v2 import Benchmark, DocTable, Filter, Resource, TrecQrels, ids_of, irds
from ir_datasets.v2.datasets.codec import queries as codec_queries, domain_benchmarks as codec_domain_benchmarks
from ir_datasets.v2.formats import Parser

DOMAINS = ['economics', 'history', 'politics']

#: KILT's own entity-ranking judgments (Wikipedia entities relevant to a
#: CODEC topic) -- distinct from codec.py's QREL_DEFS (which judges CODEC's
#: document collection), even though the wording is similar. Verified
#: textually different from codec.py's QREL_DEFS; kept as its own constant.
CODEC_QREL_DEFS = {
    3: 'Very Valuable. It is absolutely critical to understand what this entity is for understanding this topic.',
    2: 'Somewhat valuable. It is important to understand what this entity is for understanding this topic.',
    1: 'Not Valuable. It is useful to understand what this entity is for understanding this topic.',
    0: 'Not Relevant. This entity is not useful or on topic.',
}

CITATION = 'dblp:conf/naacl/PetroniPFLYCTJK21; dblp:conf/sigir/MackieOGFM022'


class _KiltDocsParser(Parser):
    name = 'KiltDocs'

    def build(self, source, node):
        return KiltDocs(source, count_hint=node.count_hint)


# Files
# -----------------------------------------
knowledgesource_file = Resource('kilt-knowledgesource.json',
    sources=['http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json'],
    md5='d1dca62aa6ba889d2e842182e3114af5',
    size=37_318_876_722,
)
qrels_file = Resource('kilt-codec-qrels.txt',
    sources=['https://raw.githubusercontent.com/grill-lab/CODEC/main/raw_judgments/raw_entity_judgments.txt'],
    md5='51781fd0de5f7ca6b537222e4001e8ba',
    size=282_367,
)

# Tables
# -----------------------------------------
docs = DocTable('kilt-docs',
    source=knowledgesource_file,
    parser=_KiltDocsParser(),
    lang='en',
    count_hint=5_903_530,
)
qrels = TrecQrels('kilt-codec-qrels', source=qrels_file, defs=CODEC_QREL_DEFS, count_hint=11_323)

# Benchmarks
# -----------------------------------------
kilt_codec = Benchmark('kilt-codec',
    docs=docs, queries=codec_queries, qrels=qrels,
    citation=CITATION,
    desc='KILT entity-ranking task: KILT Wikipedia entities judged relevant '
         "to CODEC's own 42 queries (economics, history, and politics).")

domain_benchmarks = {}
_DOMAIN_COUNTS = {
    'economics': 1_970,
    'history': 2_024,
    'politics': 2_192,
}
for _domain, _rcount in _DOMAIN_COUNTS.items():
    domain_benchmarks[_domain] = Benchmark(f'kilt-codec-{_domain}',
        derived_from=kilt_codec,
        queries=codec_domain_benchmarks[_domain].queries,
        filter=Filter(query_ids=ids_of(f'irds:codec-{_domain}-queries'), mode='include'),
        citation=CITATION,
        desc=f'KILT entity-ranking task, {_domain} domain subset.')


# Registration
# -----------------------------------------
irds.register(kilt_codec, *domain_benchmarks.values())
