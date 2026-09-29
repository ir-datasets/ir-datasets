"""SARA -- a v2 dataset family (a single, self-contained sensitivity-aware
relevance assessment collection: one corpus, one query set, one qrels file,
one benchmark; no subsets, no TREC branding).

The corpus is a zip archive containing a single CSV file (columns ``docno``,
``text``, ``sensitivity``), read through v1's own ``SaraDocs`` handler
directly rather than reimplemented -- it opens the zip in memory, reads the
*first* file inside it, and parses it with a hand-raised
``csv.field_size_limit`` (some rows have very long text); that field-size fix
is a real behavioral detail worth keeping, not incidental, so it's reused
via a thin ``Parser`` wrapper rather than redone here. Queries are a plain
``query_id<TAB>text`` TSV and qrels are plain trec-qrels-shaped lines, so
both use v2's general-purpose ``TsvQueries``/``TrecQrels`` format nodes
directly, same as e.g. ``dpr_w100.py``'s corpus.
"""
from ir_datasets.datasets.sara import SaraDoc, SaraDocs
from ir_datasets.v2 import Benchmark, DocTable, Resource, TrecQrels, TsvQueries, irds
from ir_datasets.v2.formats import Parser

QREL_DEFS = {
    2: 'highly relevant',
    1: 'partially relevant',
    0: 'not relevant',
}


class _SaraDocsParser(Parser):
    name = 'SaraDocs'

    def build(self, source, node):
        return SaraDocs(source)


# Files
# -----------------------------------------
docs_file = Resource('sara-docs.zip',
    sources=['https://zenodo.org/records/18609870/files/sara_combined_docs.zip?download=1'],
    hash='md5:e806b1d5ce35c94cec2899e190db7dd7',
)
queries_file = Resource('sara-queries.tsv',
    sources=['https://raw.githubusercontent.com/JackMcKechnie/SARA-A-Collection-of-Sensitivity-Aware-Relevance-Assessments/main/repeated_queries.tsv'],
    hash='md5:fc0247928a0b93bb344068fa238a5e3f',
)
qrels_file = Resource('sara-qrels.txt',
    sources=['https://raw.githubusercontent.com/JackMcKechnie/SARA/refs/heads/main/combined_qrels.txt'],
    hash='md5:39a24d38b4d0e352e7818abd09d6815a',
)

# Tables
# -----------------------------------------
docs = DocTable('sara-docs',
    source=docs_file,
    parser=_SaraDocsParser(),
    lang='en',
    count_hint=1_702,
)
queries = TsvQueries('sara-queries',
    source=queries_file,
    lang='en',
    count_hint=150,
)
qrels = TrecQrels('sara-qrels', source=qrels_file, defs=QREL_DEFS, count_hint=34_413)

# Benchmarks
# -----------------------------------------
sara = Benchmark('sara',
    docs=docs, queries=queries, qrels=qrels,
    citation='dblp:journals/corr/abs-2401-05144',
    desc='SARA: a collection of sensitivity-aware relevance assessments.')


# Registration
# -----------------------------------------
irds.register(sara)
