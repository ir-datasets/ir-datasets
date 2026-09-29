"""CSL -- a v2 dataset family.

CSL (Chinese Scientific Literature) is the corpus behind TREC NeuCLIR 2023's
technical-document task, and has no other use -- so unlike ``trec_web.py``/
``trec_tb.py``/``trec_mq.py``/``trec_misinfo.py`` (which exist because those
tracks span multiple corpora/years), corpus and benchmark stay in this one
file, same reasoning as ``car.py`` bundling its own ``trec-y1`` benchmark.

The corpus is JSONL, read through v1's ``JsonlDocs`` directly (``CslDoc`` is
imported from legacy ``csl.py`` as a type only, same pattern as
``clinicaltrials.py`` importing record types from ``medline.py``). Queries
are v1's ``ExctractedCCQueries``/``ExctractedCCNoReportQuery`` (note the
actual misspelling "Exctracted" -- kept, it's the real v1 name), a format
narrow enough that it has no general-purpose v2 node, so it's wrapped via a
local ``Parser`` (same shape as ``codec.py``'s ``_CodecQueriesParser``).
Qrels are a single member pulled out of a tar archive via ``.member(...)``,
then the standard v2 ``TrecQrels`` format node -- no custom parser needed.
"""
from ir_datasets.datasets.csl import CslDoc
from ir_datasets.formats import ExctractedCCNoReportQuery, ExctractedCCQueries as _V1ExctractedCCQueries
from ir_datasets.v2 import Benchmark, DocTable, QueryTable, Resource, Source, TrecQrels, irds
from ir_datasets.v2.formats import Parser

NAME = 'csl'

QREL_DEFS = {
    3: 'Very-valuable. Information in the document would be found in the lead paragraph of a report that is later written on the topic.',
    1: 'Somewhat-valuable. The most valuable information in the document would be found in the remainder of such a report.',
    0: 'Not-valuable. Information in the document might be included in a report footnote, or omitted entirely.',
}


class _JsonlDocsParser(Parser):
    name = 'JsonlDocs'

    def build(self, source, node):
        from ir_datasets.formats import JsonlDocs
        return JsonlDocs(source, doc_cls=CslDoc, namespace=NAME,
            lang=node.lang, count_hint=node.count_hint, docstore_path=str(node.docstore_path))


class _ExctractedCCQueriesParser(Parser):
    name = 'ExctractedCCQueries'

    def build(self, source, node):
        return _V1ExctractedCCQueries(source, subset_lang='zh', filter_lwq=False,
            cls=ExctractedCCNoReportQuery, namespace=NAME)


# Files
# -----------------------------------------
docs_file = Resource('csl-docs.jsonl.gz',
    sources=['https://huggingface.co/datasets/neuclir/csl/resolve/main/data/csl.jsonl.gz?download=true'],
    md5='4198f7b442187320e2351b3b473c1883',
    size=115_749_077,
)
queries_file = Resource('csl-trec-2023-queries.jsonl',
    sources=['https://trec.nist.gov/data/neuclir/2023/neuclir-2023-technical_topics.0719.jsonl', Source.mirror()],
    md5='0dd5ba173c695362a8705056edca481b',
    size=86_519,
)
qrels_file = Resource('csl-trec-2023-qrels.tar.gz',
    sources=['https://trec.nist.gov/data/neuclir/2023/neuclir-2023-qrels.final.tar.gz', Source.mirror()],
    md5='cea4ff3d9eba612c7119e6490217d4e1',
    size=6_023_886,
)

# Tables
# -----------------------------------------
docs = DocTable('csl-docs',
    source=docs_file.gunzip(),
    parser=_JsonlDocsParser(),
    lang='zh',
    count_hint=395_927,
)
queries = QueryTable('csl-trec-2023-queries',
    source=queries_file,
    parser=_ExctractedCCQueriesParser(),
    lang='zh',
    count_hint=41,
)
qrels = TrecQrels('csl-trec-2023-qrels',
    source=qrels_file.member('tech_final_qrels.txt'),
    defs=QREL_DEFS,
    count_hint=11_291,
)

# Benchmarks
# -----------------------------------------
csl_trec_2023 = Benchmark('csl-trec-2023',
    docs=docs, queries=queries, qrels=qrels,
    citation='dblp:conf/coling/LiZ0S0MZ22',
    desc='TREC NeuCLIR 2023 technical-document task.')


# Registration
# -----------------------------------------
irds.register(csl_trec_2023)
