"""NeuCLIR collection 1 (``neuclir1``) -- a v2 dataset family: the corpus.

Three large Common Crawl CLIR corpora (Chinese, Persian, Russian -- the same
three languages as ``hc4.py``, at ~5-10x the scale) plus a combined tri-lingual
"multi" corpus for TREC 2023's cross-language task. Corpus only: the TREC
2022/2023 NeuCLIR track's queries, qrels and benchmarks over it are in
``trec_neuclir.py``, which imports these docs tables.

Uses v1's own ``ExctractedCCDocs`` handler (``formats``) directly as the
``parser=`` wrapper -- the usual "thin layer over v1 machinery": it operates on
anything with a ``.stream()``, which a v2 ``Resource``/pipeline already
satisfies.
"""
from ir_datasets.formats import ExctractedCCDocs as _V1ExctractedCCDocs

from ir_datasets.v2 import DocTable, Parser, Resource, irds

DOC_COUNTS = {'zh': 3_179_209, 'fa': 2_232_016, 'ru': 4_627_543}

#: lang -> (docs url, md5, size), from HuggingFace's neuclir/neuclir1.
DOCS = {
    'fa': ('https://huggingface.co/datasets/neuclir/neuclir1/resolve/main/data/fas-00000-of-00001.jsonl.gz?download=true', 'c88f79f6b6da974db22cef3dd73fcee1', 2_359_094_118),
    'zh': ('https://huggingface.co/datasets/neuclir/neuclir1/resolve/main/data/zho-00000-of-00001.jsonl.gz?download=true', '99eb400f3a474603d1db5d41f606889b', 3_188_072_408),
    'ru': ('https://huggingface.co/datasets/neuclir/neuclir1/resolve/main/data/rus-00000-of-00001.jsonl.gz?download=true', '3aabc798a3b5dd92d7c47db9521870b1', 4_504_119_267),
}

class _NeuclirDocsParser(Parser):
    name = 'ExctractedCCDocs'

    def __init__(self, lang, count=None):
        self.lang = lang
        self.count = count

    def build(self, source, node):
        return _V1ExctractedCCDocs(source, subset_lang=self.lang, count=self.count)


#: lang -> the one Resource for that language's docs -- shared by its own
#: docs table below and by the combined "multi" corpus, so it is registered
#: exactly once (a second ``Resource(...)`` with the same name
#: would be a distinct object under one name, which the registry only
#: tolerates when it's a plain re-import -- see ``ManifestProvider.register``).
# license verified 2026-09-30: https://huggingface.co/datasets/neuclir/neuclir1 (card metadata: license odc-by); applies to the document corpus only
DOCS_FILES = {
    _lang: Resource(f'neuclir1-{_lang}.jsonl.gz', sources=[_url], hash=f'md5:{_md5}', size=_size,
        license='spdx:ODC-By-1.0')
    for _lang, (_url, _md5, _size) in DOCS.items()
}

DOCS_TABLES = {}   # lang -> the DocTable, imported by trec_neuclir.py

for _lang in DOCS:
    DOCS_TABLES[_lang] = DocTable(f'neuclir1-{_lang}', license='spdx:ODC-By-1.0',
        source=DOCS_FILES[_lang].gunzip(), parser=_NeuclirDocsParser(_lang, DOC_COUNTS[_lang]),
        lang=_lang, desc=f'NeuCLIR collection 1, {_lang} Common Crawl documents.')

# -- Combined tri-lingual corpus, TREC 2023's cross-language "multi" task --
MULTI_DOCS = DocTable('neuclir1-multi', license='spdx:ODC-By-1.0',
    source=[DOCS_FILES[_lang].gunzip() for _lang in ('zh', 'fa', 'ru')],
    parser=_NeuclirDocsParser(None, sum(DOC_COUNTS.values())),
    desc='NeuCLIR collection 1, the three languages (zh/fa/ru) combined.')


# Registration
# -----------------------------------------
# The docs tables are registered here, once; trec_neuclir.py's benchmarks
# reference these same objects.
irds.register(*DOCS_TABLES.values(), MULTI_DOCS)
