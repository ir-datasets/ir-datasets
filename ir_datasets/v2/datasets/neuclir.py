"""NeuCLIR -- a v2 dataset family.

Three large Common Crawl CLIR corpora (Chinese, Persian, Russian -- the same
three languages as ``hc4.py``, at ~5-10x the scale), the TREC 2022/2023
NeuCLIR track's benchmarks over them, a combined tri-lingual "multi" corpus
for TREC 2023's cross-language task, and an ``hc4-filtered`` variant per
language that subsets NeuCLIR's own corpus down to the documents HC4 also
covers (so HC4's dev/test queries and qrels can be reused for
development -- see the NeuCLIR docs' own "See also: hc4" note).

Directly imports v1's own ``ir_datasets.datasets.neuclir`` handler classes
(``FilteredExctractedCCDocs``, ``FilteredTrecQrels``, ``LangFilteredTrecQrels``,
``get_ids``) rather than re-deriving them -- these already *are* exactly the
"thin layer over v1 machinery" every v2 family wants, and they operate on
anything with a ``.stream()``, which a v2 ``Resource``/pipeline already
satisfies. Importing that v1 module also runs its own ``_init()`` (registering
the legacy ``neuclir/...`` ids), same as ``beir.py``/``msmarco_document.py``
importing their v1 counterparts for the same reason.

``hc4-filtered``'s id-membership Resources are ``hc4.py``'s own
``IDS_FILES`` -- imported directly rather than re-declared, the same
cross-file shared-reference story ``msmarco_document.py`` tells for MS
MARCO's TREC-DL query tables (there for a Table; here for a Resource).

The 2022 track's qrels are one file for all three languages (a TREC
``iteration`` column carries the language); 2023's are one tar with a member
per language plus a combined "multi" member -- read with ``.member(...)``
straight off the tar, the same pattern ``mr_tydi.py`` uses for its
per-language tar contents.
"""
from ir_datasets.util import Lazy
from ir_datasets.datasets.base import FilteredQueries
from ir_datasets.formats import (
    ExctractedCCDocs as _V1ExctractedCCDocs,
    ExctractedCCMultiMtQuery,
    ExctractedCCNoReportNoHtNarQuery,
    ExctractedCCNoReportQuery,
    ExctractedCCQueries as _V1ExctractedCCQueries,
)
from ir_datasets.datasets.neuclir import (
    FilteredExctractedCCDocs as _V1FilteredExctractedCCDocs,
    FilteredTrecQrels as _V1FilteredTrecQrels,
    LangFilteredTrecQrels as _V1LangFilteredTrecQrels,
)

from ir_datasets.v2 import Benchmark, DocTable, Parser, QueryTable, Resource, TrecQrels, irds
from ir_datasets.v2.nodes import QrelTable as _QrelTable
from ir_datasets.v2.datasets.hc4 import (
    IDS_FILES as HC4_IDS_FILES, LANG3, QREL_DEFS, QRELS_FILES as HC4_QRELS_FILES,
    TOPICS_FILES as HC4_TOPICS_FILES,
)

CITATION = 'dblp:conf/trec/LawrieMMMOSY22'

DOC_COUNTS = {'zh': 3_179_209, 'fa': 2_232_016, 'ru': 4_627_543}

#: lang -> (docs url, md5, size), from HuggingFace's neuclir/neuclir1.
DOCS = {
    'fa': ('https://huggingface.co/datasets/neuclir/neuclir1/resolve/main/data/fas-00000-of-00001.jsonl.gz?download=true', 'c88f79f6b6da974db22cef3dd73fcee1', 2_359_094_118),
    'zh': ('https://huggingface.co/datasets/neuclir/neuclir1/resolve/main/data/zho-00000-of-00001.jsonl.gz?download=true', '99eb400f3a474603d1db5d41f606889b', 3_188_072_408),
    'ru': ('https://huggingface.co/datasets/neuclir/neuclir1/resolve/main/data/rus-00000-of-00001.jsonl.gz?download=true', '3aabc798a3b5dd92d7c47db9521870b1', 4_504_119_267),
}

QUERIES_2022 = Resource('neuclir-trec-2022-queries.jsonl',
    sources=['https://trec.nist.gov/data/neuclir/2022/topics.0720.utf8.jsonl'],
    md5='264bf244f798670f063f32ff57ba6135', size=662_272)
QRELS_2022 = Resource('neuclir-trec-2022-qrels.txt',
    sources=['https://trec.nist.gov/data/neuclir/2022/2022-qrels.all'],
    md5='8dc1aecf13fbe358eea74ade7496b085', size=4_785_668)
QUERIES_2023 = Resource('neuclir-trec-2023-queries.jsonl',
    sources=['https://trec.nist.gov/data/neuclir/2023/neuclir-2023-topics.0605.jsonl'],
    md5='3dbb41b02bfbd719d8b55632d9b15b83', size=683_779)
QRELS_2023_TAR = Resource('neuclir-trec-2023-qrels.tar.gz',
    sources=['https://trec.nist.gov/data/neuclir/2023/neuclir-2023-qrels.final.tar.gz'],
    md5='cea4ff3d9eba612c7119e6490217d4e1', size=6_023_886)


class _NeuclirDocsParser(Parser):
    name = 'ExctractedCCDocs'

    def __init__(self, lang, count=None):
        self.lang = lang
        self.count = count

    def build(self, source, node):
        return _V1ExctractedCCDocs(source, subset_lang=self.lang, count=self.count)


class _NeuclirHc4FilteredDocsParser(Parser):
    name = 'FilteredExctractedCCDocs'

    def __init__(self, lang, include_doc_id_dlc):
        self.lang = lang
        self.include_doc_id_dlc = include_doc_id_dlc

    def build(self, source, node):
        return _V1FilteredExctractedCCDocs(
            source, subset_lang=self.lang, include_doc_id_dlc=self.include_doc_id_dlc)


class _NeuclirQueriesParser(Parser):
    """Full (unfiltered) topic set for one language -- v1's plain
    ``ExctractedCCQueries``, no id filtering."""
    name = 'ExctractedCCQueries'

    def __init__(self, lang, cls):
        self.lang = lang
        self.cls = cls

    def build(self, source, node):
        return _V1ExctractedCCQueries(source, subset_lang=self.lang,
                                      filter_lwq=False, cls=self.cls)


class _NeuclirFilteredQueriesParser(Parser):
    """A language's topic set, filtered down to the ids a (separately built,
    not-necessarily-registered) v1 qrels handler judges -- v1's own
    ``FilteredQueries`` + ``_lazy_qids_set`` pattern, wrapping whatever
    ``ExctractedCCQueries`` variant this year uses."""
    name = 'FilteredQueries'

    def __init__(self, lang, cls, qrels_handler):
        self.lang = lang
        self.cls = cls
        self.qrels_handler = qrels_handler

    def build(self, source, node):
        handler = _V1ExctractedCCQueries(source, subset_lang=self.lang,
                                         filter_lwq=False, cls=self.cls)
        lazy_qids = Lazy(lambda: {q.query_id for q in self.qrels_handler.qrels_iter()})
        return FilteredQueries(handler, lazy_qids, mode='include')


class _NeuclirLangQrelsParser(Parser):
    """The shared 2022 qrels file, filtered to one language by its
    ``iteration`` column -- v1's own ``LangFilteredTrecQrels``."""
    name = 'LangFilteredTrecQrels'

    def __init__(self, lang3):
        self.lang3 = lang3

    def build(self, source, node):
        return _V1LangFilteredTrecQrels(source, node.defs or {}, self.lang3)


class _NeuclirHc4FilteredQrelsParser(Parser):
    name = 'FilteredTrecQrels'

    def __init__(self, include_doc_id_dlc):
        self.include_doc_id_dlc = include_doc_id_dlc

    def build(self, source, node):
        return _V1FilteredTrecQrels(source, node.defs or {}, self.include_doc_id_dlc)


def _hc4_ids_dlc(lang):
    """v1's own shape: a single dlc for zh/fa, a tuple of 8 shards for ru --
    see ``get_ids``'s ``lru_cache`` (it needs a hashable key either way)."""
    files = HC4_IDS_FILES[lang]
    return files[0] if len(files) == 1 else tuple(files)


#: lang -> the one Resource for that language's docs -- shared by its own
#: docs/hc4-filtered tables below and by the combined "multi" corpus, so it
#: is registered exactly once (a second ``Resource(...)`` with the same name
#: would be a distinct object under one name, which the registry only
#: tolerates when it's a plain re-import -- see ``ManifestProvider.register``).
DOCS_FILES = {
    _lang: Resource(f'neuclir-{_lang}-docs.jsonl.gz', sources=[_url], md5=_md5, size=_size)
    for _lang, (_url, _md5, _size) in DOCS.items()
}

_benchmarks = []

for _lang in DOCS:
    _lang3 = LANG3[_lang]
    _docs_file = DOCS_FILES[_lang]

    _docs = DocTable(f'neuclir-{_lang}-docs',
        source=_docs_file.gunzip(), parser=_NeuclirDocsParser(_lang, DOC_COUNTS[_lang]),
        lang=_lang, desc=f'NeuCLIR collection 1, {_lang} Common Crawl documents.')

    # -- TREC 2022: shared multi-language qrels file, filtered by iteration --
    # TrecQrels' own parser= is fixed (_TrecQrelsParser); the language filter
    # needs LangFilteredTrecQrels instead, so this facet is built directly on
    # a bare QrelTable rather than through the TrecQrels format node.
    _name_2022 = f'neuclir-{_lang}-trec-2022'
    _qrels_2022 = _QrelTable(f'{_name_2022}-qrels',
        source=QRELS_2022, parser=_NeuclirLangQrelsParser(_lang3), defs=QREL_DEFS)
    _queries_2022 = QueryTable(f'{_name_2022}-queries',
        source=QUERIES_2022,
        parser=_NeuclirFilteredQueriesParser(
            _lang, ExctractedCCNoReportQuery, _qrels_2022.handler))
    _benchmarks.append(Benchmark(_name_2022,
        docs=_docs, queries=_queries_2022, qrels=_qrels_2022,
        citation=CITATION,
        desc=f'NeuCLIR collection 1, {_lang}: TREC 2022 NeuCLIR track benchmark.'))

    # -- TREC 2023: one tar, a qrels member per language --
    _name_2023 = f'neuclir-{_lang}-trec-2023'
    _qrels_2023 = TrecQrels(f'{_name_2023}-qrels',
        source=QRELS_2023_TAR.member(f'qrels.final.gains.{_lang3}'), defs=QREL_DEFS)
    _queries_2023 = QueryTable(f'{_name_2023}-queries',
        source=QUERIES_2023,
        parser=_NeuclirFilteredQueriesParser(
            _lang, ExctractedCCNoReportNoHtNarQuery, _qrels_2023.handler))
    _benchmarks.append(Benchmark(_name_2023,
        docs=_docs, queries=_queries_2023, qrels=_qrels_2023,
        citation=CITATION,
        desc=f'NeuCLIR collection 1, {_lang}: TREC 2023 NeuCLIR track benchmark.'))

    # -- hc4-filtered: NeuCLIR's own corpus, subset to HC4's overlap -- HC4's
    # dev+test queries/qrels stand in for a NeuCLIR-native judged set.
    _name_hc4f = f'neuclir-{_lang}-hc4-filtered'
    _include_doc_id_dlc = _hc4_ids_dlc(_lang)
    _hc4f_docs = DocTable(f'{_name_hc4f}-docs',
        source=_docs_file.gunzip(),
        parser=_NeuclirHc4FilteredDocsParser(_lang, _include_doc_id_dlc), lang=_lang,
        desc=f'NeuCLIR collection 1 {_lang} documents, filtered to the ids '
            'HC4 also covers.')
    _hc4f_queries = QueryTable(f'{_name_hc4f}-queries',
        source=[HC4_TOPICS_FILES['dev'], HC4_TOPICS_FILES['test']],
        parser=_NeuclirQueriesParser(_lang, ExctractedCCNoReportQuery))
    _hc4f_qrels = _QrelTable(f'{_name_hc4f}-qrels',
        source=[HC4_QRELS_FILES[(_lang, 'dev')], HC4_QRELS_FILES[(_lang, 'test')]],
        parser=_NeuclirHc4FilteredQrelsParser(_include_doc_id_dlc), defs=QREL_DEFS)
    _benchmarks.append(Benchmark(_name_hc4f,
        docs=_hc4f_docs, queries=_hc4f_queries, qrels=_hc4f_qrels,
        citation=CITATION,
        desc=f'NeuCLIR collection 1, {_lang}, filtered to intersect with HC4 '
            '-- HC4\'s combined dev+test queries and qrels.'))

# -- Combined tri-lingual corpus, TREC 2023's cross-language "multi" task --
_multi_docs = DocTable('neuclir-multi-docs',
    source=[DOCS_FILES[_lang].gunzip() for _lang in ('zh', 'fa', 'ru')],
    parser=_NeuclirDocsParser(None, sum(DOC_COUNTS.values())),
    desc='NeuCLIR collection 1, the three languages (zh/fa/ru) combined.')

_multi_qrels = TrecQrels('neuclir-multi-trec-2023-qrels',
    source=QRELS_2023_TAR.member('qrels.final.gains'), defs=QREL_DEFS)
_multi_queries = QueryTable('neuclir-multi-trec-2023-queries',
    source=QUERIES_2023, parser=_NeuclirQueriesParser(None, ExctractedCCMultiMtQuery))
_benchmarks.append(Benchmark('neuclir-multi-trec-2023',
    docs=_multi_docs, queries=_multi_queries, qrels=_multi_qrels,
    citation=CITATION,
    desc='NeuCLIR collection 1: TREC 2023\'s cross-language "multi" task '
        '(one combined tri-lingual ranking per query).'))


# Registration
# -----------------------------------------
irds.register(*_benchmarks)

