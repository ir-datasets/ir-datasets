"""TREC NeuCLIR 2022/2023 -- a v2 dataset family: the track's queries, qrels
and benchmarks over NeuCLIR collection 1 (``neuclir1.py``, whose docs tables
are imported from there, not re-declared).

The 2022 track's qrels are one file for all three languages (a TREC
``iteration`` column carries the language); 2023's are one tar with a member
per language plus a combined "multi" member and the technical-document task's
(``tech_final_qrels.txt``) -- read with ``.member(...)`` straight off the tar,
the same pattern ``mr_tydi.py`` uses for its per-language tar contents. 2023's
cross-language "multi" task ranks one combined tri-lingual corpus
(``neuclir1-multi``) per query; its technical-document task
(``trec-neuclir-2023-tech``) is over the Chinese Scientific Literature corpus
(``csl.py``'s ``csl-docs``), with its own topics.

Reuses v1's own ``ir_datasets.datasets.neuclir`` handler classes
(``LangFilteredTrecQrels``) and ``ExctractedCCQueries`` variants rather than
re-deriving them -- the usual "thin layer over v1 machinery".
"""
from ir_datasets.util import Lazy
from ir_datasets.datasets.base import FilteredQueries
from ir_datasets.formats import (
    ExctractedCCMultiMtQuery,
    ExctractedCCNoReportNoHtNarQuery,
    ExctractedCCNoReportQuery,
    ExctractedCCQueries as _V1ExctractedCCQueries,
)
from ir_datasets.datasets.neuclir import LangFilteredTrecQrels as _V1LangFilteredTrecQrels

from ir_datasets.v2 import Benchmark, Parser, QueryTable, Resource, Source, TrecQrels, irds
from ir_datasets.v2.nodes import QrelTable as _QrelTable
from ir_datasets.v2.datasets.csl import docs as CSL_DOCS
from ir_datasets.v2.datasets.hc4 import LANG3, QREL_DEFS
from ir_datasets.v2.datasets.neuclir1 import DOCS_TABLES, MULTI_DOCS

CITATION = 'dblp:conf/trec/LawrieMMMOSY22'

QUERIES_2022 = Resource('trec-neuclir-2022-queries.jsonl',
    sources=['https://trec.nist.gov/data/neuclir/2022/topics.0720.utf8.jsonl'],
    hash='md5:264bf244f798670f063f32ff57ba6135', size=662_272)
QRELS_2022 = Resource('trec-neuclir-2022-qrels.txt',
    sources=['https://trec.nist.gov/data/neuclir/2022/2022-qrels.all'],
    hash='md5:8dc1aecf13fbe358eea74ade7496b085', size=4_785_668)
QUERIES_2023 = Resource('trec-neuclir-2023-queries.jsonl',
    sources=['https://trec.nist.gov/data/neuclir/2023/neuclir-2023-topics.0605.jsonl'],
    hash='md5:3dbb41b02bfbd719d8b55632d9b15b83', size=683_779)
QRELS_2023_TAR = Resource('trec-neuclir-2023-qrels.tar.gz',
    sources=['https://trec.nist.gov/data/neuclir/2023/neuclir-2023-qrels.final.tar.gz', Source.mirror()],
    hash='md5:cea4ff3d9eba612c7119e6490217d4e1', size=6_023_886)
QUERIES_2023_TECH = Resource('trec-neuclir-2023-tech-queries.jsonl',
    sources=['https://trec.nist.gov/data/neuclir/2023/neuclir-2023-technical_topics.0719.jsonl', Source.mirror()],
    hash='md5:0dd5ba173c695362a8705056edca481b', size=86_519)

TECH_QREL_DEFS = {
    3: 'Very-valuable. Information in the document would be found in the lead paragraph of a report that is later written on the topic.',
    1: 'Somewhat-valuable. The most valuable information in the document would be found in the remainder of such a report.',
    0: 'Not-valuable. Information in the document might be included in a report footnote, or omitted entirely.',
}


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


class _NeuclirTechQueriesParser(Parser):
    """The technical-document task's topics (Chinese): v1's
    ``ExctractedCCQueries`` with the no-report query class."""
    name = 'ExctractedCCQueries'

    def build(self, source, node):
        return _V1ExctractedCCQueries(source, subset_lang='zh', filter_lwq=False,
            cls=ExctractedCCNoReportQuery, namespace='csl')


class _NeuclirLangQrelsParser(Parser):
    """The shared 2022 qrels file, filtered to one language by its
    ``iteration`` column -- v1's own ``LangFilteredTrecQrels``."""
    name = 'LangFilteredTrecQrels'

    def __init__(self, lang3):
        self.lang3 = lang3

    def build(self, source, node):
        return _V1LangFilteredTrecQrels(source, node.defs or {}, self.lang3)


_benchmarks = []

for _lang, _docs in DOCS_TABLES.items():
    _lang3 = LANG3[_lang]

    # -- TREC 2022: shared multi-language qrels file, filtered by iteration --
    # TrecQrels' own parser= is fixed (_TrecQrelsParser); the language filter
    # needs LangFilteredTrecQrels instead, so this facet is built directly on
    # a bare QrelTable rather than through the TrecQrels format node.
    _name_2022 = f'trec-neuclir-2022-{_lang}'
    _qrels_2022 = _QrelTable(f'{_name_2022}-qrels',
        source=QRELS_2022, parser=_NeuclirLangQrelsParser(_lang3), defs=QREL_DEFS)
    _queries_2022 = QueryTable(f'{_name_2022}-queries',
        source=QUERIES_2022,
        parser=_NeuclirFilteredQueriesParser(
            _lang, ExctractedCCNoReportQuery, _qrels_2022.handler))
    _benchmarks.append(Benchmark(_name_2022,
        docs=_docs, queries=_queries_2022, qrels=_qrels_2022,
        citation=CITATION,
        desc=f'TREC 2022 NeuCLIR track, {_lang} (NeuCLIR collection 1).'))

    # -- TREC 2023: one tar, a qrels member per language --
    _name_2023 = f'trec-neuclir-2023-{_lang}'
    _qrels_2023 = TrecQrels(f'{_name_2023}-qrels',
        source=QRELS_2023_TAR.member(f'qrels.final.gains.{_lang3}'), defs=QREL_DEFS)
    _queries_2023 = QueryTable(f'{_name_2023}-queries',
        source=QUERIES_2023,
        parser=_NeuclirFilteredQueriesParser(
            _lang, ExctractedCCNoReportNoHtNarQuery, _qrels_2023.handler))
    _benchmarks.append(Benchmark(_name_2023,
        docs=_docs, queries=_queries_2023, qrels=_qrels_2023,
        citation=CITATION,
        desc=f'TREC 2023 NeuCLIR track, {_lang} (NeuCLIR collection 1).'))

# -- TREC 2023's cross-language "multi" task: one combined tri-lingual
# ranking per query, over the combined corpus --
_multi_qrels = TrecQrels('trec-neuclir-2023-multi-qrels',
    source=QRELS_2023_TAR.member('qrels.final.gains'), defs=QREL_DEFS)
_multi_queries = QueryTable('trec-neuclir-2023-multi-queries',
    source=QUERIES_2023, parser=_NeuclirQueriesParser(None, ExctractedCCMultiMtQuery))
_benchmarks.append(Benchmark('trec-neuclir-2023-multi',
    docs=MULTI_DOCS, queries=_multi_queries, qrels=_multi_qrels,
    citation=CITATION,
    desc='TREC 2023 NeuCLIR track, the cross-language "multi" task over '
        'NeuCLIR collection 1 (one combined tri-lingual ranking per query).'))

# -- TREC 2023's technical-document task: its own (Chinese) topics over the CSL
# corpus, with qrels from the same 2023 tar as the other tasks --
_tech_queries = QueryTable('trec-neuclir-2023-tech-queries',
    source=QUERIES_2023_TECH, parser=_NeuclirTechQueriesParser(), lang='zh', count_hint=41)
_tech_qrels = TrecQrels('trec-neuclir-2023-tech-qrels',
    source=QRELS_2023_TAR.member('tech_final_qrels.txt'), defs=TECH_QREL_DEFS, count_hint=11_291)
_benchmarks.append(Benchmark('trec-neuclir-2023-tech',
    docs=CSL_DOCS, queries=_tech_queries, qrels=_tech_qrels,
    citation='dblp:conf/coling/LiZ0S0MZ22',
    desc='TREC 2023 NeuCLIR track, the technical-document task (Chinese Scientific Literature).'))

# Registration
# -----------------------------------------
irds.register(*_benchmarks)
