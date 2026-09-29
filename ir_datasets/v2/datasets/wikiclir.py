"""WikiCLIR -- a v2 dataset family (Wikipedia-derived cross-lingual
information retrieval: English queries against 24 target-language document
collections, judged by inter-language links).

Everything -- 25 languages' worth of documents, one shared set of English
queries, and 25 languages' worth of qrels -- comes from a single ~7GB
downloadable tar.gz, extracted once via v1's own ``TarExtractAll`` to a fixed
directory (``<home>/wikiclir/wikiclir_extracted``), the same shape
``istella22.py``/``c4.py``/``clueweb09.py`` use for their own checkpoint
archives; individual member files are then read off that extracted tree via
v1's ``RelativePath``, wrapped in thin ``Parser``s that reuse v2's own
``TsvDocs``/``TrecQrels`` v1 handlers -- same split every table in
``istella22.py`` uses (``source=`` stays the raw ``source_file`` Resource, for
a correct ``derived_from`` edge; the actual extraction path is read through
the shared ``_base_dlc`` inside each ``Parser``).

The 25 per-language *query* sets are not independent downloads: v1 filters one
shared English query file (Wikipedia article titles used as CLIR queries) down
to each language's own judged query ids at *parse* time
(``FilteredQueries``/``Lazy``, reusing v1's own machinery directly, same shape
as ``sara.py``'s reuse of ``SaraDocs``) rather than as a separate v2
``Filter``/``Benchmark.derived_from`` derivation -- that mechanism assumes a
docs+queries+qrels *bundle* is filtered together from one parent benchmark,
but here only the queries are shared/filtered; the docs and qrels differ
per language and are not filtered from anything. So each language gets its
own ``wikiclir-<lang>-queries`` ``QueryTable``, built by a ``Parser`` that
re-parses the shared English query file and filters it to that language's own
qrels' query ids, exactly mirroring v1's ``_qid_filter``.

Per v1, ``en-simple`` (the docs/qrels judged against Wikipedia's Simple
English edition) is included among the CLIR targets even though its queries
and documents are both English -- preserved as-is, not treated specially.
"""
import ir_datasets
from ir_datasets.util import RelativePath, TarExtractAll
from ir_datasets.v2 import Benchmark, DocTable, QrelTable, QueryTable, Resource, irds
from ir_datasets.v2.formats import Parser

NAME = 'wikiclir'
BASE_PATH = ir_datasets.util.home_path() / NAME

CITATION = 'dblp:conf/naacl/SasakiSSDI18'

QRELS_DEFS = {
    2: "Document assigned to the (English) cross-lingual mate",
    1: "All other articles that link to the mate, and are linked by the mate",
}

# (source directory, ISO code, dataset-name suffix)
LANGS = [
    ('arabic', 'ar', 'ar'),
    ('catalan', 'ca', 'ca'),
    ('chinese', 'zh', 'zh'),
    ('czech', 'cs', 'cs'),
    ('dutch', 'nl', 'nl'),
    ('finnish', 'fi', 'fi'),
    ('french', 'fr', 'fr'),
    ('german', 'de', 'de'),
    ('italian', 'it', 'it'),
    ('japanese', 'ja', 'ja'),
    ('korean', 'ko', 'ko'),
    ('norwegian_(bokmal)', 'no', 'no'),
    ('norwegian_(nynorsk)', 'nn', 'nn'),
    ('polish', 'pl', 'pl'),
    ('portuguese', 'pt', 'pt'),
    ('romanian', 'ro', 'ro'),
    ('russian', 'ru', 'ru'),
    ('simple_english', 'en', 'en-simple'),
    ('spanish', 'es', 'es'),
    ('swahili', 'sw', 'sw'),
    ('swedish', 'sv', 'sv'),
    ('tagalog', 'tl', 'tl'),
    ('turkish', 'tr', 'tr'),
    ('ukrainian', 'uk', 'uk'),
    ('vietnamese', 'vi', 'vi'),
]

_COUNTS = {
    # dsid: (docs, queries, qrels)
    'ar': (535_118, 324_489, 519_269),
    'ca': (548_722, 339_586, 965_233),
    'cs': (386_906, 233_553, 954_370),
    'de': (2_091_278, 938_217, 5_550_454),
    'en-simple': (127_089, 114_572, 250_380),
    'es': (1_302_958, 781_642, 2_894_807),
    'fi': (418_677, 273_819, 939_613),
    'fr': (1_894_397, 1_089_179, 5_137_366),
    'it': (1_347_011, 808_605, 3_443_633),
    'ja': (1_071_292, 426_431, 3_338_667),
    'ko': (394_177, 224_855, 568_205),
    'nl': (1_908_260, 687_718, 2_334_644),
    'nn': (133_290, 99_493, 250_141),
    'no': (471_420, 299_897, 963_514),
    'pl': (1_234_316, 693_656, 2_471_360),
    'pt': (973_057, 611_732, 1_741_889),
    'ro': (376_655, 199_264, 451_180),
    'ru': (1_413_945, 664_924, 2_321_384),
    'sv': (3_785_412, 639_073, 2_069_453),
    'sw': (37_079, 22_860, 57_924),
    'tl': (79_008, 48_930, 72_359),
    'tr': (295_593, 185_388, 380_651),
    'uk': (704_903, 348_222, 913_358),
    'vi': (1_392_152, 354_312, 611_355),
    'zh': (951_480, 463_273, 926_130),
}


class _WikiClirDocsParser(Parser):
    name = 'TsvDocs'

    def __init__(self, relpath):
        self.relpath = relpath

    def build(self, source, node):
        from ir_datasets.datasets.wikiclir import WikiClirDoc
        from ir_datasets.formats import TsvDocs as _V1TsvDocs
        file = RelativePath(_base_dlc, self.relpath)
        return _V1TsvDocs(file, doc_cls=WikiClirDoc, lang=node.lang,
                          docstore_size_hint=node.docstore_size_hint,
                          count_hint=node.count_hint)


class _WikiClirQrelsParser(Parser):
    name = 'TrecQrels(3col)'

    def __init__(self, relpath):
        self.relpath = relpath

    def build(self, source, node):
        from ir_datasets.formats import TrecQrels as _V1TrecQrels
        file = RelativePath(_base_dlc, self.relpath)
        return _V1TrecQrels(file, node.defs or {}, format_3col=True)


class _WikiClirQueriesParser(Parser):
    """Filters the one shared English query file down to this language's own
    judged query ids -- see the module docstring for why this isn't a
    ``Filter``/``Benchmark.derived_from`` derivation."""
    name = 'FilteredQueries(TsvQueries)'

    def __init__(self, qrels_relpath):
        self.qrels_relpath = qrels_relpath

    def build(self, source, node):
        from ir_datasets.datasets.base import FilteredQueries
        from ir_datasets.datasets.wikiclir import WikiClirQuery
        from ir_datasets.formats import TrecQrels as _V1TrecQrels
        from ir_datasets.formats import TsvQueries as _V1TsvQueries
        from ir_datasets.util import Lazy
        queries_file = RelativePath(_base_dlc, 'wiki-clir/english/wiki_en.queries')
        base_queries = _V1TsvQueries(queries_file, query_cls=WikiClirQuery, lang='en')
        qrels_file = RelativePath(_base_dlc, self.qrels_relpath)
        qrels_handler = _V1TrecQrels(qrels_file, QRELS_DEFS, format_3col=True)
        qids = Lazy(lambda: {q.query_id for q in qrels_handler.qrels_iter()})
        return FilteredQueries(base_queries, qids, mode='include')


# Files
# -----------------------------------------
source_file = Resource('wikiclir-source.tar.gz',
    sources=['https://www.cs.jhu.edu/~kevinduh/a/wikiclir2018/wiki-clir.tar.gz'],
    hash='md5:705abb611eb8cbab9ced2b8767a3bdb6',
    size=7_036_445_773,
)

# Shared, extracted-once pipeline -- see module docstring.
_base_dlc = TarExtractAll(source_file, BASE_PATH / 'wikiclir_extracted')

# Tables + Benchmarks
# -----------------------------------------
benchmarks = {}
for _source_dir, _lang, _dsid in LANGS:
    _file_suffix = _lang if _dsid != 'en-simple' else 'simple'
    _docs_count, _queries_count, _qrels_count = _COUNTS[_dsid]
    _qrels_relpath = f'wiki-clir/{_source_dir}/en2{_file_suffix}.rel'

    _docs = DocTable(f'wikiclir-{_dsid}-docs',
        source=source_file,
        parser=_WikiClirDocsParser(f'wiki-clir/{_source_dir}/wiki_{_file_suffix}.documents'),
        lang=_lang, count_hint=_docs_count,
    )
    _qrels = QrelTable(f'wikiclir-{_dsid}-qrels',
        source=source_file,
        parser=_WikiClirQrelsParser(_qrels_relpath),
        defs=QRELS_DEFS, count_hint=_qrels_count,
    )
    _queries = QueryTable(f'wikiclir-{_dsid}-queries',
        source=source_file,
        parser=_WikiClirQueriesParser(_qrels_relpath),
        lang='en', count_hint=_queries_count,
    )

    benchmarks[_dsid] = Benchmark(f'wikiclir-{_dsid}',
        docs=_docs, queries=_queries, qrels=_qrels,
        citation=CITATION,
        desc=f'WikiCLIR: English queries against the {_source_dir.replace("_", " ")} '
             f'Wikipedia edition, judged by inter-language links.')


# Registration
# -----------------------------------------
irds.register(*benchmarks.values())
