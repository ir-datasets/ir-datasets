"""CLIRMatrix -- a massively large collection of bilingual/multilingual CLIR
benchmarks built from Wikipedia: 139 languages x 139 languages x 4 splits x 2
size variants (``bi139base``/``bi139full``), plus a curated 8x8x4 "multi8"
subset. On the order of 150,000 dataset ids -- this is the exact family
``v2/base.py``'s ``Generator`` docstring cites as its motivating example
("a 139x139x4x3 parametric family"): far beyond what ``beir.py``'s/
``nano_beir.py``'s "loop over a short hardcoded list at import time" pattern
could afford. Five ``Generator``s below (docs + their backing Resource, the
bilingual Benchmark shape + its Queries/Qrels tables + their shared Resource)
resolve nodes on demand instead; ``freeze`` records only the rule, not the
expansion (see ``ManifestProvider.generated``). Each declares its structural
``edges=`` too (see ``base.Generator``), so a listing sees a Benchmark's real
docs/queries/qrels facets, and a table's real backing Resource, purely by
name-template substitution -- no resolution, no per-row cost even across a
family this size. The two Resource generators additionally declare
``row_metadata=`` (see ``_docs_resource_row``/``_qrel_resource_row`` below),
so a listing sees each file's real ``sources``/``hashes`` too -- at the cost
of one network fetch for the whole family (``DOWNLOADS_INDEX``, cached), not
per name.

Its own provider (``clirmatrix_provider.py``), not ``irds`` -- unlike every
other bundled family, which shares ``irds``'s registry because there's a
finite, hand-authored list of them. This one is generator-only, ~150,000
names deep, and doesn't need anything from ``irds``'s registry besides the
node types and edge kinds it borrows (``irds:Benchmark``, ``irds:DocTable``,
``irds:derived_from``, ...) -- which any provider may build nodes of/declare
edges with, regardless of who owns them (see ``registry.py``'s own docstring
on that). Being its own provider is also why names no longer repeat the
family name: bare names below are just ``{lang}-docs``, not
``clirmatrix-{lang}-docs`` -- the ``clirmatrix:`` prefix already says that,
and repeating it in every one of ~150,000 names would be pure duplication.

Names use this package's usual hyphen-flattened convention
(``{variant}-{doc_lang}-{query_lang}-{split}``, e.g.
``clirmatrix:bi139base-en-fr-train``), not v1's ``/``-segmented one
(``clirmatrix/en/bi139-base/fr/train``). Two departures from a literal
transcription of v1's shape, both intentional:

* The variant tokens are spelled ``bi139base``/``bi139full`` (no internal
  hyphen), not v1's ``bi139-base``/``bi139-full`` -- a hyphen-joined name
  with a hyphen *inside* one of its own segments reads ambiguously to a human
  (it doesn't confuse the ``Generator`` regex, which matches by each
  segment's known value set rather than splitting blindly on ``-``, but
  there's no reason to leave the trap in place).
* Languages sit adjacent (``{variant}-{doc_lang}-{query_lang}-{split}``, not
  v1's ``{doc_lang}-{variant}-{query_lang}-{split}``) so the doc/query
  language pair -- the part someone scanning a list of these actually cares
  about -- isn't split apart by the variant token.

Because the shape changed, a legacy v1 id doesn't become a valid v2 name for
free the way it would have under a literal ``/``-segment transcription --
but that only matters for an *alias* table, and one was never in scope here
regardless of naming: ``ManifestProvider.alias()`` is for a finite,
hand-listed set (see ``beir.py``), and this family has ~150,000 ids. There is
no alias for any CLIRMatrix v1 id, bare or otherwise; only the new
``clirmatrix:...`` form works.

Reuses v1's ``CLIRMatrixQueries``/``CLIRMatrixQrels`` handler classes
directly as v2 ``parser=`` wrappers (the same "thin layer over v1 machinery"
every other v2 family uses -- see ``beir.py``'s module docstring). Per-
language document corpora and resolved Benchmarks are memoized by their
params (mirrors v1's own ``_docs_cache``) so re-resolving the same name
always returns the same object -- without this,
``ManifestProvider._register_one`` would silently rebind the name to a fresh
object on every lookup instead of reusing the one already registered.

The per-file download URLs/hashes are not static (unlike ``beir.py``'s
hardcoded ``ZIP_SOURCES``) -- there's no way to hand-list ~150k of them, so
this reads them at runtime from a small remote JSON index
(``clirmatrix:downloads.json.gz``, one of the two standalone ``Resource``s
below) -- the same one v1 discovers from via ``etc/downloads.json``, now a
first-class node in its own right (hashable/attestable, shows up in a
listing) instead of an opaque v1 ``Download`` invisible to the graph. The
other standalone Resource, ``clirmatrix:metadata.json.lz4``, mirrors v1's own
``etc/downloads.json`` ``metadata`` entry; v1 read it into a
``MetadataProvider`` for per-dataset bookkeeping, but nothing in this port
consumes it (deprecated bookkeeping the v2 model no longer needs -- ``desc``/
``citation`` here are plain code, and counts/hashes come from
``freeze --verify`` instead). It's registered anyway, since it's a real
downloadable file this family depends on having *existed*, even unused.
Neither has any edges: nothing is structurally derived from either of them --
they're inputs to how other Resources get their URLs, not bytes any node is
parsed from.
"""
import json

from ir_datasets.formats import CLIRMatrixQueries as _V1CLIRMatrixQueries
from ir_datasets.formats import CLIRMatrixQrels as _V1CLIRMatrixQrels
from ir_datasets.util import Lazy

from ir_datasets.v2.base import Generator, Param
from ir_datasets.v2.formats import Parser, TsvDocs
from ir_datasets.v2.nodes import (
    BENCHMARK, Benchmark, DERIVED_FROM, FACET, QrelTable, QueryTable, RESOURCE,
    Resource, TABLE_TYPES,
)
from ir_datasets.v2.clirmatrix_provider import clirmatrix

CITATION = 'dblp:conf/emnlp/SunD20'

#: Verbatim from ir_datasets.datasets.clirmatrix -- the 139 Wikipedia
#: language codes this family covers.
LANGS = ('af', 'als', 'am', 'an', 'ar', 'arz', 'ast', 'az', 'azb', 'ba', 'bar',
         'be', 'bg', 'bn', 'bpy', 'br', 'bs', 'bug', 'ca', 'cdo', 'ce', 'ceb',
         'ckb', 'cs', 'cv', 'cy', 'da', 'de', 'diq', 'el', 'eml', 'en', 'eo',
         'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'fy', 'ga', 'gd', 'gl',
         'gu', 'he', 'hi', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'id', 'ilo',
         'io', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'kn', 'ko', 'ku', 'ky',
         'la', 'lb', 'li', 'lmo', 'lt', 'lv', 'mai', 'mg', 'mhr', 'min', 'mk',
         'ml', 'mn', 'mr', 'mrj', 'ms', 'my', 'mzn', 'nap', 'nds', 'ne', 'new',
         'nl', 'nn', 'no', 'oc', 'or', 'os', 'pa', 'pl', 'pms', 'pnb', 'ps',
         'pt', 'qu', 'ro', 'ru', 'sa', 'sah', 'scn', 'sco', 'sd', 'sh', 'si',
         'simple', 'sk', 'sl', 'sq', 'sr', 'su', 'sv', 'sw', 'szl', 'ta', 'te',
         'tg', 'th', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vec', 'vi', 'vo',
         'wa', 'war', 'wuu', 'xmf', 'yi', 'yo', 'zh')
MULTI8_LANGS = ('ar', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'zh')
SPLITS = ('train', 'dev', 'test1', 'test2')
#: v1 spells these ``bi139-base``/``bi139-full``; hyphen-free here so a
#: hyphen-joined v2 name segments unambiguously -- see the module docstring.
VARIANTS = ('bi139base', 'bi139full', 'multi8')

#: variant -> the etc/downloads.json sub-context it reads per-pair query/qrels
#: files from (docs live under a separate, variant-independent context).
_DLC_CONTEXT = {'bi139base': 'clirmatrix_bi139_base',
               'bi139full': 'clirmatrix_bi139_full',
               'multi8': 'clirmatrix_multi8'}

#: Verbatim from ir_datasets.datasets.clirmatrix -- these are automatically
#: derived (Jenks-optimized BM25 scores in the source language), not human
#: judgments.
QRELS_DEFS = {
    6: "Most relevant, based on Jenks-optimized BM25 retrieval scores in the source language",
    5: "Jenks-optimized BM25 retrieval scores in the source language",
    4: "Jenks-optimized BM25 retrieval scores in the source language",
    3: "Jenks-optimized BM25 retrieval scores in the source language",
    2: "Jenks-optimized BM25 retrieval scores in the source language",
    1: "Jenks-optimized BM25 retrieval scores in the source language",
    0: "Document not retrieved in the source language",
}

# These two are standalone: nothing is derived from either (no edges=), and
# neither derives from anything -- they're inputs to how every other
# Resource's URL is found, not bytes any node is parsed from. Registered
# directly (not through a Generator: there's exactly one of each).
DOWNLOADS_INDEX = Resource('downloads.json.gz',
    sources=['http://www.cs.jhu.edu/~shuosun/clirmatrix/data/downloads.json.gz'],
    md5='371cc532aca236759bd3602eb6ce2181',
    size=5_143_717,
    desc='The per-file {url, cache_path, expected_md5} index this family '
        'resolves every other download from -- see _fetch_index below.')

METADATA_FILE = Resource('metadata.json.lz4',
    sources=['https://macavaney.us/clirmatrix-metadata.json.lz4'],
    md5='537510770a139b25dd12684c6711c91a',
    size=6_517_585,
    desc='Per-language/pair metadata mirrored from v1; not read by this v2 '
        'port -- see the module docstring.')

clirmatrix.register(DOWNLOADS_INDEX, METADATA_FILE)


def _fetch_index():
    """The remote per-file {url, cache_path, expected_md5} index -- one small
    gzip'd JSON (~150k entries), fetched and cached once."""
    with DOWNLOADS_INDEX.gunzip().stream() as f:
        return json.load(f)


_index = Lazy(_fetch_index)


def _resource(name, dlc_context, key):
    entry = _index()[dlc_context][key]
    return Resource(name, sources=[entry['url']], md5=entry.get('expected_md5'))


class _ClirMatrixQueriesParser(Parser):
    name = 'CLIRMatrixQueries'

    def __init__(self, query_lang):
        self.query_lang = query_lang

    def build(self, source, node):
        return _V1CLIRMatrixQueries(source, self.query_lang)


class _ClirMatrixQrelsParser(Parser):
    name = 'CLIRMatrixQrels'

    def build(self, source, node):
        return _V1CLIRMatrixQrels(source, node.defs or {})


#: lang -> the one download backing that language's Wikipedia corpus --
#: memoized separately from the DocTable that wraps it (below), so the
#: Resource has one stable identity whether it's reached via the docs table's
#: own resolution or resolved directly by its own generator (see the
#: `RESOURCE`-typed generator registered below): same object either way.
_docs_resource_cache = {}


def _docs_resource(lang):
    if lang not in _docs_resource_cache:
        # The remote index's own key shape (`docs/{lang}`) is fixed by the
        # JHU-hosted downloads.json.gz -- unrelated to our v2 naming choice.
        _docs_resource_cache[lang] = _resource(
            f'{lang}-docs.txt.gz', 'clirmatrix_docs', f'docs/{lang}')
    return _docs_resource_cache[lang]


def _docs_resource_row(lang):
    """``Generator(row_metadata=...)`` for the docs Resource generator below
    -- real sources/hashes for a listing (``discover_edges()``), not just a
    bare type row, without running the docs Table's own resolver (a parser,
    the TsvDocs wrapper, ...). Still cheap: ``_docs_resource`` only ever
    touches ``_index()`` (fetched once, cached by ``Lazy``) plus a dict
    lookup, same as if this name were actually resolved."""
    return _docs_resource(lang).metadata


#: lang -> the one DocTable for that language's Wikipedia corpus, shared by
#: every bilingual pair using it as the doc side (v1's own ``_docs_cache``).
_docs_cache = {}


def _docs(lang):
    if lang not in _docs_cache:
        name = f'{lang}-docs'
        resource = _docs_resource(lang)
        _docs_cache[lang] = TsvDocs(name, source=resource.gunzip(), lang=lang)
    return _docs_cache[lang]


#: (variant, doc_lang, query_lang, split) -> the one downloaded gzip'd JSONL
#: file backing both the queries and qrels tables for that pair+split --
#: memoized separately from ``_benchmark`` for the same reason
#: ``_docs_resource`` is: a stable identity whether reached via the
#: benchmark's own resolution or this Resource's own generator, below.
_qrel_resource_cache = {}


def _qrel_resource(variant, doc_lang, query_lang, split):
    key = (variant, doc_lang, query_lang, split)
    if key not in _qrel_resource_cache:
        # The remote index's own key shape (`queries/{query_lang}_{doc_lang}/
        # {split}`) is fixed by the JHU-hosted downloads.json.gz -- unrelated
        # to our v2 naming choice.
        name = f'{variant}-{doc_lang}-{query_lang}-{split}.jsonl.gz'
        _qrel_resource_cache[key] = _resource(
            name, _DLC_CONTEXT[variant], f'queries/{query_lang}_{doc_lang}/{split}')
    return _qrel_resource_cache[key]


def _qrel_resource_row(variant, doc_lang, query_lang, split):
    """``row_metadata=`` for the shared queries+qrels Resource generator --
    see ``_docs_resource_row``."""
    return _qrel_resource(variant, doc_lang, query_lang, split).metadata


#: (doc_lang, variant, query_lang, split) -> the one Benchmark for it.
_benchmark_cache = {}


def _bilingual_constraint(variant, doc_lang, query_lang, split):
    """CLIR pairs a doc language against a *different* query language --
    there's no same-language ``en-en`` split in any variant (verified
    against the real ``downloads.json.gz`` index: every ``dlc_context`` has
    exactly ``n * (n - 1) * len(SPLITS)`` entries, never ``n * n * ...``).
    Without this, ``_BILINGUAL_PARAMS``'s cross product overclaims ~1,100
    same-language names per variant that ``_resource``'s index lookup would
    ``KeyError`` on the first time anything actually resolves one -- row
    metadata during enumeration (see ``_qrel_resource_row``) now does that
    for every enumerated name, not just a name someone happens to load."""
    if variant == 'multi8' and not (
        doc_lang in MULTI8_LANGS and query_lang in MULTI8_LANGS):
        return False
    return doc_lang != query_lang


def _benchmark(variant, doc_lang, query_lang, split):
    key = (variant, doc_lang, query_lang, split)
    if key not in _benchmark_cache:
        name = f'{variant}-{doc_lang}-{query_lang}-{split}'
        docs = _docs(doc_lang)
        # Queries and qrels for one pair+split are the SAME downloaded
        # gzip'd JSONL file (one query per line, its judged docs inline) --
        # shared here exactly as v1 shares one ``qrel_dlc`` object between
        # its two handler classes.
        stream = _qrel_resource(variant, doc_lang, query_lang, split).gunzip()
        queries = QueryTable(f'{name}-queries', source=stream,
                             parser=_ClirMatrixQueriesParser(query_lang), lang=query_lang)
        qrels = QrelTable(f'{name}-qrels', source=stream,
                          parser=_ClirMatrixQrelsParser(), defs=QRELS_DEFS)
        _benchmark_cache[key] = Benchmark(
            name, docs=docs, queries=queries, qrels=qrels, citation=CITATION,
            desc=f'CLIRMatrix {variant}: {doc_lang} docs, {query_lang} queries, '
                 f'{split} split. Qrels are automatic (BM25-derived), not '
                 f'human judgments -- see the qrels table\'s relevance defs.')
    return _benchmark_cache[key]


def _queries(variant, doc_lang, query_lang, split):
    """The queries table on its own -- resolves (and shares) the whole
    Benchmark, same as v1's own facet accessors; this just gives the table
    its own addressable/generator identity (``...{split}-queries``)."""
    return _benchmark(variant, doc_lang, query_lang, split).queries


def _qrels(variant, doc_lang, query_lang, split):
    """The qrels table on its own -- see ``_queries``."""
    return _benchmark(variant, doc_lang, query_lang, split).qrels


clirmatrix.register_generator(Generator(
    '{lang}-docs.txt.gz', params={'lang': Param(values=LANGS)},
    type=RESOURCE, resolver=_docs_resource, enumerable=True,
    row_metadata=_docs_resource_row,
    desc='The downloaded Wikipedia-derived corpus file backing one '
        'language\'s docs table.'))

clirmatrix.register_generator(Generator(
    '{lang}-docs', params={'lang': Param(values=LANGS)},
    type=TABLE_TYPES['docs'], resolver=_docs, enumerable=True,
    edges=[(DERIVED_FROM, '{lang}-docs.txt.gz')],
    desc='One language\'s Wikipedia-derived document corpus.'))

#: Shared by every bilingual generator below -- queries/qrels/the shared
#: resource/the benchmark itself are all keyed on the same four params.
_BILINGUAL_PARAMS = {
    'variant': Param(values=VARIANTS), 'doc_lang': Param(values=LANGS),
    'query_lang': Param(values=LANGS), 'split': Param(values=SPLITS),
}

clirmatrix.register_generator(Generator(
    '{variant}-{doc_lang}-{query_lang}-{split}.jsonl.gz',
    params=_BILINGUAL_PARAMS, constraints=[_bilingual_constraint],
    type=RESOURCE, resolver=_qrel_resource, enumerable=True,
    row_metadata=_qrel_resource_row,
    desc='The downloaded file backing one pair+split\'s queries and qrels '
        '(one query per line, its judged docs inline) -- shared by both '
        'tables.'))

clirmatrix.register_generator(Generator(
    '{variant}-{doc_lang}-{query_lang}-{split}-queries',
    params=_BILINGUAL_PARAMS, constraints=[_bilingual_constraint],
    type=TABLE_TYPES['queries'], resolver=_queries, enumerable=True,
    edges=[(DERIVED_FROM, '{variant}-{doc_lang}-{query_lang}-{split}.jsonl.gz')],
    desc='The queries for one bilingual pair+split.'))

clirmatrix.register_generator(Generator(
    '{variant}-{doc_lang}-{query_lang}-{split}-qrels',
    params=_BILINGUAL_PARAMS, constraints=[_bilingual_constraint],
    type=TABLE_TYPES['qrels'], resolver=_qrels, enumerable=True,
    edges=[(DERIVED_FROM, '{variant}-{doc_lang}-{query_lang}-{split}.jsonl.gz')],
    desc='The (automatic, BM25-derived) qrels for one bilingual pair+split.'))

clirmatrix.register_generator(Generator(
    '{variant}-{doc_lang}-{query_lang}-{split}',
    params=_BILINGUAL_PARAMS, constraints=[_bilingual_constraint],
    type=BENCHMARK, resolver=_benchmark, enumerable=True,
    edges=[(FACET['docs'], '{doc_lang}-docs'),
          (FACET['queries'], '{variant}-{doc_lang}-{query_lang}-{split}-queries'),
          (FACET['qrels'], '{variant}-{doc_lang}-{query_lang}-{split}-qrels')],
    desc='A bilingual CLIR benchmark: one doc language, one query language, '
        'one split, at the bi139base/bi139full/multi8 size.'))
