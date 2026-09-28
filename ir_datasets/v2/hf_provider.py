"""``hf:`` -- a dynamic provider over HuggingFace Hub dataset repos.

Every family elsewhere in this catalog is a *static* provider: a manifest
frozen ahead of time by a hand-authored module. HF Hub datasets can't be
enumerated or frozen that way (hundreds of thousands of them, growing, with
their own metadata living on the Hub) -- this is the "dynamic provider" the
package docstring anticipates, resolving nodes on demand via ``Generator``
instead of importing a per-family module.

Naming (see the design plan for the full reasoning)::

    hf:<owner>/<name>                bare repo -- whatever `default:` names,
                                     or a KeyError asking for a fragment
    hf:<owner>/<name>@<revision>     pinned to a branch/tag/commit
    hf:<owner>/<name>/<fragment>     one table (a ``tables:`` key), a
                                     benchmark (a ``benchmarks:`` key), or
                                     (when there's exactly one benchmark) one
                                     of its facets: docs/queries/qrels
    hf:<owner>/<name>@<rev>/<frag>   both at once
    hf:<owner>/<name>.git            the underlying HfDataset resource

A repo opts in by carrying an ``ir_datasets:`` (or ``ir-datasets:``) block in
its dataset card (``README.md`` YAML frontmatter) declaring, per addressable
table: which entity it is, and a ``column_map:`` -- ``{source_column:
field_name}``, ``null`` to drop a column, and (this is the point of having
the mapping run in this direction) any source column *not* mentioned passes
through under its own name. So an absent/empty ``column_map:`` takes every
column verbatim, and a typical one only needs to name the handful of
exceptions -- a rename or two, maybe a drop -- not the whole schema (see
``_apply_column_map``). By default a table reads via
``datasets.load_dataset(repo, name=config, split=split, revision=commit)`` --
``config``/``split`` come from the table's own key (``"split"`` or
``"config/split"``) unless overridden with explicit ``config:``/``split:``
fields (the key is just an addressable name; it need not match the Hub's own
split name) -- which handles the common, Hub-native case with no
path-guessing. A table may instead declare an explicit ``file:`` path
(relative to the repo root, read directly off the ``HfDataset``) for repos
``datasets.load_dataset`` can't handle -- loading-script-based ones
especially, which recent ``datasets`` releases no longer load at all.

Optionally, a card also declares ``benchmarks:`` -- a mapping of benchmark
name to its facets (``docs``/``queries``/``qrels`` -> a ``tables:`` key, plus
``metrics``), addressed by fragment exactly like tables are
(``hf:<owner>/<name>/<benchmark_key>``); a card may declare several. A facet
need not name a table in this same card: giving a fully qualified name
instead (``hf:<other-owner>/<other-name>/<key>``, ``irds:<dataset>-docs``, or
any other provider's own name) borrows that node in place of one declared
locally -- e.g. a repo publishing only queries+qrels for an existing corpus
can point its ``docs`` facet at that corpus directly, instead of duplicating
it. Resolved the same way ``Benchmark`` resolves any name-string facet: lazily,
against the whole graph, the first time it's actually used -- the referenced
repo/provider is never touched otherwise. A
``default:`` field names what the bare (fragment-less) repo name resolves to
-- a ``tables:`` key, a ``benchmarks:`` key, or (only when the card declares
exactly one benchmark) one of that benchmark's facet names. Nothing is
inferred: no ``default:`` means the bare name is a KeyError, even with
exactly one table or benchmark present -- every repo's "headline" node (if it
has one) is the author's choice, not a guess. The default table/benchmark is
a *direct* identity, not the bare name aliased to a separately-fragmented
one: it's registered as ``hf:<owner>/<name>`` outright, so it has no
``/<default>`` fragment of its own (any other, non-default path to the same
underlying table -- a different raw key, say -- still works).

A plain ``/`` rather than ``#``/``~``: this name is a URL path segment on the
website (``/n/hf:<owner>/<name>/<fragment>``), reading like an ordinary
nested path (``owner/name/fragment``) instead of a special-character suffix.
A HF repo id is always exactly two slash-free segments, so the first ``/``
always separates owner from name and anything after the second is
unambiguously the fragment -- with one caveat: a pinned ``@<revision>`` is
assumed not to itself contain a ``/`` (git ref names technically can; this
naming scheme doesn't support that rare case). ``#`` was rejected outright: it
is the URL *fragment* delimiter, so browsers strip anything after a literal
``#`` before the request ever reaches the server -- a pasted or typed link
would silently lose it.

No vocabulary of its own: nodes are built from the existing ``DocTable``/
``QueryTable``/``QrelTable``/``Benchmark``/``Resource`` classes (already
declared by the ``irds`` provider), reused exactly as a third party would --
see the package docstring's own "a third-party Table is exactly as valid a
node as any of ours."
"""
import collections
import datetime as _dt
import re

import ir_datasets
from ir_datasets.lazy_libs import hf_datasets as _hf_datasets_lib
from ir_datasets.lazy_libs import huggingface_hub as _hf_lib
from ir_datasets.lazy_libs import pyarrow_parquet as _pq_lib
from ir_datasets.lazy_libs import yaml as _yaml_lib

from .base import Generator, Param
from .formats import Parser
from .nodes import (
    Benchmark, DocTable, ENTITIES, GitRepo, QrelTable, QueryTable, TABLE,
)
from .registry import ManifestProvider, row_triples

_logger = ir_datasets.log.easy()

#: The Hub search/tag field that scopes listing -- see the design plan's
#: "Listing / search tag" and its "only sees the default branch" caveat.
TAG = 'ir-datasets'

#: Both spellings accepted (HF's own card keys are snake_case; the project is
#: branded both ways). Exactly one may be present.
CARD_KEYS = ('ir_datasets', 'ir-datasets')

_TABLE_CLASSES = {'docs': DocTable, 'queries': QueryTable, 'qrels': QrelTable}
#: Fields every record of an entity must have, regardless of what else a
#: ``columns:`` mapping adds -- everything downstream (lookup, docstores,
#: qrels dicts) keys off these.
_REQUIRED_FIELDS = {'docs': ('doc_id',), 'queries': ('query_id',),
                    'qrels': ('query_id', 'doc_id', 'relevance')}


def _record_class(entity, fields):
    """A NamedTuple with exactly the declared fields, in declared order --
    not forced into ``Generic*``'s fixed shape, so a card can add columns
    (``title``, ...) beyond the minimum ir_datasets itself requires."""
    missing = [f for f in _REQUIRED_FIELDS[entity] if f not in fields]
    if missing:
        raise ValueError(f'{entity} table needs a column for {", ".join(missing)}; '
                         f'declared: {list(fields)}')
    cls = collections.namedtuple(f'Hf{entity.capitalize()}', fields)
    if 'text' in fields:
        cls.default_text = lambda self: self.text
    return cls


def _apply_column_map(source_columns, column_map):
    """``column_map`` is ``{source_column: field_name}`` (or ``None`` to drop
    that column) -- the reverse direction of a record's fields, since that's
    the direction most tables actually need: rename or drop a handful of
    columns, keep the rest as-is. Any source column *not* mentioned passes
    through under its own name, so an empty/absent ``column_map`` takes every
    column verbatim, and a partial one only needs to name the exceptions.

    Returns ``(fields, columns)`` -- ``fields`` in source-column order,
    ``columns`` the field-name -> source-column dict ``_iter`` reads by (the
    same shape the old ``columns:`` direction used).
    """
    column_map = column_map or {}
    unknown = set(column_map) - set(source_columns)
    if unknown:
        raise ValueError(f'column_map names column(s) not present in the source: '
                         f'{sorted(unknown)}; actual columns: {source_columns}')
    fields, columns = [], {}
    for col in source_columns:
        field = column_map.get(col, col)
        if field is None:
            continue
        fields.append(field)
        columns[field] = col
    return fields, columns


class HfProvider(ManifestProvider):
    """No manifest to freeze from (see the module docstring: a repo can't be
    enumerated or frozen ahead of time) -- ``discover_edges`` overrides the
    manifest-based default entirely: it runs the live Hub crawl (``known()``,
    defined below), which resolves and registers every repo it finds into
    ``self.nodes`` with a real type, then reports exactly that -- and nothing
    else -- as triples, the same shape a frozen manifest would report. See
    ``known()`` for what "every repo it finds" means and its own caveats."""
    def discover_edges(self):
        from .freeze import row_for
        known()
        snapshot_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        for name, node in sorted(self.nodes.items()):
            row = row_for(node)
            row['snapshot_at'] = snapshot_at
            yield from row_triples(name, row)
        for src, kind, dst in self.edge_rows():
            yield src, kind, dst


hf = HfProvider('hf')


# ── addressing ────────────────────────────────────────────────────────────

_SPEC = re.compile(r'^(?P<repo>[^/@]+/[^/@]+)(?:@(?P<revision>[^/]+))?(?:/(?P<fragment>.+))?$')


def _parse_spec(spec):
    """``owner/name[@revision][/fragment]`` -> (repo, revision, fragment).

    A repo id is always exactly two slash-free segments, so the first ``/``
    unambiguously separates owner from name and a further ``/`` starts the
    fragment (which may itself contain more slashes, e.g. a ``config/split``
    compound table key) -- the one thing this can't disambiguate is a pinned
    revision that itself contains a ``/`` (see the module docstring)."""
    m = _SPEC.match(spec)
    if m is None:
        raise ValueError(f'not a valid hf: name: {spec!r} (expected owner/name'
                         f'[@revision][/fragment])')
    return m.group('repo'), m.group('revision'), m.group('fragment')


# ── the dataset card ─────────────────────────────────────────────────────

def _frontmatter(readme_text):
    m = re.match(r'^---\n(.*?)\n---\n', readme_text, re.DOTALL)
    if m is None:
        raise KeyError('no YAML frontmatter in README.md')
    return _yaml_lib().safe_load(m.group(1)) or {}


_card_cache = {}


def _card(repo, commit):
    """This repo's parsed ``ir_datasets:`` card at a resolved commit. Raises
    KeyError if the repo isn't annotated, ValueError if the card is malformed
    -- both become "no such node" at ``Provider.__getitem__``."""
    key = (repo, commit)
    if key not in _card_cache:
        hub = _hf_lib()
        path = hub.hf_hub_download(repo_id=repo, repo_type='dataset',
                                   filename='README.md', revision=commit)
        with open(path, encoding='utf-8') as fin:
            front = _frontmatter(fin.read())
        present = [k for k in CARD_KEYS if k in front]
        if len(present) > 1:
            raise ValueError(
                f'{repo}: card declares both {" and ".join(CARD_KEYS)} -- pick one')
        if not present:
            raise KeyError(f'{repo}: no {"/".join(CARD_KEYS)} block in its card '
                           f'-- not annotated for ir_datasets')
        block = front[present[0]] or {}
        tables = block.get('tables') or {}
        if not tables:
            raise ValueError(f'{repo}: {present[0]} block declares no tables')
        _card_cache[key] = {'tags': front.get('tags') or [], 'tables': tables,
                            'benchmarks': block.get('benchmarks') or {},
                            'default': block.get('default')}
    return _card_cache[key]


def _resolve_commit(repo, revision):
    hub = _hf_lib()
    return hub.HfApi().dataset_info(repo, revision=revision).sha


# ── HfDataset: the whole repo tree at a pinned commit, not a single file ────

class HfDataset(GitRepo):
    """A HuggingFace Hub dataset repo, as a whole, at a pinned commit --
    ``GitRepo`` (pinned-commit identity, commit-based ``attest``/``verify``,
    ``.relative(path)`` member access) plus the one thing that's actually
    HF-specific: how the tree is fetched.

    ``path()`` is fetched via ``huggingface_hub.snapshot_download`` rather
    than a literal ``git``/``git-lfs`` clone: same logical operation, no
    extra system dependency, no reimplementing LFS pointer resolution -- the
    ``.git``-suffixed name (from ``GitRepo``) still reflects what it is (this
    really is the repo's git tree), not how every read path happens to fetch
    it.
    """
    def url(self):
        return f'https://huggingface.co/datasets/{self.repo}'

    def path(self, force=True):
        hub = _hf_lib()
        return hub.snapshot_download(repo_id=self.repo, repo_type='dataset',
                                     revision=self.commit)

    def _resolve_live_commit(self):
        return _resolve_commit(self.repo, None)

    def __repr__(self):
        return f'HfDataset({self.url()!r}, commit={self.commit!r})'


_repo_cache = {}


def _hf_dataset(repo, revision):
    """The one ``HfDataset`` for (repo, requested revision) -- shared across
    every fragment resolved off the same repo within this process, so a
    Benchmark's several facets (and repeated single-table lookups) all point
    at the same node instead of independent, differently-timed resolutions."""
    key = (repo, revision)
    if key not in _repo_cache:
        commit = _resolve_commit(repo, revision)
        _repo_cache[key] = HfDataset(repo, commit=commit, pinned=revision is not None)
    return _repo_cache[key]


# ── generic parquet reading, column-mapped per the card ──────────────────

def _parquet_iter(path):
    pq = _pq_lib()
    with pq.ParquetFile(path) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def _jsonl_iter(path):
    import gzip
    import json
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def _csv_iter(path, delimiter=','):
    import csv
    with open(path, newline='', encoding='utf-8') as fin:
        yield from csv.DictReader(fin, delimiter=delimiter)


def _rows_of(path):
    """Dispatch by extension. Deliberately not format auto-detection by
    content -- a repo's actual layout (loading-script-defined, arbitrary
    upload, or Hub-auto-converted parquet) is too varied to sniff reliably,
    so the card's own ``file:`` declaration is the source of truth for both
    the path and, implicitly via its extension, the format."""
    lower = path.lower()
    if lower.endswith('.parquet'):
        return _parquet_iter(path)
    if lower.endswith(('.jsonl', '.jsonl.gz', '.json', '.json.gz')):
        return _jsonl_iter(path)
    if lower.endswith(('.tsv', '.tsv.gz')):
        return _csv_iter(path, delimiter='\t')
    if lower.endswith(('.csv', '.csv.gz')):
        return _csv_iter(path)
    raise ValueError(f'{path}: unrecognized format (expected .parquet, '
                     f'.jsonl[.gz], .csv[.gz], or .tsv[.gz])')


def _make_handler(node, iter_fn, record_cls_fn, defs):
    """The v1-shaped handler object every ``Table`` needs (``<entity>_iter``/
    ``_cls``, plus ``docs_store``/``qrels_defs`` where relevant) -- shared by
    both read strategies below, which differ only in how ``iter_fn`` is
    produced. ``record_cls_fn`` is a 0-arg callable rather than a plain class
    because, when a table has no ``columns:``, the record type isn't known
    until the source is actually peeked at (see ``_HfFileParser``/
    ``_HfDatasetsParser``'s own ``_resolve``) -- calling it is what triggers
    (and caches) that peek."""
    handler = type('_HfHandler', (), {})()
    setattr(handler, f'{node.entity}_iter', iter_fn)
    setattr(handler, f'{node.entity}_cls', record_cls_fn)
    if node.entity == 'qrels':
        setattr(handler, 'qrels_defs', lambda: defs or {})
    if node.entity == 'docs':
        from ir_datasets.indices import DEFAULT_DOCSTORE_OPTIONS, PickleLz4FullStore
        setattr(handler, 'docs_store', lambda: PickleLz4FullStore(
            path=str(node.docstore_path), init_iter_fn=iter_fn,
            data_cls=record_cls_fn(), lookup_field='doc_id',
            index_fields=['doc_id'], count_hint=node.count_hint,
            options=DEFAULT_DOCSTORE_OPTIONS))
    return handler


class _HfDatasetsParser(Parser):
    """The default: reads via ``datasets.load_dataset(repo, name=config,
    split=split, revision=commit)`` -- handles the common case (Hub-native,
    usually auto-converted-to-parquet) with no path-guessing at all. Not
    universal: as of recent ``datasets`` releases, loading-script-based repos
    raise ``RuntimeError: Dataset scripts are no longer supported`` -- a
    card can override with an explicit ``file:`` entry (``_HfFileParser``)
    for exactly that case.

    ``column_map`` is ``{source_column: field_name}`` (``None`` for a column
    to drop); anything not mentioned passes through under its own name -- see
    ``_apply_column_map``. Always resolved lazily, on first use: knowing the
    *full* field set (to auto-include whatever ``column_map`` doesn't
    mention) needs the dataset's real ``column_names``, which needs loading
    it -- there's no I/O-free eager path here, unlike a table that names
    every field explicitly.
    """
    name = 'HfDatasets'

    def __init__(self, repo, commit, config, split, entity, column_map, defs=None):
        self.repo = repo
        self.commit = commit
        self.config = config
        self.split = split
        self.entity = entity
        self.column_map = column_map
        self.defs = defs
        self.columns = None
        self._resolved = None
        self._dataset = None

    def _load(self):
        if self._dataset is None:
            ds_lib = _hf_datasets_lib()
            kwargs = {'split': self.split, 'revision': self.commit}
            if self.config:
                kwargs['name'] = self.config
            self._dataset = ds_lib.load_dataset(self.repo, **kwargs)
        return self._dataset

    def _resolve(self):
        """(record_cls, columns), computed and cached on first use."""
        if self._resolved is None:
            fields, self.columns = _apply_column_map(self._load().column_names, self.column_map)
            self._resolved = _record_class(self.entity, fields)
        return self._resolved, self.columns

    def _iter(self):
        record_cls, columns = self._resolve()
        for row in self._load():
            yield record_cls(*(row[columns[f]] for f in record_cls._fields))

    def build(self, source, node):
        return _make_handler(node, self._iter, lambda: self._resolve()[0], self.defs)


class _HfFileParser(Parser):
    """The explicit override: reads one file the card points at (``file:``),
    for repos ``datasets.load_dataset`` can't handle (loading-script-based
    ones especially) or whose layout is otherwise easier to just name
    directly. The file path is author-declared, not discovered -- see the
    module docstring.

    ``column_map`` is ``{source_column: field_name}`` (``None`` for a column
    to drop); anything not mentioned passes through under its own name -- see
    ``_apply_column_map``. Always resolved lazily, on first use, same
    reasoning as ``_HfDatasetsParser``: the full column set comes from
    peeking the file's first row.
    """
    name = 'HfFile'

    def __init__(self, entity, column_map, defs=None):
        self.entity = entity
        self.column_map = column_map
        self.defs = defs
        self.columns = None
        self._resolved = None

    def _resolve(self, source):
        """(record_cls, columns), computed and cached on first use."""
        if self._resolved is None:
            first_source = source[0] if isinstance(source, list) else source
            first_row = next(iter(_rows_of(first_source.path())))
            fields, self.columns = _apply_column_map(list(first_row.keys()), self.column_map)
            self._resolved = _record_class(self.entity, fields)
        return self._resolved, self.columns

    def _iter(self, source):
        record_cls, columns = self._resolve(source)
        sources = source if isinstance(source, list) else [source]
        for s in sources:
            for row in _rows_of(s.path()):
                yield record_cls(*(row[columns[f]] for f in record_cls._fields))

    def build(self, source, node):
        return _make_handler(node, lambda: self._iter(source),
                             lambda: self._resolve(source)[0], self.defs)


def _build_table(name, hf_dataset, key, spec):
    entity = spec['entity']
    cls = _TABLE_CLASSES[entity]
    # {source_column: field_name} (None to drop); anything unmentioned passes
    # through under its own name. Resolved lazily -- see the parsers.
    column_map = spec.get('column_map')
    defs = spec.get('defs')
    file = spec.get('file')
    if file:
        source = ([hf_dataset.relative(f) for f in file] if isinstance(file, list)
                 else hf_dataset.relative(file))
        parser = _HfFileParser(entity, column_map, defs=defs)
    else:
        # key doubles as "config/split" (or bare "split", default config)
        # unless overridden explicitly -- same convention as everywhere else
        # a key addresses something Hub-native.
        config_from_key, _, split_from_key = key.rpartition('/')
        config = spec.get('config', config_from_key or None)
        split = spec.get('split', split_from_key)
        source = hf_dataset
        parser = _HfDatasetsParser(hf_dataset.repo, hf_dataset.commit, config, split,
                                   entity, column_map, defs=defs)
    return cls(name, source=source, parser=parser, defs=defs)


_table_cache = {}


def _cached_table(repo, revision, name, hf_dataset, key, card):
    """The one Table for (repo, revision, key), built once and shared however
    it's reached (a raw ``/key`` fragment, or a benchmark facet like
    ``/docs``). Without this, each name path would rebuild an independent
    object. ``name`` is the qualified name it gets -- ordinarily ``{base}/
    {key}``, but ``{base}`` directly (no fragment at all) when ``key`` is the
    card's ``default:``, so the default has one true identity, not a bare
    name aliased to a separate fragmented one."""
    cache_key = (repo, revision, key)
    if cache_key not in _table_cache:
        if key not in card['tables']:
            raise KeyError(f'{repo}: no table {key!r}; declared: '
                           f'{sorted(card["tables"])}')
        _table_cache[cache_key] = _build_table(name, hf_dataset, key, card['tables'][key])
    return _table_cache[cache_key]


_benchmark_cache = {}


def _cached_benchmark(repo, revision, name, key, card, table_fn):
    """The one Benchmark for (repo, revision, key), built once and shared
    however it's reached -- mirrors ``_cached_table``. ``table_fn`` resolves a
    benchmark facet's raw ``tables:`` key into its (cached) Table, for a facet
    that lives in this same card. A facet may instead name a node from
    *elsewhere* in the graph -- another ``hf:`` repo (``hf:other/name/docs``)
    or a different provider entirely (``irds:msmarco-passage``) -- by
    giving its fully qualified name instead of a raw key; a local key never
    contains ``:``, so that's what distinguishes the two. Such a string is
    passed straight through as the facet: ``Benchmark`` already resolves a
    name-string facet lazily, against the default graph, the first time it's
    accessed (see ``Benchmark._lookup``/``.edge``) -- so the referenced repo
    or provider need not be touched (or even installed) until that facet is
    actually used."""
    cache_key = (repo, revision, key)
    if cache_key not in _benchmark_cache:
        if key not in card['benchmarks']:
            raise KeyError(f'{repo}: no benchmark {key!r}; declared: '
                           f'{sorted(card["benchmarks"])}')
        spec = card['benchmarks'][key]
        facets = {e: (spec[e] if ':' in spec[e] else table_fn(spec[e]))
                 for e in ENTITIES if e in spec}
        _benchmark_cache[cache_key] = Benchmark(name, metrics=spec.get('metrics'), **facets)
    return _benchmark_cache[cache_key]


def _sole_benchmark(card):
    """``(key, spec)`` of this card's one benchmark, or ``None`` if it
    declares zero or several -- the facet-name shorthand (``default: docs``,
    ``/docs``) is only unambiguous when there's exactly one to resolve it
    against."""
    benchmarks = card['benchmarks']
    if len(benchmarks) == 1:
        ((key, spec),) = benchmarks.items()
        return key, spec
    return None


def _resolve_default(repo, card):
    """``('table', key)`` or ``('benchmark', key)`` that ``default:`` refers
    to, or ``None`` if there's no default. A facet name (docs/queries/qrels)
    is shorthand for the underlying table key, but only when there's exactly
    one declared benchmark to resolve it against."""
    default = card.get('default')
    if default is None:
        return None
    if default in card['tables']:
        return 'table', default
    if default in card['benchmarks']:
        return 'benchmark', default
    if default in ENTITIES:
        sole = _sole_benchmark(card)
        if sole is not None:
            _, spec = sole
            underlying = spec.get(default)
            if underlying is not None:
                return 'table', underlying
        raise ValueError(
            f'{repo}: default: {default!r} is a facet name, but '
            + ('there is no benchmark to resolve it against' if not card['benchmarks']
               else 'there is more than one benchmark -- name one explicitly'))
    raise ValueError(f'{repo}: default: {default!r} is not a declared table, '
                     f'benchmark, or (single-benchmark) facet name')


# ── the resolver ──────────────────────────────────────────────────────────

def _HfResolver(spec):
    repo, revision, fragment = _parse_spec(spec)
    hf_dataset = _hf_dataset(repo, revision)
    card = _card(repo, hf_dataset.commit)
    base = repo if revision is None else f'{repo}@{revision}'
    default_key = card.get('default')
    # Resolved once, up front: whichever table/benchmark `default:` refers to
    # gets named `base` directly (a direct identity, not the bare name
    # aliased to a fragmented one) -- so it has no separate `/key` fragment.
    default_ref = _resolve_default(repo, card)

    def table(key):
        name = base if default_ref == ('table', key) else f'{base}/{key}'
        return _cached_table(repo, revision, name, hf_dataset, key, card)

    def benchmark(key):
        name = base if default_ref == ('benchmark', key) else f'{base}/{key}'
        return _cached_benchmark(repo, revision, name, key, card, table)

    def resolve_fragment(key):
        """A raw ``tables:`` key, a ``benchmarks:`` key, or (failing both,
        with exactly one benchmark declared) a benchmark facet name -- the
        one rule for both ``/fragment`` addressing and ``default:``."""
        if key == default_key:
            raise KeyError(f'{repo}: {key!r} is the default; load hf:{base} '
                           f'directly instead of hf:{base}/{key}')
        if key in card['tables']:
            return table(key)
        if key in card['benchmarks']:
            return benchmark(key)
        if key in ENTITIES:
            sole = _sole_benchmark(card)
            if sole is not None:
                sole_key, spec = sole
                underlying = spec.get(key)
                if underlying is None:
                    raise KeyError(f'{repo}: benchmark {sole_key!r} has no {key!r} facet')
                # `underlying` is either a local `tables:` key, or (see
                # `_cached_benchmark`) a fully qualified name of a node
                # elsewhere in the graph -- distinguished the same way there:
                # a local key never contains `:`. Either way this is the same
                # node a direct lookup of `underlying` would give -- `table()`
                # and `default_graph()[...]` are themselves memoized (see
                # `_cached_table`/module-level caches), so re-resolving this
                # facet name later just re-derives the same object cheaply,
                # rather than needing its own alias entry.
                if ':' in underlying:
                    from .graph import default_graph
                    node = default_graph()[underlying]
                else:
                    node = table(underlying)
                return node
        raise KeyError(f'{repo}: no such fragment {key!r}; declared tables: '
                       f'{sorted(card["tables"])}'
                       + (f', benchmarks: {sorted(card["benchmarks"])}'
                          if card['benchmarks'] else ''))

    if fragment is not None:
        return resolve_fragment(fragment)

    # The bare (fragment-less) name resolves to whatever `default:` names --
    # nothing is inferred (not "the one table," not "the one benchmark, if
    # there's one"). A card with several tables/benchmarks and no default, or
    # a benchmark the author doesn't want as the headline node, is completely
    # legitimate; it just isn't reachable by the bare name.
    if default_ref is None:
        raise KeyError(f'{repo}: no default: declared; use a fragment '
                       f'(tables: {sorted(card["tables"])}'
                       + (f', benchmarks: {sorted(card["benchmarks"])}'
                          if card['benchmarks'] else '') + ')')
    kind, key = default_ref
    return benchmark(key) if kind == 'benchmark' else table(key)


hf.register_generator(Generator(
    '{spec}', params={'spec': Param(pattern=r'.+')},
    # A generator declares one produced type; ours actually varies between
    # a per-entity DocTable/QueryTable/QrelTable type and irds:Benchmark
    # depending on what's requested. The generic TABLE (irds:Table, the
    # shared parent of all of those) is the honest common description --
    # this field is descriptive metadata only (see
    # registry.ManifestProvider.register_generator), not a runtime check against
    # what resolve_node() actually returns.
    type=TABLE, resolver=_HfResolver, enumerable=False))


def search(query, **kwargs):
    """Live Hub search, scoped to repos tagged ``ir-datasets`` -- a listing
    convenience, never consulted by ``load()``. Returns repo ids (not yet
    resolved into nodes -- resolve the ones you want via ``load('hf:<id>')``).
    """
    hub = _hf_lib()
    # `filter=`, not the deprecated `tags=` -- huggingface_hub warns that
    # `tags` stops being supported in its 1.0.
    return [d.id for d in hub.HfApi().list_datasets(search=query, filter=TAG, **kwargs)]


def _expand(repo):
    """Every bare hf: spec ``repo`` actually addresses -- its bare/default
    identity (if it has one), every ``tables:`` key, and every
    ``benchmarks:`` key -- each one *resolved*, not just named, as a side
    effect registering it into the graph with a real type. ``[]`` if the repo
    turns out not to carry a valid card, or its card fails to resolve; see
    ``known``."""
    try:
        commit = _resolve_commit(repo, None)
        card = _card(repo, commit)
        default_ref = _resolve_default(repo, card)
    except (KeyError, ValueError):
        return []
    except Exception:
        _logger.warn(f'{repo}: tagged {TAG!r} but failed to expand; skipping')
        return []
    specs = set()
    if default_ref is not None:
        specs.add(repo)
    for key in card['tables']:
        specs.add(repo if default_ref == ('table', key) else f'{repo}/{key}')
    for key in card['benchmarks']:
        specs.add(repo if default_ref == ('benchmark', key) else f'{repo}/{key}')
    out = []
    for spec in specs:
        try:
            hf[spec]
        except KeyError:
            continue
        except Exception:
            _logger.warn(f'hf:{spec}: tagged repo failed to resolve; skipping')
            continue
        out.append(spec)
    return out


def known():
    """Every repo tagged ``ir-datasets`` on the Hub, expanded into every name
    it actually addresses -- its bare/default identity (if it has one), every
    ``tables:`` key, and every ``benchmarks:`` key -- not just the repo id.
    Each is actually resolved (and so registered into the graph with a real
    type/entity/fields, edges to its underlying ``.git`` resource, etc.)
    rather than just listed as a string: expanding a repo needs its card
    fetched once regardless (a small README download plus a metadata call,
    both already cached -- see ``_card``/``_hf_dataset``), and enumerating
    every name it declares from that same card costs no further Hub calls --
    none of it touches the dataset's actual content (``_HfDatasetsParser``/
    ``_HfFileParser`` stay lazy, so no download of the data itself happens
    here). A tagged repo that turns out not to carry a valid card, or one
    whose card fails to resolve, is skipped, not an error -- the tag is a
    hint, not a guarantee. Called by ``HfProvider.discover_edges`` (always --
    there is no opt-in flag) so a listing shows real types/subtypes without
    every page having to be visited first by hand."""
    names = []
    for repo in search(None):
        names.extend(_expand(repo))
    return names
