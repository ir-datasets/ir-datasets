"""``hf:`` -- a dynamic provider over HuggingFace Hub dataset repos.

Every family elsewhere in this catalog is a *static* provider: a manifest
frozen ahead of time by a hand-authored module. HF Hub datasets can't be
enumerated or frozen that way (hundreds of thousands of them, growing, with
their own metadata living on the Hub) -- this is the "dynamic provider" the
package docstring anticipates, resolving nodes on demand via ``Generator``
instead of importing a per-family module.

Naming::

    hf:<owner>/<name>                the repo's top-level node (the card entry
                                     with no ``fragment``)
    hf:<owner>/<name>/<fragment>     any other card entry, by its ``fragment``
    hf:<owner>/<name>.git            the underlying HfDataset resource

The card
--------

Where the card comes from
-------------------------

Two places, tried in this order:

1. **The registry:** a file in this package, ``hf_cards/<owner>/<name>.yaml``. It is shaped
   like README frontmatter -- an ``ir-datasets`` key (below), plus whatever standard
   fields the card should supply (``license``, ``language``, ``pretty_name``) -- and may
   carry a ``commit:`` to pin the data to. It makes a dataset usable without its
   authors changing anything, is reviewed and tested here like any other node, and is
   self-contained: nothing is fetched to read it. To try a card, put it there (in a
   development checkout) and load the name.
2. **The repo itself:** the same block in its ``README.md`` frontmatter.

A repo with a registry card needs no tag on the Hub; ``known()`` lists the registry
alongside the tagged repos.

The card
--------

An ``ir-datasets`` (or ``ir_datasets``) key holds a list of node declarations, or -- as
shorthand for a one-item list -- a single mapping::

    ir-datasets:
      - fragment: null            # the top-level node, hf:owner/name
        type: irds:Benchmark
        docs: hf:other/corpus     # a fragment of this card, or any qualified name
        queries: queries
        qrels: qrels
        metrics: [nDCG@10]
        description: ...
      - fragment: queries
        type: irds:Queries
        file: data/queries.jsonl  # or config:/split: for datasets.load_dataset
        column_map: {_id: query_id, junk: null}

Each entry has a ``fragment`` (its name under the repo; absent or ``null`` is
the top-level node) and a ``type`` (a fully qualified graph class:
``irds:Docs``, ``irds:Queries``, ``irds:Qrels``, ``irds:Overview``, ``irds:Bundle``, ``irds:Benchmark`` or one of its
task types, ``irds:AdhocBenchmark`` -- docs, queries and qrels -- and
``irds:QaBenchmark`` -- queries and answers; a task type is checked against the
facets given). Nothing else is required, and nothing is inferred --
a repo with no top-level entry simply has no bare name.

A table entry says where its rows are: ``file:`` (a path, or a list, relative to
the repo root, read directly off the ``HfDataset``; parquet, jsonl, csv and tsv
by extension), or -- the default -- ``datasets.load_dataset(repo, name=config,
split=split, revision=commit)`` with ``config:``/``split:`` (absent, they are
taken from the fragment: ``train`` is split ``train``, ``en/train`` is config
``en``, split ``train``). ``column_map:`` is ``{source_column: field}``, ``null``
to drop a column; a source column it doesn't mention passes through under its
own name, so a typical map names only the exceptions (``_apply_column_map``).
Common alternative names are mapped for you, by the table's type: a docs table's
``docno``/``docid``/``_id``/``id`` is its ``doc_id``; a queries table's ``qid``/
``id``/... its ``query_id`` and ``query``/``question`` its ``text``; a qrels table's
``qid``/``query-id``, ``docno``/``corpus-id`` and ``label``/``score`` its
``query_id``, ``doc_id`` and ``relevance`` (see ``_COLUMN_ALIASES``). That applies
only where the canonical name isn't already a column, the source column isn't named
in ``column_map``, and exactly one column could be the field (more than one is an
error); ``auto_columns: false`` turns it off.
``defs:`` gives a qrels table's relevance labels.

A bundle entry groups nodes that belong together -- ``parts:`` is a list of fragments
(or fully qualified names), or ``all`` (the default: every other entry except bundles).
``irds:Overview`` is the front page of a dataset: a bundle that must carry a hand-written
``summary:`` (Markdown). Give it ``fragment: null`` and the repo's bare name is a landing
page for the whole dataset. ``irds:Bundle`` is the bare grouping, with a summary only if
you want one.

A benchmark entry names its facets (``docs``/``queries``/``qrels``/...) and
``metrics``. A facet is a fragment of this same card or, if it contains a ``:``,
a fully qualified node from anywhere in the graph (``irds:<dataset>-docs``,
another ``hf:`` repo, this repo's own top level as ``hf:<owner>/<name>``) --
borrowed in place of a local one, resolved lazily the first time it's used.

What the card already says in the standard HF frontmatter is used where an entry
doesn't say otherwise: ``license`` (every node; ``other`` is skipped), ``language``
(a table's ``lang``, when the card names exactly one language) and ``pretty_name``
(the top-level node's ``prettyName``).

Every other key is a property of the node: ``description``, ``summary``, ``prettyName``,
``citation``, ``license``, ``deprecated`` and, for tables, ``lang`` and ``dua``.
A key this provider doesn't know is reported and ignored.

A plain ``/`` rather than ``#``/``~`` separates the fragment: the name is a URL
path segment on the website (``/n/hf:<owner>/<name>/<fragment>``), and ``#`` is
the URL fragment delimiter, so browsers strip anything after it. A HF repo id
is always exactly two slash-free segments, so the first ``/`` separates owner
from name and anything after the second is the fragment. A node is always the
repo's default branch as of the crawl; there is no pinning in the name (the commit
it was read at is recorded on the ``.git`` resource).

No vocabulary of its own: nodes are built from the existing ``DocTable``/
``QueryTable``/``QrelTable``/``Benchmark``/``Resource`` classes (already
declared by the ``irds`` provider), reused exactly as a third party would.
"""
import collections
import re
from pathlib import Path

import ir_datasets
from ir_datasets.lazy_libs import hf_datasets as _hf_datasets_lib
from ir_datasets.lazy_libs import huggingface_hub as _hf_lib
from ir_datasets.lazy_libs import pyarrow_parquet as _pq_lib
from ir_datasets.lazy_libs import yaml as _yaml_lib

from .base import Generator, Param
from .formats import Parser
from .nodes import (
    Benchmark, Bundle, DocTable, Overview, ENTITIES, GitRepo, QrelTable, QueryTable, TABLE,
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


#: Per entity, ``(field, alternative source names)`` in priority order. Names are compared
#: lower-case with ``-`` as ``_``. Deliberately conservative: the id fields every
#: record needs, plus the one text field of a query or a doc where the usual alternatives
#: are unambiguous (a ``title`` or ``abstract`` is never taken for ``text``).
_COLUMN_ALIASES = {
    'docs': (('doc_id', ('docno', 'docid', 'doc_no', 'document_id', 'documentid', '_id', 'id')),
             ('text', ('contents', 'content', 'body', 'passage', 'passage_text'))),
    'queries': (('query_id', ('qid', 'queryid', 'query_no', 'qno', '_id', 'id')),
                ('text', ('query', 'query_text', 'question', 'topic'))),
    'qrels': (('query_id', ('qid', 'queryid', 'query_no', 'qno')),
              ('doc_id', ('docno', 'docid', 'doc_no', 'corpus_id', 'document_id')),
              ('relevance', ('label', 'score', 'rel', 'rating', 'grade'))),
}


def _normalized(column):
    return str(column).lower().replace('-', '_')


def _apply_column_map(source_columns, column_map, entity=None, auto=True):
    """``column_map`` is ``{source_column: field_name}`` (or ``None`` to drop
    that column) -- the reverse direction of a record's fields, since that's
    the direction most tables actually need: rename or drop a handful of
    columns, keep the rest as-is. Any source column *not* mentioned passes
    through under its own name, so an empty/absent ``column_map`` takes every
    column verbatim, and a partial one only needs to name the exceptions.

    Then, for ``entity`` (unless ``auto`` is off), a column passing through under a
    common alternative name takes the entity's own name (``_COLUMN_ALIASES``) --
    but only when the entity's name isn't already a field, and when the column
    wasn't named in ``column_map``: what the card says always wins. If more than one
    column could be the same field, that is an error rather than a guess.

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
    if auto and entity in _COLUMN_ALIASES:
        for target, alternatives in _COLUMN_ALIASES[entity]:
            if target in columns:
                continue
            names = (target, *alternatives)    # the field's own name too, spelled with '-' or in caps
            candidates = [c for c in source_columns
                          if _normalized(c) in names and c not in column_map and columns.get(c) == c]
            if len(candidates) > 1:
                raise ValueError(
                    f'{entity} table: more than one source column could be its {target!r} '
                    f'({", ".join(map(repr, candidates))}); name the right one in column_map '
                    f'(e.g. {{{candidates[0]!r}: {target}}}), or set auto_columns: false')
            if candidates:
                fields[fields.index(candidates[0])] = target
                columns[target] = columns.pop(candidates[0])
    return fields, columns


class HfProvider(ManifestProvider):
    """``discover_edges`` overrides the default entirely: it runs the live Hub
    crawl (``known()``, defined below), which resolves and registers every
    repo it finds into ``self.nodes`` with a real type, then reports exactly
    that -- and nothing else -- as triples. See
    ``known()`` for what "every repo it finds" means and its own caveats."""
    def discover_edges(self):
        from .vocabulary import row_for
        known()
        for name, node in sorted(self.nodes.items()):
            # Whatever the node reports about itself (language, license, ...) -- never
            # the data: a table whose columns need I/O leaves them out (see
            # ``Table.discovery_literals``).
            literals = getattr(node, 'discovery_literals', lambda: {})()
            row = {**row_for(node), **{k: v for k, v in literals.items() if v}}
            yield from row_triples(name, row)
        for src, kind, dst in self.edge_rows():
            yield src, kind, dst


hf = HfProvider('hf')


# ── addressing ────────────────────────────────────────────────────────────

_SPEC = re.compile(r'^(?P<repo>[^/@]+/[^/@]+)(?:/(?P<fragment>.+))?$')


def _parse_spec(spec):
    """``owner/name[/fragment]`` -> (repo, fragment).

    A repo id is always exactly two slash-free segments, so the first ``/``
    unambiguously separates owner from name and a further ``/`` starts the
    fragment (which may itself contain more slashes, e.g. a ``config/split``
    fragment)."""
    m = _SPEC.match(spec)
    if m is None:
        raise ValueError(f'not a valid hf: name: {spec!r} (expected owner/name[/fragment])')
    return m.group('repo'), m.group('fragment')


# ── the dataset card ─────────────────────────────────────────────────────

def _frontmatter(readme_text):
    m = re.match(r'^---\n(.*?)\n---\n', readme_text, re.DOTALL)
    if m is None:
        raise KeyError('no YAML frontmatter in README.md')
    return _yaml_lib().safe_load(m.group(1)) or {}


_card_cache = {}


#: What a card entry's ``type`` may say, always the fully qualified graph class ->
#: (kind, entity). A table entry produces that entity's table class.
_TYPES = {'irds:Docs': ('table', 'docs'), 'irds:Queries': ('table', 'queries'),
          'irds:Qrels': ('table', 'qrels'), 'irds:Benchmark': ('benchmark', None),
          'irds:AdhocBenchmark': ('benchmark', None), 'irds:QaBenchmark': ('benchmark', None),
          'irds:Bundle': ('bundle', None), 'irds:Overview': ('bundle', None)}
#: Keys that are properties of the node (card key -> constructor argument).
_PROPERTY_KEYS = {'description': 'desc', 'prettyName': 'pretty_name', 'summary': 'summary',
                  'citation': 'citation',
                  'license': 'license', 'deprecated': 'deprecated'}
_TABLE_PROPERTY_KEYS = {'lang': 'lang', 'dua': 'dua'}
_TABLE_KEYS = {'file', 'config', 'split', 'column_map', 'auto_columns', 'defs'}
_BENCHMARK_KEYS = set(ENTITIES) | {'metrics'}
_BUNDLE_KEYS = {'parts'}


def _type_of(repo, entry):
    """``(kind, entity)`` for an entry's ``type``, which must be a fully qualified
    graph class."""
    given = entry.get('type')
    if given not in _TYPES:
        raise ValueError(f'{repo}: {entry.get("fragment")!r} has type {given!r}; expected one of '
                         f'{", ".join(_TYPES)}')
    return _TYPES[given]


def _entries_of(repo, block):
    """``{fragment: entry}`` from a card block -- a list of entries, or one mapping
    standing for a one-item list. The top-level node's fragment is ``None``."""
    items = [block] if isinstance(block, dict) else block
    if not isinstance(items, list) or not items:
        raise ValueError(f'{repo}: the ir-datasets block must be a mapping or a non-empty list')
    entries = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f'{repo}: each ir-datasets entry must be a mapping, got {item!r}')
        fragment = item.get('fragment')
        if fragment is not None:
            fragment = str(fragment)
            if not fragment or fragment.startswith('/') or fragment.endswith('/'):
                raise ValueError(f'{repo}: invalid fragment {fragment!r}')
        if fragment in entries:
            raise ValueError(f'{repo}: fragment {fragment!r} is declared twice')
        _type_of(repo, item)
        entries[fragment] = item
    return entries


def _card_defaults(front):
    """Node properties the card's own standard frontmatter already supplies."""
    out = {}
    license = front.get('license')
    licenses = [x for x in (license if isinstance(license, list) else [license])
                if isinstance(x, str) and x.lower() not in ('other', 'unknown')]
    if licenses:
        out['license'] = licenses[0] if len(licenses) == 1 else licenses
    language = front.get('language')
    language = [language] if isinstance(language, str) else language
    if isinstance(language, list) and len(language) == 1 and isinstance(language[0], str):
        out['lang'] = language[0]
    if isinstance(front.get('pretty_name'), str):
        out['pretty_name'] = front['pretty_name']
    return out


#: Cards shipped with ir_datasets, one file per repo: ``<owner>/<name>.yaml``.
CARD_DIR = Path(__file__).parent / 'hf_cards'


def _registry_front(repo):
    """The registry's card for ``repo`` (a README-frontmatter-shaped mapping), or ``None``."""
    path = CARD_DIR / f'{repo}.yaml'
    if not path.is_file():
        return None
    front = _yaml_lib().safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(front, dict):
        raise ValueError(f'{path}: a registry card must be a mapping with an ir-datasets key')
    return front


def _registry_repos():
    """Every repo the registry has a card for."""
    if not CARD_DIR.is_dir():
        return []
    return sorted(f'{path.parent.name}/{path.stem}' for path in CARD_DIR.glob('*/*.yaml'))


def _card_of_front(repo, front):
    present = [k for k in CARD_KEYS if k in front]
    if len(present) > 1:
        raise ValueError(
            f'{repo}: card declares both {" and ".join(CARD_KEYS)} -- pick one')
    if not present:
        raise KeyError(f'{repo}: no {"/".join(CARD_KEYS)} block in its card '
                       f'-- not annotated for ir_datasets')
    return {'tags': front.get('tags') or [], 'meta': _card_defaults(front),
            'entries': _entries_of(repo, front[present[0]])}


def _card(repo, commit):
    """This repo's parsed ``ir-datasets`` card: the registry's if it has one, else the
    one in the repo's README at a resolved commit --
    ``{'tags': [...], 'meta': {...}, 'entries': {fragment: entry}}``. Raises KeyError if
    the repo isn't annotated, ValueError if the card is malformed -- both become "no
    such node" at ``Provider.__getitem__``."""
    key = (repo, commit)
    if key not in _card_cache:
        front = _registry_front(repo)
        if front is None:
            hub = _hf_lib()
            path = hub.hf_hub_download(repo_id=repo, repo_type='dataset',
                                       filename='README.md', revision=commit)
            with open(path, encoding='utf-8') as fin:
                front = _frontmatter(fin.read())
        _card_cache[key] = _card_of_front(repo, front)
    return _card_cache[key]


def _resolve_commit(repo):
    """The commit the repo's default branch is at."""
    hub = _hf_lib()
    return hub.HfApi().dataset_info(repo).sha


# ── HfDataset: the whole repo tree at a pinned commit, not a single file ────

class HfDataset(GitRepo):
    """A HuggingFace Hub dataset repo, as a whole, at a pinned commit --
    ``GitRepo`` (pinned-commit identity and ``.file(path)`` access) plus the one thing that's actually
    HF-specific: how the tree is fetched.

    ``path()`` is fetched via ``huggingface_hub.snapshot_download`` rather
    than a literal ``git``/``git-lfs`` clone: same logical operation, no
    extra system dependency, no reimplementing LFS pointer resolution -- the
    ``.git``-suffixed name (from ``GitRepo``) still reflects what it is (this
    really is the repo's git tree), not how every read path happens to fetch
    it.
    """
    @property
    def requires(self):
        return ('huggingface_hub', 'datasets')

    def url(self):
        return f'https://huggingface.co/datasets/{self.repo}'

    def path(self, force=True):
        hub = _hf_lib()
        return hub.snapshot_download(repo_id=self.repo, repo_type='dataset',
                                     revision=self.commit)

    def __repr__(self):
        return f'HfDataset({self.url()!r}, commit={self.commit!r})'


_repo_cache = {}


def _hf_dataset(repo):
    """The one ``HfDataset`` for ``repo`` -- shared across every fragment resolved
    off the same repo within this process, so a Benchmark's several facets (and
    repeated single-table lookups) all point at the same node instead of
    independent, differently-timed resolutions."""
    if repo not in _repo_cache:
        front = _registry_front(repo) or {}
        pinned = front.get('commit')      # a registry card may pin the data it was written against
        if pinned is not None and not (isinstance(pinned, str) and re.fullmatch(r'[0-9a-f]{40}', pinned)):
            raise ValueError(f'{repo}: commit must be a full 40-character commit hash, got {pinned!r}')
        _repo_cache[repo] = HfDataset(repo, commit=pinned or _resolve_commit(repo),
                                      pinned=pinned is not None)
    return _repo_cache[repo]


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
    columns_need_io = True      # the column set comes from the data itself
    requires = ('huggingface_hub', 'datasets')

    def __init__(self, repo, commit, config, split, entity, column_map, defs=None, auto_columns=True):
        self.repo = repo
        self.commit = commit
        self.config = config
        self.split = split
        self.entity = entity
        self.column_map = column_map
        self.auto_columns = auto_columns
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
            fields, self.columns = _apply_column_map(self._load().column_names, self.column_map,
                                                     self.entity, self.auto_columns)
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
    columns_need_io = True
    requires = ('huggingface_hub', 'datasets')

    def __init__(self, entity, column_map, defs=None, auto_columns=True):
        self.entity = entity
        self.column_map = column_map
        self.auto_columns = auto_columns
        self.defs = defs
        self.columns = None
        self._resolved = None

    def _resolve(self, source):
        """(record_cls, columns), computed and cached on first use."""
        if self._resolved is None:
            first_source = source[0] if isinstance(source, list) else source
            first_row = next(iter(_rows_of(first_source.path())))
            fields, self.columns = _apply_column_map(list(first_row.keys()), self.column_map,
                                                     self.entity, self.auto_columns)
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


def _unknown_keys(repo, entry, allowed):
    extra = sorted(set(entry) - {'fragment', 'type'} - allowed)
    if extra:
        _logger.warn(f'{repo}: {entry.get("fragment")!r}: ignoring unknown key(s) {extra}')


def _properties(entry, table, card_meta, top_level, entity=None):
    """The node's constructor arguments from its entry; the card's own standard
    metadata fills in what the entry leaves out."""
    keys = {**_PROPERTY_KEYS, **(_TABLE_PROPERTY_KEYS if table else {})}
    props = {arg: entry[key] for key, arg in keys.items() if key in entry}
    for arg in ('license', 'lang', 'pretty_name'):
        # a language is the text's: the card's goes to docs and queries, not to qrels
        if arg in card_meta and arg not in props and (arg != 'lang' or entity in ('docs', 'queries')) \
                and (arg != 'pretty_name' or top_level):
            props[arg] = card_meta[arg]
    return props


def _build_table(name, repo, entity, hf_dataset, fragment, entry, card_meta):
    cls = _TABLE_CLASSES[entity]
    _unknown_keys(repo, entry, _TABLE_KEYS | set(_PROPERTY_KEYS) | set(_TABLE_PROPERTY_KEYS))
    # {source_column: field_name} (None to drop); anything unmentioned passes
    # through under its own name. Resolved lazily -- see the parsers.
    column_map = entry.get('column_map')
    auto = entry.get('auto_columns', True)
    defs = entry.get('defs')
    file = entry.get('file')
    if file:
        source = ([hf_dataset.file(f) for f in file] if isinstance(file, list)
                  else hf_dataset.file(file))
        parser = _HfFileParser(entity, column_map, defs=defs, auto_columns=auto)
    else:
        # the fragment doubles as "config/split" (or a bare "split", default
        # config) unless overridden explicitly
        config_from_key, _, split_from_key = (fragment or '').rpartition('/')
        split = entry.get('split', split_from_key)
        if not split:
            raise ValueError(f'{repo}: the top-level table needs a file: or a split:')
        config = entry.get('config', config_from_key or None)
        source = hf_dataset
        parser = _HfDatasetsParser(hf_dataset.repo, hf_dataset.commit, config, split,
                                   entity, column_map, defs=defs, auto_columns=auto)
    return cls(name, source=source, parser=parser, defs=defs,
               **_properties(entry, True, card_meta, fragment is None, entity))


def _build_benchmark(name, repo, entry, node_fn, card_meta, top_level):
    """A facet that is a bare fragment is this card's own node (``node_fn``); one
    containing ``:`` is a fully qualified name, passed straight through --
    ``Benchmark`` already resolves a name-string facet lazily against the default
    graph the first time it's accessed, so the referenced repo or provider need
    not be touched (or even installed) until that facet is used."""
    _unknown_keys(repo, entry, _BENCHMARK_KEYS | set(_PROPERTY_KEYS))
    facets = {e: (entry[e] if ':' in str(entry[e]) else node_fn(str(entry[e])))
              for e in ENTITIES if entry.get(e) is not None}
    benchmark = Benchmark(name, metrics=entry.get('metrics'), **facets,
                          **_properties(entry, False, card_meta, top_level))
    # A task type (irds:AdhocBenchmark, irds:QaBenchmark) is an assertion about the
    # facets, checked rather than trusted: the node derives its types from them.
    if entry['type'] != 'irds:Benchmark' and entry['type'] not in benchmark.types:
        needs = {'irds:AdhocBenchmark': 'docs, queries and qrels',
                 'irds:QaBenchmark': 'queries and answers'}[entry['type']]
        raise ValueError(f'{repo}: {entry.get("fragment")!r} is declared {entry["type"]} '
                         f'but does not have {needs}')
    return benchmark


def _build_bundle(name, repo, entry, node_fn, card, top_level):
    """``parts:`` is a list of fragments of this card (or fully qualified names, passed
    straight through as in a benchmark's facets), or ``all`` -- every entry of the card
    except bundles. Cycles between bundles are an error."""
    _unknown_keys(repo, entry, _BUNDLE_KEYS | set(_PROPERTY_KEYS))
    parts = entry.get('parts', 'all')
    if parts == 'all':
        parts = [f for f, e in card['entries'].items()
                 if f != entry.get('fragment') and _type_of(repo, e)[0] != 'bundle']
        if not parts:
            raise ValueError(f'{repo}: {entry.get("fragment")!r} is a bundle of everything, '
                             f'but the card has nothing else')
    elif not isinstance(parts, list):
        raise ValueError(f'{repo}: {entry.get("fragment")!r}: parts must be a list or "all"')
    nodes = [p if ':' in str(p) else node_fn(None if p is None else str(p)) for p in parts]
    cls = Overview if entry['type'] == 'irds:Overview' else Bundle
    return cls(name, parts=nodes, **_properties(entry, False, card['meta'], top_level))


_node_cache = {}


# ── the resolver ──────────────────────────────────────────────────────────

def _declared(card):
    return sorted('(top level)' if f is None else f for f in card['entries'])


def _HfResolver(spec):
    repo, fragment = _parse_spec(spec)
    hf_dataset = _hf_dataset(repo)
    card = _card(repo, hf_dataset.commit)
    building = set()

    def node(frag):
        """The one node for (repo, fragment), built once and shared however
        it's reached -- a benchmark's facet and a direct load are the same object."""
        key = (repo, frag)
        if key not in _node_cache:
            if frag not in card['entries']:
                raise KeyError(f'{repo}: no such fragment {frag!r}; declared: {_declared(card)}')
            entry = card['entries'][frag]
            name = repo if frag is None else f'{repo}/{frag}'
            kind, entity = _type_of(repo, entry)
            if key in building:
                raise ValueError(f'{repo}: bundle parts form a cycle through {frag!r}')
            building.add(key)
            try:
                if kind == 'table':
                    built = _build_table(name, repo, entity, hf_dataset, frag, entry, card['meta'])
                elif kind == 'bundle':
                    built = _build_bundle(name, repo, entry, node, card, frag is None)
                else:
                    built = _build_benchmark(name, repo, entry, node, card['meta'], frag is None)
            finally:
                building.discard(key)
            _node_cache[key] = built
        return _node_cache[key]

    if fragment is None and None not in card['entries']:
        raise KeyError(f'{repo}: no top-level entry (one with no fragment); use a '
                       f'fragment: {_declared(card)}')
    return node(fragment)


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
    """Every bare hf: spec ``repo`` actually addresses -- its top level (if it has
    one) and every fragment its card declares -- each one *resolved*, not just
    named, as a side effect registering it into the graph with a real type. ``[]``
    if the repo turns out not to carry a valid card, or its card fails to resolve;
    see ``known``."""
    try:
        card = _card(repo, _hf_dataset(repo).commit)
    except (KeyError, ValueError):
        return []
    except Exception:
        _logger.warn(f'{repo}: tagged {TAG!r} but failed to expand; skipping')
        return []
    out = []
    for fragment in card['entries']:
        spec = repo if fragment is None else f'{repo}/{fragment}'
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
    """Every repo with a registry card, and every repo tagged ``ir-datasets`` on the Hub,
    expanded into every name
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
    for repo in dict.fromkeys([*_registry_repos(), *search(None)]):   # the registry's, then the tagged
        names.extend(_expand(repo))
    return names
