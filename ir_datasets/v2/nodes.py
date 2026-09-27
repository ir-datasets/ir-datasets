"""ir_datasets' node types.

Every node has a name and is directly loadable::

    load('irds:antique-docs')             -> docs table
    load('irds:antique-test-qrels')       -> qrels table
    load('irds:antique-test')             -> benchmark

Nodes are single-type. A benchmark does not *become* docs by wearing a second
type-hat; it has a ``docs`` edge pointing at a docs table. Facets are therefore
always unambiguous, and a shared corpus is shared by reference.

Node kinds defined here -- nine, matching the paper's scope:

    Resource    bytes + where to get them (+ integrity/access metadata)
    Table       one kind of record (docs/queries/qrels/scoreddocs/docpairs),
               parsed from a source; ``.entity`` names which kind
      DocTable, QueryTable, QrelTable, RunTable, DocPairTable
                   -- Table's five per-entity subtypes (``TABLE_TYPES``); each
                   is its own declared graph type (a subtype of ``Table``, not
                   just ``.entity`` alone), so a type-filtered lookup can ask
                   for the specific kind or the whole family
    Benchmark   docs + queries + qrels (etc.) bundled into an evaluable task,
               plus flat metadata (``citation``, ``metrics``)
    Suite       a named, structural set of Benchmarks (e.g. BEIR)

Citations as papers-with-edges and metrics as measures-with-edges are future
work for a broader knowledge graph; for now both are plain metadata.

``Node``, ``Edge``, ``Generator`` and the registration/graph machinery are
generic (``base.py``, ``registry.py``, ``graph.py``); this module is where the
dataset vocabulary actually gets declared: node types, edge kinds, defaultable
fields, and what an attestation of a table is.
"""
import collections
import contextlib
import hashlib
import itertools
import json
import warnings

import ir_datasets
from .base import Edge, Node
from .context import default
from .verify import Divergence

from .provider import irds
from .sources import (
    DOCSTORE_FORMAT, Readable, as_source, build_download, default_cache_path,
    legacy_path, local_copy_hint, materialize, migrate_legacy,
)

_logger = ir_datasets.log.easy()


def _deprecated(old, new):
    """A v1-style method (``docs_iter()``) that a property-style equivalent
    (``.docs``) has superseded. Warns rather than removes -- existing code
    keeps working -- one level up from the caller of the deprecated method
    itself (``stacklevel=3``: this frame, the method that called us, then the
    caller that should see the warning point at its own line)."""
    warnings.warn(
        f'{old}() is deprecated; use .{new} instead (see the Table/Benchmark '
        f'docstrings for the property-style API)',
        DeprecationWarning, stacklevel=3)


# ── Vocabulary (owned by the irds provider) ──────────────────────────────────

ENTITIES = ('docs', 'queries', 'qrels', 'scoreddocs', 'docpairs')

#: Node types. Qualified (``irds:DocTable``) because the provider declares them --
#: capitalized, RDF/OWL-convention-style, matching the Python class name
#: (a type is class-like); edge kinds below stay lowercase (property-like).
#: ``TABLE`` is the shared parent of the five per-entity table types declared
#: below it (``irds:DocTable``, ``irds:QueryTable``, ...) rather than every
#: Table subclass sharing it *as their own* type -- each entity is its own
#: first-class type (``list(type='irds:QrelTable')`` finds exactly qrels
#: tables), while ``list(type='irds:Table')`` still finds all of them, by
#: walking the declared ``parent`` chain (see ``vocabulary.is_subtype``). This
#: also keeps the hierarchy open to extension: a third-party provider may
#: declare its own ``ext:MyTable`` with ``parent=TABLE`` (a new entity kind
#: this module never enumerated) or register a bare ``Table`` instance
#: directly under the generic ``irds:Table`` type.
RESOURCE = irds.node_type('Resource', desc='bytes (a file, or a directory tree), and where to get them')
TABLE = irds.node_type('Table', desc='a structured set of records (docs/queries/qrels/...); '
                                     'the shared parent of the per-entity table types below')
BENCHMARK = irds.node_type('Benchmark', desc='an evaluable task: tables + citation + metrics')
SUITE = irds.node_type('Suite', desc='a named set of related benchmarks (e.g. BEIR)')

#: One declared node type per entity, each a subtype of ``TABLE`` -- what
#: ``DocTable``/``QueryTable``/``QrelTable``/``RunTable``/``DocPairTable``
#: (and their format subclasses, e.g. ``TsvDocs``) set as their class ``type``.
#: Keyed by the same entity strings as ``ENTITIES`` so ``TABLE_TYPES[entity]``
#: is the one place that mapping lives (``DerivedTable`` also uses it, to pick
#: its type from a runtime ``entity=`` argument rather than a class attribute).
TABLE_TYPES = {
    'docs': irds.node_type('DocTable', parent=TABLE, desc='A Table representing a collection of documents (a corpus). Each row has a doc_id field.'),
    'queries': irds.node_type('QueryTable', parent=TABLE, desc='A Table representing a collection of queries (topis). Each row has a query_id field.'),
    'qrels': irds.node_type('QrelTable', parent=TABLE, desc='A Table representing a collection of relevance assessments (qrels). Each row has query_id, doc_id, and relevnace fields.'),
    'scoreddocs': irds.node_type('RunTable', parent=TABLE, desc='A Table representing a collection of retrieved and scored documents for a set of queries (a run). Each row has query_id, doc_id, and score fields.'),
    'docpairs': irds.node_type('DocPairTable', parent=TABLE, desc='A Table representing paired documents for a query (e.g. positive/negative pairs for training). Each row has query_id, doc_id_a, and doc_id_b fields.'),
}

#: Structural edges form a DAG and are what builds, caching and verification
#: follow. Keeping them separate from informational edges is what stops a
#: "see also" link from creating a cycle in the build graph.
FACET = {e: irds.edge_kind(e, structural=True, desc='benchmark -> table providing that facet')
         for e in ENTITIES}
#: What a node is built from -- covers "table parsed from these bytes"
#: (formerly a separate ``source`` kind), "table filtered from this other
#: table", and "derived table's filter got its id set from this node" (a
#: single ``filtered_by`` kind used to carve the last case out; dropped --
#: it was always paired with a derived_from to the same target, so it was
#: redundant, and reverse-edge browsing already tells you why a table
#: depends on something by looking at what it is). All are the same shape
#: of relationship (this node would not exist without that one), just with
#: the far end being a Resource or another Table -- one kind, not several.
DERIVED_FROM = irds.edge_kind('derived_from', structural=True,
                              desc='what a node is built from: bytes it is '
                                   'parsed from, another table it is filtered '
                                   'from, or a node its filter took ids from')
#: Suite membership. Not ``derived_from``: like a Benchmark (a bundle of facet
#: edges), a Suite is a bundle, not a derived thing. One kind whether the
#: member is a Benchmark or a nested Suite (e.g. BEIR nests CQADupStack) --
#: the member's own ``.type`` already says which, so a second kind would only
#: duplicate it.
SUITE_MEMBER = irds.edge_kind('member', structural=True,
                              desc='suite -> a benchmark or nested suite it contains')
STRUCTURAL_EDGES = (*FACET.values(), DERIVED_FROM, SUITE_MEMBER)

#: Fields ``irds.defaults()`` may set. Descriptive only -- never identity or
#: data (``name``, ``source``, ``md5``, ``defs``, ...).
DEFAULTABLE = irds.defaultable('dua', 'lang', 'deprecated')


# ── Attestation of tables ────────────────────────────────────────────────────
# What ``freeze --verify`` records about a table and what ``verify`` compares.
# This used to be a hand-written integration test per family.

#: Bump when the canonicalization below changes, so old attestations are known
#: to be incomparable rather than mysteriously mismatched.
HASH_SCHEME = 'v1'

DEFAULT_SAMPLES = (0, 9, -1)


def _canonical(record):
    """Stable bytes for one record. Field *names* are included so a schema
    change is visible as a hash change rather than silently comparing tuples."""
    data = dict(record._asdict()) if hasattr(record, '_asdict') else {'value': record}
    return json.dumps(data, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), default=str).encode()


def _schema(node):
    try:
        cls = node.record_type
    except Exception:
        return None
    return f'{cls.__name__}({", ".join(getattr(cls, "_fields", ()))})'


# ── Resource ─────────────────────────────────────────────────────────────────────

def parse_hash(spec):
    """Parse an ``algo:hexdigest`` hash spec -- the same convention as OCI/Docker
    content digests (``sha256:...``) and pip's ``--hash``/PEP 508 hash-checking
    mode. A bare hex string with no ``:`` is treated as md5, matching what this
    field meant before it supported more than one algorithm.

    ``algo`` must be a name ``hashlib.new`` accepts -- checked here, at
    construction, rather than only surfacing as a confusing failure the first
    time something is actually verified.
    """
    algo, sep, digest = spec.partition(':')
    if not sep:
        algo, digest = 'md5', algo
    algo, digest = algo.lower(), digest.lower()
    try:
        hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f'unknown hash algorithm {algo!r} in {spec!r}') from e
    return algo, digest


def _cache_base(node):
    """The ``<home>/<provider>/<qualified name>`` path a node's own cache
    artifacts are rooted at (see ``sources.default_cache_path``) -- shared by
    ``Resource.cache_path`` (used directly, as the file itself) and
    ``DocTable.docstore_path`` (used as a base, with its own suffix appended, so
    a docstore never collides with its source Resource's own bytes even when
    both happen to be registered under the same name). Lazy, not cached at
    construction: ``provider``/``qualified_name`` aren't set until the node
    is registered (``ManifestProvider.register()``). An ad hoc node that's
    never registered (a local test fixture, a one-off script -- see
    ``registry.ManifestProvider``'s own docstring on that being legitimate)
    falls back to a fixed ``_local`` bucket instead of a real provider's
    directory.
    """
    if node.provider is None:
        return default_cache_path(None, node.name)
    return default_cache_path(node.provider.prefix, node.qualified_name.split(':', 1)[1])


class Resource(Node, Readable):
    """Bytes, and every place they can be obtained from.

    ``sources`` accepts bare URLs or ``Source``/``Source.irds()``/
    ``Source.local()`` for headers, auth, or manual acquisition.

    Integrity is declared as one or more ``algo:hexdigest`` strings --
    ``hashes=['sha256:...']``, or several at once (``hashes=['md5:...',
    'sha256:...']``) when a source publishes more than one and there's no
    reason to pick. ``md5=`` remains a plain shorthand for the common single-
    hash case (``md5='...'`` == ``hashes=['md5:...']``) -- most callers only
    ever declare one hash, and md5 is what the cache path and the live
    download check (both inherited from v1) are keyed on regardless of what
    else is declared, so it stays privileged: ``self.md5`` is always the
    ``md5`` entry of ``self.hashes``, if any.

    Every declared hash is checked at ``verify`` time (a single streamed pass,
    all algorithms computed together) -- not only md5, and not only at
    download time. A ``Resource`` with no hash declared at all is legal:
    parametric families would otherwise require downloading every member at
    freeze time just to record one, so those rely on the (lazy, additive)
    verification registry instead.

    The cache path is **automatic** (see ``sources.default_cache_path``):
    ``<home>/<provider>/<qualified name>``, human-readable and naturally
    partitioned by provider. It is therefore not known until this node is
    registered (``self.provider``/``self.qualified_name`` are set then, not
    at construction) -- see the ``cache_path`` property below, not an
    ``__init__``-time attribute. An existing v1 cache for the same bytes is
    migrated in place the first time it's found (moved to this location,
    with a symlink left at the old v1 path) -- see ``existing_path``.
    """
    type = RESOURCE

    def __init__(self, name, *, sources=(), md5=None, hashes=(), size=None,
                 dua=None, **meta):
        self.hashes = {}
        for spec in (hashes.items() if isinstance(hashes, dict) else hashes):
            algo, digest = parse_hash(spec if isinstance(spec, str) else f'{spec[0]}:{spec[1]}')
            self.hashes[algo] = digest
        if md5:
            self.hashes.setdefault('md5', md5.lower())
        # Hashes must be settled first: materializing a Source.irds()
        # placeholder needs self.md5, which is derived from self.hashes.
        self.sources = [materialize(as_source(s), self.md5) for s in sources]
        self.size = size
        self.dua = dua = default('dua', dua)
        self._download = None
        self._hinted = False
        super().__init__(name, metadata={
            'hashes': [f'{a}:{self.hashes[a]}' for a in sorted(self.hashes)],
            'size': size,
            'sources': [repr(s) for s in self.sources],
            'dua': dua,
        }, **meta)

    @property
    def md5(self):
        """Privileged, not just another entry of ``hashes``: it's what the
        (v1-inherited) live download check is keyed on, and what identifies
        an existing v1 cache file to migrate in ``existing_path``. A
        property, not a value cached at construction, so it can never read
        stale if ``hashes`` is mutated after the fact."""
        return self.hashes.get('md5')

    @property
    def cache_path(self):
        """Where this Resource's bytes live -- see ``_cache_base``, used
        directly (this node *is* the file, unlike a derived artifact such as
        ``DocTable.docstore_path``, which appends its own suffix)."""
        return _cache_base(self)

    def _compute_hashes(self, algos):
        """Every named algorithm's digest, in one streamed pass."""
        hashers = {a: hashlib.new(a) for a in algos}
        with self.stream() as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                for hasher in hashers.values():
                    hasher.update(chunk)
        return {a: hasher.hexdigest() for a, hasher in hashers.items()}

    def attest(self, *, verify=False, **options):
        if not verify or not self.hashes:
            return None
        computed = self._compute_hashes(self.hashes)
        return {'hashes_confirmed': [f'{a}:{computed[a]}' for a in sorted(computed)]}

    def verify(self, frozen, **options):
        """Unlike ``Table.verify`` (which has no author-declared expectation
        and so compares against what a prior freeze observed), a Resource's
        most meaningful check is against what the *author* declared
        (``self.hashes``) -- ``frozen`` is accepted for interface consistency
        but unused: a mismatch against a stale frozen value would just be the
        same divergence already caught here, since ``attest()`` computes from
        ``self.hashes`` in the first place.
        """
        if not self.hashes:
            return []
        name = self.qualified_name or self.name
        computed = self._compute_hashes(self.hashes)
        out = []
        for algo, expected in sorted(self.hashes.items()):
            actual = computed.get(algo)
            if actual != expected:
                out.append(Divergence(name, f'hash[{algo}]', expected, actual))
        return out

    @property
    def download(self):
        if self._download is None:
            self._download = build_download(self)
        return self._download

    def existing_path(self):
        """An already-present local copy: the v2 cache path, if something's
        already there, else an existing v1 cache file -- migrated to the v2
        location the first time it's found (see ``sources.migrate_legacy``),
        with a symlink left at the old v1 path so anything still looking
        there keeps working. Only ever happens once per file: the next call
        (this node's, or any future run's) finds it already at
        ``cache_path`` and takes the fast path above.
        """
        if self.cache_path.exists():
            return self.cache_path
        legacy = legacy_path(self.md5)
        if legacy is not None:
            return migrate_legacy(legacy, self.cache_path)
        return None

    def _hint_local_copy(self):
        """Tell the user where to symlink an existing copy, once, before a
        large download starts."""
        if self._hinted:
            return
        self._hinted = True
        hint = local_copy_hint(self)
        if hint:
            _logger.info(hint)

    def path(self, force=True):
        existing = self.existing_path()
        if existing is not None:
            return str(existing)
        if force:
            self._hint_local_copy()
        return self.download.path(force)

    @contextlib.contextmanager
    def stream(self):
        existing = self.existing_path()
        if existing is None:
            self._hint_local_copy()
        if existing is not None:
            with open(existing, 'rb') as fin:
                yield fin
        else:
            with self.download.stream() as stream:
                yield stream


class File(Resource):
    """A single blob -- exactly what a bare ``Resource`` already is above.

    Kept as a thin, behavior-free subclass rather than folding its own copy
    of ``Resource``'s body in: existing code (dozens of dataset modules)
    constructs single files as plain ``Resource(...)`` and that keeps working
    unchanged. ``File`` exists so *new* code can say what it means once
    ``Resource`` also has tree-shaped siblings (``Directory``, ``GitRepo``)
    that a bare ``Resource(...)`` could otherwise be confused with.
    """


class Directory(Resource):
    """A Resource that is a tree of files, not one blob -- addressed
    member-by-member via ``.relative(path)`` (inherited from ``Readable``,
    the same op ``.member()``/``.gunzip()`` use on a single file), never read
    as a single stream. ``path()`` is unchanged from ``Resource``: it still
    returns the root of the tree, for a caller that wants to walk it itself;
    a parser reading one known member goes through ``.relative(path)``
    instead.
    """
    def stream(self):
        raise TypeError(
            f'{self.name} is a directory, not a single file -- use '
            f'.relative(path) to address one file within it')


class GitRepo(Directory):
    """A git repository, as a whole, pinned to a resolved commit -- the
    shape behind any provider that ships a git-backed dataset (today, HF Hub
    repos via ``hf_provider.HfDataset``; any other git host is the same
    shape). Not fetched via a literal ``git``/``git-lfs`` clone necessarily
    -- a subclass picks how ``path()`` actually materializes the tree (a
    snapshot download, a real clone, ...) -- but it really is the repo's git
    tree, addressed the same way either way: whole, at one commit.

    ``commit`` is always the *resolved* SHA (never a floating branch name),
    so fetching is deterministic; ``pinned`` records only whether the caller
    asked for a specific revision, which affects the node's *name* and what
    ``verify`` checks -- an unpinned repo's verify asks "has the default
    branch moved since"; a pinned one's is trivially satisfied (a commit SHA
    cannot change what it points at). A subclass wanting that check
    overrides ``_resolve_live_commit``; without it, an unpinned repo simply
    has nothing for ``verify`` to compare against drift.
    """
    def __init__(self, repo, *, commit, pinned, **meta):
        self.repo = repo
        self.commit = commit
        self.pinned = pinned
        name = f'{repo}@{commit}.git' if pinned else f'{repo}.git'
        super().__init__(name, **meta)
        self.metadata['repo'] = repo
        self.metadata['commit'] = commit
        try:
            self.metadata['url'] = self.url()
        except NotImplementedError:
            pass  # a host that hasn't implemented url() yet -- repo/commit still identify it

    def url(self):
        """A human-visitable URL for this exact repo -- what you'd paste into
        a browser to look at it, not necessarily how ``path()`` actually
        fetches it (see class docstring). ``self.repo`` alone (e.g.
        "neuclir/csl") isn't enough to tell one host's repos apart from
        another's, so this -- not ``repo`` -- is what a caller (the graph
        export, a UI) should use to distinguish/link to the repo. Override
        per host; the default means "no known URL for this host."""
        raise NotImplementedError

    def _resolve_live_commit(self):
        """The current commit of ``self.repo``'s default branch -- used by
        ``verify`` to detect drift on an unpinned repo. Override per host;
        the default means "can't check," not "no divergence found"."""
        raise NotImplementedError

    def attest(self, *, verify=False, **options):
        return {'commit': self.commit}

    def verify(self, frozen, **options):
        if 'commit' not in frozen or self.pinned:
            return []
        try:
            current = self._resolve_live_commit()
        except NotImplementedError:
            return []
        if current != frozen['commit']:
            name = self.qualified_name or self.name
            return [Divergence(name, 'commit', frozen['commit'], current)]
        return []


# ── Tables ───────────────────────────────────────────────────────────────────

class Table(Node):
    """The abstract base of the per-entity table types: ``DocTable``,
    ``QueryTable``, ``QrelTable``, ``RunTable``, ``DocPairTable``.

    A table is a named set of homogeneous records: a schema (the NamedTuple in
    ``record_type``), rows, and usually a primary key (``doc_id``, ``query_id``).
    Subclasses fix *which* kind of record via ``entity`` (and, correspondingly,
    a more specific ``type`` -- see ``TABLE_TYPES``); the format subclasses in
    ``formats.py`` fix how bytes become those records. ``Table`` itself carries
    the generic ``TABLE`` type (``irds:Table``) -- every concrete entity type
    below is declared as a *subtype* of it (``parent=TABLE``), so a bare
    ``Table`` can still be constructed and registered directly (for a record
    kind that doesn't fit ``ENTITIES``, or in a third-party extension) and is
    found by ``list(type='irds:Table')`` right alongside every ``DocTable``/
    ``QueryTable``/... -- see ``vocabulary.is_subtype``.

    Records come either from a ``source`` + ``parser``, or from a prebuilt
    ``handler`` (which is how a derived table wraps a filtered view of another).

    v1 offered this shape behind a ``_BetaPythonApi*`` wrapper around a handler,
    with ``docs_iter()``/``docs_store()`` as the default. Here it is the
    default, and it needs no wrapper at all: ``collection.docs`` already returns
    the DocTable *node*, so the node simply is the accessor::

        len(collection.docs)                # count
        for doc in collection.docs: ...     # iterate
        collection.docs[5], collection.docs[:10]
        collection.docs.lookup(['d1', 'd2'])
        collection.docs.record_type         # the NamedTuple class

    The legacy API remains available on the same objects (``docs_iter()``,
    ``docs_store()``, ``queries_iter()``, ...), so existing code keeps
    working, but is deprecated (a ``DeprecationWarning`` on each call) in
    favor of the property-style API above.

    Every subclass also has a ``.<entity>`` attribute pointing at itself
    (``docs_table.docs is docs_table``) -- so code written against a
    ``Benchmark``'s facets (``thing.docs``, ``thing.queries``, ...) works
    unchanged when handed a single Table directly instead of a Benchmark.

    Two deliberate divergences from v1's beta wrapper:

    * ``.type`` is the *node* kind (``'docs'``), which is the graph-level concept
      shared with files and collections. The record class is ``.record_type``
      (v1 beta overloaded ``.type`` for it).
    * ``metadata()`` is a method returning the frozen manifest row, a superset of
      what v1's ``.metadata`` property exposed.
    """
    type = TABLE
    entity = None  # 'docs' | 'queries' | 'qrels' | 'scoreddocs' | 'docpairs'

    def __getattr__(self, attr):
        # Self-reference: `docs_table.docs is docs_table`, keyed off whatever
        # `.entity` this instance has -- so a bare Table is a drop-in for
        # anywhere a Benchmark facet (`.docs`, `.queries`, ...) is expected.
        # `__getattr__` (not a per-subclass property) so this also covers
        # DerivedTable, whose `.entity` is set per-instance rather than
        # declared on the class (see DerivedTable.__getattr__, which extends
        # this same check since it fully overrides it, rather than inheriting
        # it -- Python doesn't chain __getattr__ down the MRO on its own).
        if attr == self.entity:
            return self
        raise AttributeError(attr)

    def __init__(self, name, *, source=None, parser=None, handler=None,
                 lang=None, defs=None, count_hint=None,
                 docstore_size_hint=None, **meta):
        if (source is None) == (handler is None):
            raise ValueError('pass exactly one of source= or handler=')
        self.source = source
        self.parser = parser
        self.lang = lang = default('lang', lang)
        self.defs = defs
        self._count_hint = count_hint
        self.docstore_size_hint = docstore_size_hint
        self._handler = handler
        self._index = None
        # A source is an *expression* (a pipeline over one or more Resources), so
        # the edges point at the Resource nodes it bottoms out in; the pipeline
        # steps themselves are unnamed and are not nodes. Recorded as derived_from,
        # same kind as a filtered-from-another-table derivation -- this table
        # would not exist without these bytes either.
        self._structural = [Edge(DERIVED_FROM, r) for r in source_resources(source)]
        super().__init__(
            name,
            metadata={
                'format': repr(parser) if parser is not None else None,
                'lang': lang,
                'defs': ({str(k): v for k, v in defs.items()} if defs else None),
                **meta.pop('metadata', {}),
            },
            **meta)

    def structural_edges(self):
        return list(self._structural)

    @property
    def count_hint(self):
        """Record count, taken from the frozen manifest rather than written by
        hand (v1 kept these in etc/metadata.json)."""
        if self._count_hint is not None:
            return self._count_hint
        return self._frozen().get('count')

    @property
    def handler(self):
        """The v1 handler; built lazily so import stays cheap and offline."""
        if self._handler is None:
            self._handler = self.parser.build(self.source, self)
        return self._handler

    # -- default (beta) API -------------------------------------------------

    def __iter__(self):
        return iter(self._iter())

    def __len__(self):
        return self.count()

    def __getitem__(self, key):
        """Index or slice into the records.

        v1's iterators accept only slices, so a plain ``docs[5]`` failed there.
        Here an integer index is translated into a one-element slice, which also
        keeps it cheap on the seekable doc iterators.
        """
        records = self._iter()
        # Seekable iterators are much cheaper, but they reject negative bounds
        # when the record count is unknown -- fall back to scanning in that case.
        if hasattr(records, '__getitem__'):
            try:
                if isinstance(key, slice):
                    return records[key]
                for record in records[key:] if key < 0 else records[key:key + 1]:
                    return record
                raise IndexError(key)
            except (ValueError, TypeError):
                records = self._iter()

        if isinstance(key, slice):
            if (key.start or 0) < 0 or (key.stop or 0) < 0:
                return list(records)[key]
            return list(itertools.islice(records, key.start, key.stop, key.step))
        if key < 0:
            tail = collections.deque(records, maxlen=-key)
            if len(tail) < -key:
                raise IndexError(key)
            return tail[0]
        try:
            return next(itertools.islice(records, key, key + 1))
        except StopIteration:
            raise IndexError(key) from None

    def count(self):
        """Record count. Prefers the frozen manifest, so it is usually free."""
        frozen = self._frozen().get('count')
        if frozen is not None:
            return frozen
        handler_count = getattr(self.handler, f'{self.entity}_count', None)
        if handler_count is not None:
            value = handler_count()
            if value is not None:
                return value
        return sum(1 for _ in self._iter())

    @property
    def record_type(self):
        """The NamedTuple class for this table's rows."""
        return getattr(self.handler, f'{self.entity}_cls')()

    def lookup(self, ids):
        """Look up records by id. A single id returns one record; an iterable
        returns a dict. Docs use the docstore; queries are held in memory."""
        index = self._lookup_index()
        if isinstance(ids, str):
            return index.get_one(ids)
        return index.get_many(ids)

    def lookup_iter(self, ids):
        index = self._lookup_index()
        if isinstance(ids, str):
            yield index.get_one(ids)
        else:
            yield from index.get_many_iter(ids)

    # -- legacy API (deprecated) ---------------------------------------------

    def record_cls(self):
        _deprecated('record_cls', 'record_type')
        return self.record_type

    # -- attestation --------------------------------------------------------

    def attest(self, *, verify=False, samples=DEFAULT_SAMPLES, **options):
        """Count, content hash and sample records. Expensive (iterates the
        table), so only with ``verify=True``."""
        if not verify:
            return None
        hasher = hashlib.sha256()
        wanted = {i for i in samples if i >= 0}
        collected, last, count = {}, None, 0
        for i, record in enumerate(self._iter()):
            hasher.update(_canonical(record))
            hasher.update(b'\n')
            if i in wanted:
                collected[i] = record
            last = record
            count += 1
        out = {
            'count': count,
            'content_sha256': hasher.hexdigest(),
            'hash_scheme': HASH_SCHEME,
            'record_schema': _schema(self),
            'samples': {str(i): _canonical(r).decode() for i, r in sorted(collected.items())},
        }
        if -1 in samples and last is not None and count:
            out['samples'][str(count - 1)] = _canonical(last).decode()
        return out

    def verify(self, frozen, *, check_samples=True, **options):
        if 'content_sha256' not in frozen:
            return []
        name = self.qualified_name or self.name
        if frozen.get('hash_scheme') != HASH_SCHEME:
            # Not a mismatch: the canonicalization changed, so the two hashes
            # are simply not comparable. Say so rather than reporting a false
            # failure.
            return [Divergence(name, 'hash_scheme', frozen.get('hash_scheme'),
                               HASH_SCHEME)]
        out = []
        schema = _schema(self)
        if frozen.get('record_schema') not in (None, schema):
            out.append(Divergence(name, 'record_schema', frozen['record_schema'], schema))
        hasher = hashlib.sha256()
        wanted = {int(i): v for i, v in frozen.get('samples', {}).items()}
        count = 0
        for i, record in enumerate(self._iter()):
            line = _canonical(record)
            hasher.update(line)
            hasher.update(b'\n')
            if check_samples and i in wanted and line.decode() != wanted[i]:
                out.append(Divergence(name, f'sample[{i}]', wanted[i], line.decode()))
            count += 1
        if count != frozen['count']:
            out.append(Divergence(name, 'count', frozen['count'], count))
        digest = hasher.hexdigest()
        if digest != frozen['content_sha256']:
            out.append(Divergence(name, 'content_sha256', frozen['content_sha256'], digest))
        return out

    # -- internals ----------------------------------------------------------

    def _iter(self):
        return getattr(self.handler, f'{self.entity}_iter')()

    def _lookup_index(self):
        if getattr(self, '_index', None) is None:
            if self.entity == 'docs':
                self._index = _DocstoreIndex(self.docs_store())
            elif self.entity == 'queries':
                self._index = _MemoryIndex(
                    {q.query_id: q for q in self._iter()})
            else:
                raise TypeError(f'{self.entity} does not support lookup')
        return self._index


class _DocstoreIndex:
    def __init__(self, docstore):
        self._docstore = docstore

    def get_one(self, key):
        return self._docstore.get(key)

    def get_many(self, keys):
        return self._docstore.get_many(keys)

    def get_many_iter(self, keys):
        return self._docstore.get_many_iter(keys)


class _MemoryIndex:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_one(self, key):
        return self._mapping[key]

    def get_many(self, keys):
        return {k: self._mapping[k] for k in keys if k in self._mapping}

    def get_many_iter(self, keys):
        for key in keys:
            if key in self._mapping:
                yield self._mapping[key]


class DocTable(Table):
    type = TABLE_TYPES['docs']
    entity = 'docs'

    def __init__(self, name, *, docstore='auto', **kwargs):
        self.docstore = docstore
        super().__init__(name, **kwargs)

    @property
    def docstore_path(self):
        """Where the random-access index lives: the same ``<home>/<provider>/
        <qualified name>`` base a Resource's own ``cache_path`` uses (see
        ``_cache_base``), plus a format-versioned suffix of its own -- a
        derived artifact should be identifiable on disk as one, and a format
        bump must not collide with the old one. Provider-partitioned and
        lazy for the same reason as ``cache_path``. An existing v1 docstore
        is migrated to this location the first time it's found (see
        ``sources.migrate_legacy``), rather than read from the v1 layout
        forever.
        """
        base = _cache_base(self)
        cache_path = base.with_name(f'{base.name}.docstore.fmt{DOCSTORE_FORMAT}.pklz4')
        if not cache_path.exists():
            legacy = legacy_path(self.source_resource.md5) if self.source_resource else None
            if legacy is not None:
                candidate = legacy.parent / f'{legacy.name}.pklz4'
                if candidate.exists():
                    return migrate_legacy(candidate, cache_path)
        return cache_path

    @property
    def source_resource(self):
        """The Resource at the root of this node's source pipeline, if any."""
        resources = source_resources(self.source)
        return resources[0] if resources else None

    def docs_store(self):
        """Random-access lookup by doc_id; built on first use and cached.

        ``docstore=None`` opts out for corpora where an index is not worth it.
        """
        if self.docstore is None:
            raise TypeError(f'{self.name} has no docstore')
        return self.handler.docs_store()

    # -- legacy API (deprecated) ---------------------------------------------

    def docs_iter(self):
        _deprecated('docs_iter', 'docs')
        return self._iter()

    def docs_cls(self):
        _deprecated('docs_cls', 'docs.record_type')
        return self.record_type

    def docs_count(self):
        _deprecated('docs_count', 'docs (via len)')
        return self.count()


class QueryTable(Table):
    type = TABLE_TYPES['queries']
    entity = 'queries'

    # -- legacy API (deprecated) ---------------------------------------------

    def queries_iter(self):
        _deprecated('queries_iter', 'queries')
        return self._iter()

    def queries_cls(self):
        _deprecated('queries_cls', 'queries.record_type')
        return self.record_type

    def queries_count(self):
        _deprecated('queries_count', 'queries (via len)')
        return self.count()


class QrelTable(Table):
    type = TABLE_TYPES['qrels']
    entity = 'qrels'

    # -- legacy API (deprecated) ---------------------------------------------

    def qrels_iter(self):
        _deprecated('qrels_iter', 'qrels')
        return self._iter()

    def qrels_cls(self):
        _deprecated('qrels_cls', 'qrels.record_type')
        return self.record_type

    def qrels_count(self):
        _deprecated('qrels_count', 'qrels (via len)')
        return self.count()

    def qrels_defs(self):
        _deprecated('qrels_defs', 'qrels.defs')
        return self.handler.qrels_defs()

    def asdict(self):
        _deprecated('asdict', 'qrels (iterate directly)')
        from ir_datasets.formats.base import qrels_dict
        return qrels_dict(self.handler)


class RunTable(Table):
    type = TABLE_TYPES['scoreddocs']
    entity = 'scoreddocs'

    # -- legacy API (deprecated) ---------------------------------------------

    def scoreddocs_iter(self):
        _deprecated('scoreddocs_iter', 'scoreddocs')
        return self.handler.scoreddocs_iter()


class DocPairTable(Table):
    type = TABLE_TYPES['docpairs']
    entity = 'docpairs'

    # -- legacy API (deprecated) ---------------------------------------------

    def docpairs_iter(self):
        _deprecated('docpairs_iter', 'docpairs')
        return self.handler.docpairs_iter()


def source_resources(source):
    """Every Resource a source expression bottoms out in.

    A source may be a Resource, a pipeline over one (``.member().gunzip()``), or a
    list of either -- so this walks the chain rather than assuming one root.
    """
    out, seen, stack = [], set(), [source]
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            stack.extend(item)
            continue
        if isinstance(item, Resource):
            if item.name not in seen:
                seen.add(item.name)
                out.append(item)
            continue
        stack.append(getattr(item, '_parent', None))
    return out


# ── Benchmark ────────────────────────────────────────────────────────────────

class Benchmark(Node):
    """Docs + queries + qrels (etc.), bundled into one evaluable task.

    ``citation`` (inherited from the generic ``Node``) and ``metrics`` are
    plain metadata here, not edges to paper/measure nodes -- that richer model
    is future work for a broader knowledge graph. ``metrics`` is a list of
    measure-spec strings (e.g. ``['nDCG@10', 'P(rel=3)@10']``), understood by
    ``ir_measures`` but not resolved or validated against it.

    Facets may be given as a node (intra-package: refactor-safe) or as a name
    string (cross-package, or another family in the same package: resolved
    lazily at access time, so the other module need not be imported until used;
    a bare name means this provider's own namespace).

    A *derived* benchmark names a parent plus a Filter; it inherits every facet
    the parent has and applies the filter to the query-keyed ones. The derived
    facets are nodes of their own (``<benchmark>-<entity>``); they are built
    when the benchmark's structural edges are read, which is what registers
    them alongside it -- so ``load('irds:antique-test-non-offensive-queries')``
    works like any other name, with no special case anywhere.
    """
    type = BENCHMARK

    def __init__(self, name, *, docs=None, queries=None, qrels=None,
                 scoreddocs=None, docpairs=None, derived_from=None, filter=None,
                 metrics=None, **meta):
        self._facets = {'docs': docs, 'queries': queries, 'qrels': qrels,
                        'scoreddocs': scoreddocs, 'docpairs': docpairs}
        self.derived_from = derived_from
        self.filter = filter
        self.metrics = list(metrics) if metrics else None
        self._resolved = {}
        super().__init__(
            name,
            metadata={'filter': repr(filter) if filter is not None else None,
                      'metrics': self.metrics,
                      **meta.pop('metadata', {})},
            **meta)

    def structural_edges(self):
        edges = []
        for entity in ENTITIES:
            target = self._facets.get(entity)
            if target is None and self.derived_from is not None:
                # Inherited: build the (cheap, lazy) derived facet now so it is
                # registered with the collection and has its own edges.
                target = self.edge(entity)
            if target is not None:
                edges.append(Edge(FACET[entity], target))
        # No derived_from here: a Benchmark is a bundle of facet edges, not
        # itself a derived thing. What a derivation actually produces is
        # derived *tables* (see DerivedTable/Filter.apply), and that is where
        # derived_from belongs -- each one naming exactly what that table
        # would not exist without (its filtered parent table, and, for
        # "queries with qrels", the qrels it reads its id set from too). A
        # benchmark built entirely from shared/unfiltered
        # facets (e.g. docs, and now a tautologically-unfiltered qrels) has no
        # derivation edge of its own, and that's fine: the graph shows
        # derivation exactly where it's real, not one level up by convention.
        return edges

    # -- facets -------------------------------------------------------------

    def _lookup(self, name):
        """Resolve a name-string target through the default graph. A bare
        name is read in this benchmark's own provider namespace."""
        from .graph import default_graph
        if self.provider is not None:
            name = self.provider.qualify(name)
        return default_graph()[name]

    @property
    def parent(self):
        if self.derived_from is None:
            return None
        if isinstance(self.derived_from, str):
            return self._lookup(self.derived_from)
        return self.derived_from

    def edge(self, entity):
        """Resolve one facet, applying a derivation filter if present."""
        if entity in self._resolved:
            return self._resolved[entity]
        node = self._facets.get(entity)
        if node is None and self.parent is not None:
            node = self.parent.edge(entity)
            if node is not None and self.filter is not None:
                node = self.filter.apply(entity, node, self.parent, self.name)
        elif isinstance(node, str):
            node = self._lookup(node)
        self._resolved[entity] = node
        return node

    def has(self, entity):
        return self.edge(entity) is not None

    @property
    def docs(self):
        return self.edge('docs')

    @property
    def queries(self):
        return self.edge('queries')

    @property
    def qrels(self):
        return self.edge('qrels')

    @property
    def scoreddocs(self):
        return self.edge('scoreddocs')

    @property
    def docpairs(self):
        return self.edge('docpairs')

    # -- legacy API (deprecated) ---------------------------------------------
    # The default API is property access (`benchmark.docs`, which returns the
    # table node and so supports len/iter/slice/lookup). These keep v1-style
    # method code working, with a DeprecationWarning nudging toward the
    # property-style API above.

    def _entity(self, entity):
        node = self.edge(entity)
        if node is None:
            raise TypeError(f'{self.qualified_name} has no {entity}')
        return node

    def docs_iter(self):
        _deprecated('docs_iter', 'docs')
        return self._entity('docs')._iter()

    def queries_iter(self):
        _deprecated('queries_iter', 'queries')
        return self._entity('queries')._iter()

    def qrels_iter(self):
        _deprecated('qrels_iter', 'qrels')
        return self._entity('qrels')._iter()

    def scoreddocs_iter(self):
        _deprecated('scoreddocs_iter', 'scoreddocs')
        return self._entity('scoreddocs')._iter()

    def docpairs_iter(self):
        _deprecated('docpairs_iter', 'docpairs')
        return self._entity('docpairs')._iter()

    def docs_count(self):
        _deprecated('docs_count', 'docs (via len)')
        return self._entity('docs').count()

    def queries_count(self):
        _deprecated('queries_count', 'queries (via len)')
        return self._entity('queries').count()

    def qrels_count(self):
        _deprecated('qrels_count', 'qrels (via len)')
        return self._entity('qrels').count()

    def docs_cls(self):
        _deprecated('docs_cls', 'docs.record_type')
        return self._entity('docs').record_type

    def queries_cls(self):
        _deprecated('queries_cls', 'queries.record_type')
        return self._entity('queries').record_type

    def qrels_cls(self):
        _deprecated('qrels_cls', 'qrels.record_type')
        return self._entity('qrels').record_type

    def docs_store(self):
        _deprecated('docs_store', 'docs.lookup')
        return self._entity('docs').docs_store()

    def qrels_defs(self):
        _deprecated('qrels_defs', 'qrels.defs')
        return self._entity('qrels').qrels_defs()

    def has_docs(self):
        _deprecated('has_docs', "has('docs')")
        return self.has('docs')

    def has_queries(self):
        _deprecated('has_queries', "has('queries')")
        return self.has('queries')

    def has_qrels(self):
        _deprecated('has_qrels', "has('qrels')")
        return self.has('qrels')

    def has_scoreddocs(self):
        _deprecated('has_scoreddocs', "has('scoreddocs')")
        return self.has('scoreddocs')

    def has_docpairs(self):
        _deprecated('has_docpairs', "has('docpairs')")
        return self.has('docpairs')

    def docs_handler(self):
        _deprecated('docs_handler', 'docs.handler')
        return self._entity('docs').handler

    def queries_handler(self):
        _deprecated('queries_handler', 'queries.handler')
        return self._entity('queries').handler

    def qrels_handler(self):
        _deprecated('qrels_handler', 'qrels.handler')
        return self._entity('qrels').handler

    def scoreddocs_handler(self):
        _deprecated('scoreddocs_handler', 'scoreddocs.handler')
        return self._entity('scoreddocs').handler

    def docpairs_handler(self):
        _deprecated('docpairs_handler', 'docpairs.handler')
        return self._entity('docpairs').handler


# ── Suite ────────────────────────────────────────────────────────────────────

class Suite(Node):
    """A named set of related Benchmarks (e.g. BEIR: many corpora, one
    evaluation convention).

    Membership is structural: what's in a suite is a fact about the suite's
    identity, fixed at construction, not a curated list that might drift. A
    benchmark may belong to more than one suite -- nothing here prevents it,
    and ``graph.referrers(name, kind=SUITE_MEMBER)`` answers "which suites
    directly contain this benchmark?" without either side needing a
    back-reference.

    A suite may also nest other suites (``suites=``) -- e.g. BEIR nests a
    ``beir-cqadupstack`` suite (its 12 StackExchange sub-forums) rather than
    flattening all 12 into BEIR's own direct member list. ``.benchmarks``
    still returns every benchmark the suite contains *transitively*, so
    existing code that just wants "all benchmarks in this suite" doesn't need
    to know nesting exists; ``.direct_benchmarks``/``.suites`` expose the
    unflattened structure for anything that cares about it (e.g. a site
    wanting to render CQADupStack as its own collapsible group). Structural
    edges are ``SUITE_MEMBER`` to direct members only -- a benchmark reached
    via a nested suite is not redundantly double-edged.

    Unlike Benchmark, a Suite has no lazy facet resolution -- membership is
    fixed at construction (a nested suite's own membership was already fixed
    at *its* construction), so ``structural_edges()`` is all there is.
    """
    type = SUITE

    def __init__(self, name, *, benchmarks=(), suites=(), **meta):
        self.direct_benchmarks = list(benchmarks)
        self.suites = list(suites)
        super().__init__(name, **meta)

    @property
    def benchmarks(self):
        """Every benchmark this suite contains, transitively through any
        nested suites. Recomputed on each access rather than cached -- cheap
        (list concatenation over what's already fixed), and avoids the
        question of whose construction time it would otherwise be frozen at.
        A nested suite given by bare name (not an object) can't be expanded
        here without a graph, so it contributes nothing; its edge still exists.
        """
        result = list(self.direct_benchmarks)
        for suite in self.suites:
            if isinstance(suite, Suite):
                result.extend(suite.benchmarks)
        return result

    def structural_edges(self):
        return [Edge(SUITE_MEMBER, m) for m in (*self.direct_benchmarks, *self.suites)]


