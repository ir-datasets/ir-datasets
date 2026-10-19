"""The structural interfaces a graph node satisfies -- not base classes.

Nothing here requires inheritance. A ``Node`` is any object with these
attributes and methods; ``registry.Provider.register()`` and ``graph.Graph``
only ever touch a node through this shape, never through ``isinstance``
against a concrete class. That's deliberate: the concrete classes in
``base.py``/``nodes.py`` (``base.Node``, ``nodes.Resource`` (the ``Data`` node class), ``nodes.Table``,
``nodes.Benchmark``, ``nodes.Suite``) are *one* implementation of these
protocols -- a convenient one to inherit from, so authors don't re-write
``__init__``'s bookkeeping by hand -- but they are not the contract. A
third-party package can build a node -- a ``Table`` backed by a live API
instead of a downloaded file, say -- that satisfies ``Table`` below
without importing anything from this package except (optionally) this module,
and it joins the graph exactly as any of ours does.

Protocols may declare attributes as well as methods (PEP 544) -- a bare
annotation (``name: str``) is satisfied by an instance attribute set however
the implementer likes, not only by a matching method. ``Node``'s
``qualified_name``/``provider``/``defined_in`` are exactly that: plain
settable attributes, because ``Provider.register()`` assigns them directly
(``node.qualified_name = name``).

How types and protocols fit together
------------------------------------
Every node has a *type* (``irds:Docs``) and every type promises a protocol
(this module's ``Docs``). The protocol is also the type's only declaration:
mark the class with ``@graph_class`` and the provider declares it from there::

    @graph_class
    @runtime_checkable
    class Docs(Table, Protocol):
        '''A Table representing a collection of documents (a corpus).

        More detail, not part of the short description.'''

    TYPES = irds.declare_graph_classes(this_module)   # {protocol class: 'irds:Docs'}

The type's name is the class name, its parent is the type of the nearest marked
base protocol (``Table``), its short description is the docstring's first
paragraph, and its protocol is the class itself -- so there is nothing to keep
in step by hand. A third party does the same with its own provider and
protocols that extend ours. A type declared without a protocol of its own
(``provider.declare_graph_class('Name', parent=...)``) inherits its parent's.

A protocol is a promise about *shape*, not a base class: anything with these
members is a valid node of the type, whether or not it inherits from this
package's classes. ``ir_datasets.v2.nodes`` is one implementation of all of
them; ``test/v2_conformance.py`` checks that a real node of every type
satisfies the protocol its type declares.

These are marked ``@runtime_checkable`` for convenience (``isinstance(x, Table)``
reads well in a test), but that check is presence-only: it confirms the named
attributes and methods exist, never their types or signatures. Nothing in this
package relies on it for anything load-bearing -- see ``registry.py``'s own,
narrower check.
"""
from typing import Any, ContextManager, Optional, Protocol, runtime_checkable


def graph_class(cls):
    """Mark a protocol class as declaring a graph class.

    The class name is the type's name, the nearest marked base protocol is its
    parent type, the docstring's first paragraph is its short description, and
    the class itself is the protocol its nodes satisfy. Nothing is registered
    by marking: a provider does that with ``provider.declare_graph_classes(module)``,
    which declares every marked protocol class the module defines. So this
    module stays plain Python and a protocol can be written without a provider.
    """
    _NODE_TYPES.add(cls)   # not an attribute on the class: protocols would then require it
    return cls


_NODE_TYPES = set()


def is_graph_class(cls):
    """Whether ``cls`` was marked with ``graph_class``."""
    return cls in _NODE_TYPES


#: The ``Node`` members a provider assigns when a node is registered, so a node
#: need not have them yet.
ASSIGNED_AT_REGISTRATION = ('qualified_name', 'provider', 'defined_in')


def members(protocol):
    """The member names a protocol requires, including those of the protocols it
    extends: its annotated attributes, methods and properties."""
    names = set()
    for cls in protocol.__mro__:
        if cls is Protocol or not getattr(cls, '_is_protocol', False):
            continue
        names.update(cls.__dict__.get('__annotations__', {}))
        names.update(name for name, member in cls.__dict__.items()
                     if (callable(member) or isinstance(member, property))
                     and name not in ('__init__', '__subclasshook__', '__class_getitem__')
                     and not name.startswith('_abc') and name != '_is_protocol')
    return names


@runtime_checkable
class Node(Protocol):
    """What ``Provider.register()`` and ``Graph`` require of every node: an
    identity, a type, some descriptive metadata, and one hook -- what it
    depends on. Every other protocol below extends this one.

    A node is created with a bare ``name``; registering it with a provider
    qualifies that name (``antique-test`` -> ``irds:antique-test``) and fills
    in ``qualified_name``, ``provider`` and ``defined_in``.

    A node may belong to several types at once (a benchmark with both qrels and
    reference answers is an ``irds:AdhocBenchmark`` and an ``irds:QaBenchmark``).
    If it does, it also has ``types``, a tuple with ``type`` first; ``type`` is
    always present and is the first (primary) one.
    """
    #: The bare, unqualified name given at construction.
    name: str
    #: The qualified graph class (``irds:Docs``) -- declared with
    #: ``provider.declare_graph_class(...)``, which also says which protocol applies.
    type: str
    #: Descriptive fields (``desc``, ``citation``, ``license``, ...). These are
    #: what the graph reports as the node's properties; never identity or
    #: data.
    metadata: dict
    #: ``<prefix>:<name>``. ``None`` until the node is registered.
    qualified_name: Optional[str]
    #: The provider that registered it. ``None`` until then.
    provider: Optional[Any]
    #: The module that defined it (for error messages).
    defined_in: Optional[str]

    def structural_edges(self) -> list:
        """What this node is built from, as ``Edge`` objects whose targets are
        nodes or qualified names. They form the build DAG: building, caching
        and verification follow them, so a cycle is an error. Informational
        links (citations, alternatives, aliases) are *not* returned here."""
        ...


@graph_class
@runtime_checkable
class Data(Node, Protocol):
    """Raw data and where to get it

    The ``irds:File`` and ``irds:Directory`` protocols below are the two
    shapes it takes, and what they share is only where the bytes are, not how
    to read them. A Table is parsed *from* Data (a ``derived_from``
    edge), and Data knows nothing about what its bytes mean."""
    #: Every place the bytes can be obtained: URLs, mirrors, or instructions for
    #: getting them by hand.
    sources: list
    #: Expected size in bytes, if declared.
    size: Optional[int]
    #: Where the bytes live once fetched, derived from the provider and name.
    cache_path: Any

    def path(self, force: bool = True) -> str:
        """The local filesystem path of the bytes. With ``force=True`` (the
        default) fetch and validate them first if needed; with ``False``
        return the path without fetching anything."""
        ...


@graph_class
@runtime_checkable
class File(Data, Protocol):
    """A single blob of bytes

    ``path()`` is the file itself and ``stream()`` reads it. This is what a
    plain ``Resource(...)`` is."""

    def stream(self) -> ContextManager:
        """A context manager yielding a binary file object over the bytes."""
        ...


@graph_class
@runtime_checkable
class Directory(Data, Protocol):
    """A tree of files (usually a non-downloadable external resource)

    It is addressed file by file. ``path()`` is the root of the tree (for a caller that wants to walk it). There is no ``stream()``, since
    a directory is not one blob -- list the files with ``files()`` and reach
    one with ``file()``."""

    def file(self, path: str):
        """One file in the tree, as an object with the same ``path()`` and
        ``stream()`` operations as a File. ``path`` is relative to the root
        of the tree, ``/``-separated."""
        ...

    def files(self):
        """Iterate every file in the tree as a relative path (the kind
        ``file()`` takes), sorted by path, so a tree always lists the same
        way. Fetches and validates the tree first, like ``path()``."""
        ...


@graph_class
@runtime_checkable
class Table(Node, Protocol):
    """A structured set of records

    A homogeneous, ordered set of records of one ``entity`` kind: the
    shared contract of every table type (``Docs``, ``Queries``,
    ``Qrels``, ...). Each type fixes which fields a record has; the
    contract here is how you read them. A table is lazy: nothing is parsed or
    downloaded until you iterate, index or look something up."""
    #: What the records are: ``'docs'``, ``'queries'``, ``'qrels'``,
    #: ``'scoreddocs'``, ``'docpairs'``, ``'answers'`` or ``'qlogs'``.
    entity: str

    def count(self) -> int:
        """The number of records: the author's count hint if there is one,
        else the handler's own count, else by iterating."""
        ...

    @property
    def record_type(self) -> type:
        """The ``NamedTuple`` class every record is an instance of (its
        ``_fields`` are the column names)."""
        ...

    def __iter__(self):
        """Iterate the records, in the table's own order, starting from the
        beginning each time."""
        ...

    def __len__(self) -> int:
        """Same as ``count()``."""
        ...

    def __getitem__(self, key):
        """A record by position (``table[5]``, negative counts from the end)
        or a list of them by slice (``table[:10]``). Raises ``IndexError`` past
        the end. Cheap on tables that can seek; otherwise it scans."""
        ...

    def lookup(self, ids):
        """Records by *id* (not position). One id (a string) returns that
        record; an iterable of ids returns a dict of id -> record. Supported
        by docs, queries and answers tables (an answers lookup yields that
        query's list of answers); other kinds raise ``TypeError``."""
        ...


@graph_class
@runtime_checkable
class Docs(Table, Protocol):
    """A Table representing a collection of documents (a corpus). Each row has a doc_id field.

    ``lookup(doc_id)`` fetches a document by id."""


@graph_class
@runtime_checkable
class Queries(Table, Protocol):
    """A Table representing a collection of queries (topics). Each row has a query_id field.

    ``lookup(query_id)`` fetches a query by id."""


@graph_class
@runtime_checkable
class Qrels(Table, Protocol):
    """A Table representing a collection of relevance assessments (qrels). Each row has query_id, doc_id, and relevance fields.

    The ``relevance`` values are integer labels; the table's ``defs`` say what
    each label means."""


@graph_class
@runtime_checkable
class AdhocRun(Table, Protocol):
    """A Table representing a collection of retrieved and scored documents for a set of queries (a run). Each row has query_id, doc_id, and score fields."""


@graph_class
@runtime_checkable
class DocPairs(Table, Protocol):
    """A Table representing paired documents for a query (e.g. positive/negative pairs for training). Each row has query_id, doc_id_a, and doc_id_b fields."""


@graph_class
@runtime_checkable
class Answers(Table, Protocol):
    """A Table representing reference answers to queries (e.g. the ground-truth answer to a question, for end-to-end QA / deep-research evaluation). Each row has query_id and text fields.

    A query may have several answers (one row each), so ``lookup(query_id)``
    returns that query's list of answers."""


@graph_class
@runtime_checkable
class QueryLog(Table, Protocol):
    """A Table representing a raw query log (e.g. search-session records issued against a live system).

    Not a Benchmark facet: a log is a standalone artifact alongside a corpus,
    not a component an evaluable task is built from. Row shape is
    family-specific, but typically carries a user/session id, the (possibly
    hashed) query, and the ranked/clicked items shown for it."""


@graph_class
@runtime_checkable
class Bundle(Node, Protocol):
    """A node composed of Tables and/or other Bundles.

    The parts are the node's
    structural edges (see ``Node.structural_edges``), so a Bundle's contents
    are part of the build DAG; the subtypes below add what the parts are
    called and what they mean. Not a "collection" in the IR sense -- that is a
    ``Docs`` table."""


@graph_class
@runtime_checkable
class Alias(Node, Protocol):
    """A node that redirects to another node

    Stands for another node and has no content of its own (for example a
    dataset id from the first version of the library). Its ``irds:aliasOf``
    edge names the target, and loading the alias through the graph returns that
    target instead. An alias with no ``alias_of`` edge is listed and
    browsable but cannot be loaded."""


@graph_class
@runtime_checkable
class Benchmark(Bundle, Protocol):
    """A bundle of tables that represents a task that can be evaluated.

    Tables bundled into one evaluable task: each is a *facet*, named by its
    entity (``docs``, ``queries``, ``qrels``, ``scoreddocs``, ``docpairs``,
    ``answers``) and linked by a structural edge of the same name. The
    subtypes say which facets the task needs: an ``irds:AdhocBenchmark`` has
    docs, queries and qrels; an ``irds:QaBenchmark`` has queries and answers.
    The Benchmark also carries what the Tables do not: the citation and the
    measures it is scored with."""

    def edge(self, entity: str) -> Optional[Table]:
        """The Table that is this benchmark's ``entity`` facet, or ``None``
        if it has none."""
        ...

    def has(self, entity: str) -> bool:
        """Whether it has an ``entity`` facet, without building the Table."""
        ...


@graph_class
@runtime_checkable
class AdhocBenchmark(Benchmark, Protocol):
    """An ad hoc retrieval Benchmark: rank documents from a corpus for each query, evaluated against qrels.

    Has ``docs``, ``queries`` and ``qrels`` facets (and may have a run)."""


@graph_class
@runtime_checkable
class QaBenchmark(Benchmark, Protocol):
    """A question-answering Benchmark: generate an answer for each query, evaluated against reference answers.

    Has ``queries`` and ``answers`` facets. A ``docs`` facet is optional
    (open-book versus closed-book)."""


@graph_class
@runtime_checkable
class Overview(Bundle, Protocol):
    """The front page of a dataset: a bundle with a hand-written summary.

    Brings the tables, benchmarks and resources of one dataset (or a family that share a
    story) under a single name, and explains them in a Markdown ``summary``: what the data
    is, how it was made, how its parts relate and how to use them. An Overview always has a
    summary; a plain ``irds:Bundle`` is only a grouping and need not."""


@graph_class
@runtime_checkable
class Suite(Bundle, Protocol):
    """A set of related benchmarks.

    A named set of related Benchmarks (for example BEIR) evaluated
    together. Suites can contain other Suites."""
    #: Every Benchmark in the suite, including those of nested suites.
    benchmarks: list


@runtime_checkable
class Provider(Protocol):
    """What joins a package to the graph. ``Graph`` touches a provider through
    exactly these two members -- nothing else is load-bearing.

    ``registry.ManifestProvider`` is one implementation of this (a big one:
    registration, vocabulary declaration, generators) --
    a convenient default for a provider with a static catalog, not
    the contract itself. A provider can be as small as a ``name``, a
    ``load()``, and a ``discover_edges()`` that yields nothing; a typical one
    keeps a list of registered datasets internally, however it likes.
    """
    #: The namespace prefix this provider owns (``irds``, ``hf``, ...).
    #: Every node it resolves is named ``{prefix}:...``.
    prefix: str

    def load(self, name: str) -> Node:
        """Resolve one (already-qualified) name to a node. KeyError if this
        provider has nothing by that name."""
        ...

    def discover_edges(self):
        """This provider's whole current catalog, as RDF-ready
        ``(subject, predicate, object)`` rows -- ``object`` wrapped in
        ``base.Literal`` when it's a property value rather than another
        node's name (see ``registry.row_triples``) -- a ``type`` row per node
        plus one row per field/edge. Never materializes a real node (``load()`` does
        that); this is the cheap-to-read side of the graph -- listing,
        traversal, validation -- that works without importing or downloading
        anything a node itself would need.

        Always does full discovery -- a live crawl, if that's what finding
        everything takes (see ``hf``, which lists the Hub here). There is no
        separate cheap/expensive mode: a provider that wants a fast default
        should cache its own result and invalidate it on its own terms;
        ``Graph`` caches its own call to this per provider so a query doesn't
        repeat the work, but only for that ``Graph``'s lifetime.
        """
        ...
