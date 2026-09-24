"""Derivation: the ``derived_from`` + ``Filter`` pair.

A large fraction of real datasets are *derived* — "the judged subset", "these
200 held-out queries", "minus the offensive ones". In v1 each of these was
hand-assembled from ``FilteredQueries``/``FilteredQrels``/... per facet. Here
the derivation is declared once on the collection and applied to every
query-keyed facet automatically.

Derived facets are named too (``<collection>-<entity>``) and registered with
the collection that owns them, so ``load('antique-test-non-offensive-queries')``
works like any other node.
"""
from ir_datasets.datasets.base import (
    FilteredDocPairs, FilteredQrels, FilteredQueries, FilteredScoredDocs,
)
from ir_datasets.util import Lazy

from .nodes import DERIVED_FROM, Edge, TABLE, Table, _deprecated, source_resources

_FILTERED = {
    'queries': FilteredQueries,
    'qrels': FilteredQrels,
    'scoreddocs': FilteredScoredDocs,
    'docpairs': FilteredDocPairs,
}


class DerivedTable(Table):
    """A filtered view of another table.

    An ordinary ``Table`` in every respect -- same default and legacy record
    APIs -- it just takes a prebuilt (filtered) handler instead of a source.
    It is constructed by ``Filter.apply`` when the owning collection's
    structural edges are read, and registered along with that collection.
    """

    def __init__(self, name, entity, handler, derived_from, *,
                 extra_derived_from=(), filtered_by=()):
        self.type = TABLE
        self.entity = entity
        self.derived_from = derived_from
        super().__init__(
            name,
            handler=handler,
            lang=getattr(derived_from, 'lang', None),
            namespace=getattr(derived_from, 'namespace', None),
            defs=getattr(derived_from, 'defs', None))
        # A table may be derived from more than one node at once -- e.g. a
        # "/judged" queries table is filtered *from* the parent's queries but
        # its id set comes *from* the parent's qrels, and an explicit-id-set
        # filter's id source is a third case; all are things this table would
        # not exist without, so all are plain derived_from edges (there used
        # to be a separate filtered_by kind for the id-source case; dropped
        # as redundant -- it always paired 1:1 with a derived_from anyway).
        self._structural = [Edge(DERIVED_FROM, derived_from)]
        self._structural += [Edge(DERIVED_FROM, t) for t in extra_derived_from]
        self._structural += [Edge(DERIVED_FROM, t) for t in filtered_by]

    def __getattr__(self, attr):
        # Fully overrides Table.__getattr__ (Python doesn't chain __getattr__
        # down the MRO), so both checks it would otherwise inherit are
        # repeated here: the self-reference (`derived.<entity> is derived`)...
        entity = self.__dict__.get('entity')
        if attr == entity:
            return self
        # ...and forwarding legacy queries_iter / qrels_defs / *_cls to the
        # handler, since a DerivedTable is not a Queries/Qrels subclass with
        # its own deprecated wrapper methods to call through.
        if entity and attr.startswith(f'{entity}_'):
            _deprecated(attr, entity)
            return getattr(self.__dict__['_handler'], attr)
        raise AttributeError(attr)


class Filter:
    """Which queries survive a derivation.

    Exactly one selector:

    * ``queries_with_qrels=True`` — keep queries that have >= 1 qrel (the
      ubiquitous ``/judged`` case)
    * ``query_ids=<set | callable | ids_of(...) | ids_from_lines(...)>`` —
      an explicit id set, with ``mode='include'`` or ``'exclude'``
    """

    def __init__(self, *, query_ids=None, mode='include', queries_with_qrels=False):
        if (query_ids is None) == (not queries_with_qrels):
            raise ValueError('pass exactly one of query_ids or queries_with_qrels')
        if mode not in ('include', 'exclude'):
            raise ValueError("mode must be 'include' or 'exclude'")
        self.query_ids = query_ids
        self.mode = mode
        self.queries_with_qrels = queries_with_qrels

    def depends_on(self):
        """Nodes an explicit ``query_ids`` source reads its ids from, if any
        (``queries_with_qrels`` is not one of these -- its dependency is the
        parent's own qrels, handled directly in ``apply`` as a second
        ``derived_from``, not an external filter source)."""
        return tuple(getattr(self.query_ids, 'depends_on', ()))

    def _lazy_qids(self, parent):
        if self.queries_with_qrels:
            return Lazy(lambda: {q.query_id for q in parent.qrels_iter()})
        ids = self.query_ids
        if callable(ids):
            return Lazy(lambda: set(ids()))
        return Lazy(lambda: set(ids))

    def apply(self, entity, node, parent, owner_name):
        # Docs are not query-keyed: a derivation never changes the corpus.
        if entity == 'docs':
            return node
        if entity == 'qrels' and self.queries_with_qrels:
            # Tautology: every qrel's own query trivially "has a qrel", so
            # filtering qrels down to queries-with-qrels changes nothing.
            # Share the parent's qrels node instead of manufacturing an
            # identical-but-differently-named duplicate.
            return node
        cls = _FILTERED.get(entity)
        if cls is None:
            return node
        mode = 'include' if self.queries_with_qrels else self.mode
        handler = cls(node.handler, self._lazy_qids(parent), mode=mode)
        extra_derived_from, filtered_by = (), ()
        if self.queries_with_qrels:
            if parent is not None and parent.has('qrels'):
                extra_derived_from = (parent.edge('qrels'),)
        else:
            filtered_by = self.depends_on()
        return DerivedTable(f'{owner_name}-{entity}', entity, handler, node,
                            extra_derived_from=extra_derived_from,
                            filtered_by=filtered_by)

    def __repr__(self):
        if self.queries_with_qrels:
            return 'Filter(queries_with_qrels=True)'
        depends = self.depends_on()
        if depends:
            source = ', '.join(d if isinstance(d, str) else d.name for d in depends)
        else:
            source = f'{len(self.query_ids)} ids' \
                if hasattr(self.query_ids, '__len__') else 'callable'
        return f'Filter(query_ids={source}, mode={self.mode!r})'


def ids_of(name):
    """Lazily take the query ids of another node (possibly another package).

    Carries ``depends_on`` so the dependency shows up as a ``derived_from``
    edge instead of disappearing into a closure.
    """
    def _ids():
        from . import graph
        node = graph[name]
        if hasattr(node, 'queries_iter'):
            return {q.query_id for q in node.queries_iter()}
        return {q.query_id for q in node.qrels_iter()}
    _ids.depends_on = (name,)
    return _ids


def ids_from_lines(file_node):
    """Lazily read one id per line from a Resource, or a pipeline over one
    (``some_archive.member('ids.txt')``) -- either way ``depends_on``
    resolves down to the real ``Resource`` node(s) it bottoms out in (see
    ``source_resources``), the same way ``Table.__init__`` does for a
    ``source=``, since a bare pipeline step has no name/identity of its own
    to hang a ``derived_from`` edge off of.
    """
    def _ids():
        with file_node.stream() as stream:
            text = stream.read().decode('utf8')
        return {line.rstrip() for line in text.splitlines() if line.strip()}
    _ids.depends_on = tuple(source_resources(file_node))
    return _ids
