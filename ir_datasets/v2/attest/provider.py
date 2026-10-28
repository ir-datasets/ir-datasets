"""The ``attest:`` provider: attestations held by a server, as nodes in the v2 graph.

Discovery downloads the whole attestation graph: every attestation's type, record, signature and
signer, and an edge from each attested node to its attestations. Records never change, so the copy
on disk (``<ir_datasets home>/attest/cache.json``) is kept up to date incrementally -- once an
hour at most, only what is newer than the last record seen is fetched. A server that is down
falls back to the copy on disk (with a warning); ``IR_DATASETS_ATTEST_OFFLINE=1`` never touches the
network.
"""
import json
import logging
import os
import time

from ..registry import ManifestProvider, row_triples

_log = logging.getLogger(__name__)
CACHE_TTL = 3600
_FETCH_TIMEOUT = 15


def _cache_path():
    from ...util import home_path
    return home_path() / 'attest' / 'cache.json'


class AttestProvider(ManifestProvider):
    def __init__(self, prefix='attest'):
        super().__init__(prefix, package=None, distribution='ir_datasets')
        #: Replaceable (tests, mirrors): anything with ``records(after)`` and ``get(*ids)``.
        self.client = None
        self._rows = None      # id -> record row, for this process

    def _client(self):
        if self.client is None:
            from .client import Client
            self.client = Client(timeout=_FETCH_TIMEOUT)
        return self.client

    def refresh(self):
        """Forget what is loaded and fetch again from the server on next use. Keeps the on-disk copy,
        so only what is new is downloaded."""
        self._rows = None
        self.nodes = {}
        try:
            cached = json.loads(_cache_path().read_text())
            cached['fetched'] = 0
            _cache_path().write_text(json.dumps(cached))
        except (OSError, ValueError):
            pass

    def rows(self):
        """``{id: row}`` for every attestation (see the module docstring for how it is kept)."""
        if self._rows is not None:
            return self._rows
        path, url = _cache_path(), self._client().url
        try:
            cached = json.loads(path.read_text())
            if cached.get('server') != url:
                cached = None
        except (OSError, ValueError):
            cached = None
        rows = cached['rows'] if cached else {}
        stale = not cached or time.time() - cached['fetched'] >= CACHE_TTL
        if stale and not os.environ.get('IR_DATASETS_ATTEST_OFFLINE'):
            try:
                after = max((r['seq'] for r in rows.values()), default=0)
                rows = {**rows, **{r['id']: r for r in self._client().records(after)}}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'fetched': time.time(), 'server': url, 'rows': rows}))
            except Exception as e:     # a down server must not break listing every other dataset
                _log.warning('attest: could not reach %s (%s); using %s', url, e,
                             'the copy on disk' if cached else 'nothing')
        self._rows = rows
        return rows

    def for_node(self, name):
        """The attestations about a node (``name`` qualified), as loaded nodes."""
        return [self[f'{self.prefix}:{i}'] for i, r in self.rows().items() if r['subject'] == name]

    # -- protocols.Provider ----------------------------------------------------

    def load(self, name):
        return self[name]

    def __getitem__(self, name):
        from .nodes import Attestation
        name = self.qualify(name)
        if name not in self.nodes:
            guid = name.split(':', 1)[1]
            row = self.rows().get(guid)
            if row is None:       # newer than the copy we hold?
                try:
                    found = self._client().get(guid)
                except Exception:
                    found = []
                row = found[0] if found else None
            if row is None:
                raise KeyError(f'no such node: {name!r}')
            self.register(Attestation.from_row(row), module='ir_datasets.v2.attest.nodes')
        return self.nodes[name]

    def discover_edges(self):
        from .nodes import Attestation
        for guid, row in self.rows().items():
            node = Attestation.from_row(row)
            name = f'{self.prefix}:{guid}'
            fields = {'name': name, 'type': node.type, 'record': node.record}
            if node.signature:
                fields.update(signature=node.signature, verified=str(node.verified).lower())
            yield from row_triples(name, fields)
            yield from node.attestation_edges()
        for qualified, entry in self.graph_classes.items():
            if entry.get('parent'):
                yield qualified, 'rdfs:subClassOf', entry['parent']
        yield from self.property_triples()


attest = AttestProvider()

# Declares the vocabulary (graph classes and properties) as soon as the provider is discovered.
from . import nodes  # noqa: E402,F401
