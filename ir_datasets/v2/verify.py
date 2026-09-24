"""Verification against frozen attestations.

Deliberately advisory, not a gate: ``verify`` reports divergence rather than
raising. Attestations are observations of what materializing a node yielded,
and legitimate divergence happens (a mirror changed, an access tier differs, a
parser improved). A caller decides what to do about it.

What an attestation *is* belongs to the node type (``Node.attest`` /
``Node.verify``, e.g. ``nodes.Table``); this module only drives it, bound to
the installed graph.
"""


class Divergence:
    def __init__(self, node, field, expected, actual):
        self.node = node
        self.field = field
        self.expected = expected
        self.actual = actual

    def __repr__(self):
        return (f'{self.node}: {self.field} expected {self.expected!r}, '
                f'got {self.actual!r}')


def attested():
    """Names of every node with a content attestation."""
    from . import graph
    return sorted(n for p in graph.providers.values()
                  for n, row in p.manifest()['nodes'].items()
                  if 'content_sha256' in row)


def verify(name, **options):
    """Materialize a node and compare against its frozen attestation.

    Returns a list of Divergence (empty == matches, or nothing to compare).
    """
    from . import graph
    frozen = graph.frozen(name)
    if not frozen:
        return []
    return graph[name].verify(frozen, **options)


def verify_all(names=None, **options):
    """``{name: [Divergence, ...]}`` for every attested node (or ``names``)."""
    return {name: verify(name, **options) for name in (names or attested())}
