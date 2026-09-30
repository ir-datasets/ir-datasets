"""``freeze``: run a provider's modules once, at author/CI time.

Imports a provider's package, collects the nodes the modules registered, and
writes that provider's manifest -- everything a user needs *without importing
those modules*:

* the node catalog (names, types, metadata)
* the edges it contributes (adjacency list), incl. edges about other providers' nodes
* each node's attestation (``Node.attest``): counts, hashes, ...
* its generators, and the vocabulary it owns (types, edge kinds)

Two tiers:

    freeze                 cheap attestations only; no materialization
    freeze --verify        ``attest(verify=True)``: tables are iterated to
                           record counts, content hashes and sample records

Attestations are additive: a parametric family records theirs lazily on first
materialization instead, which is why verification is advisory and never a
build-time gate.

    python -m ir_datasets.v2.freeze --verify           # the irds provider
    python -m ir_datasets.v2.freeze acme.graph:acme     # anyone else's
"""
import argparse
import importlib
import json

from .graph import default_graph

MANIFEST_VERSION = 2

#: Fields an earlier manifest wrote that nothing writes any more.
RETIRED_FIELDS = {'hashes', 'size', 'validation', 'sources', 'dua'}

#: The provider a bare (no positional arg) ``freeze`` call targets.
DEFAULT_PROVIDER = 'ir_datasets.v2.provider:irds'


def row_for(node):
    """A node's manifest row. Empty values are dropped so rows stay readable."""
    row = {'name': node.qualified_name, 'type': node.type,
           'module': node.defined_in, **node.metadata}
    return {k: v for k, v in row.items() if v or k == 'name'}


def _attested_row(node, graph, verify, options):
    row = row_for(node)
    attestation = node.attest(verify=verify, **options)
    if attestation is None:
        # Keep what an earlier freeze recorded (a table not re-verified this
        # run) rather than dropping it.
        previous = graph.frozen(node.qualified_name)
        # Retired manifest fields (``hashes``; ``size``, ``validation``, ``sources`` and ``dua`` are
        # now reported at discovery, see ``Node.discovery_literals``) are not
        # attestations: never carry a stale copy forward.
        attestation = {k: v for k, v in previous.items()
                       if k not in row and k not in RETIRED_FIELDS}
    row.update(attestation)
    return row, bool(attestation)


def build_manifest(provider, graph=None, *, verify=False, only=None, **options):
    graph = graph or default_graph()
    provider.import_all()
    nodes = {}
    for name, node in sorted(provider.nodes.items()):
        if name in provider.generated:
            continue  # produced by a generator: the rule is frozen, not the expansion
        wanted = verify and (only is None or name in only)
        nodes[name], _ = _attested_row(node, graph, wanted, options)
    return {
        'version': MANIFEST_VERSION,
        'provider': provider.prefix,
        'nodes': nodes,
        'edges': provider.edge_rows(),
        'generators': [g.metadata() for g in provider.generators],
        'types': dict(sorted(provider.types.items())),
        'edge_kinds': dict(sorted(provider.edge_kinds.items())),
        'defaultable': sorted(provider.defaultable_fields),
    }


def load_provider(spec):
    """'pkg.module:attr' -> the Provider object."""
    module, _, attr = spec.partition(':')
    return getattr(importlib.import_module(module), attr or 'provider')


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='ir_datasets.v2.freeze', description="Generate a provider's manifest.")
    parser.add_argument('provider', nargs='?', default=DEFAULT_PROVIDER,
                        help=f"module:attr of the Provider to freeze (default: {DEFAULT_PROVIDER})")
    parser.add_argument('--verify', action='store_true',
                        help='materialize nodes to record counts/hashes/samples')
    parser.add_argument('--only', nargs='*',
                        help='restrict --verify to these (qualified) node names')
    parser.add_argument('--offline', action='store_true',
                        help='no network for attestations (cache/frozen only)')
    parser.add_argument('--out', help="output path (default: the provider's manifest_path)")
    args = parser.parse_args(argv)

    provider = load_provider(args.provider)
    graph = default_graph()
    graph.add(provider)
    manifest = build_manifest(provider, graph, verify=args.verify,
                              only=set(args.only) if args.only else None,
                              offline=args.offline)
    out = args.out or provider.manifest_path
    if out is None:
        parser.error(f'{provider} has no manifest_path; pass --out')
    with open(out, 'w') as fout:
        json.dump(manifest, fout, indent=2, sort_keys=False)
        fout.write('\n')
    print(f'wrote {out}: {len(manifest["nodes"])} nodes, {len(manifest["edges"])} '
          f'edges, {len(manifest["generators"])} generators')
    # Dangling/cycle checks run over the whole graph: an edge from this provider
    # to another's node is validated against that provider.
    for problem in graph.check():
        print(f'GRAPH: {problem}')


if __name__ == '__main__':
    main()
