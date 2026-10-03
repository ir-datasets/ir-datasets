# ir_datasets v2 prototype

A working sketch of the knowledge-graph architecture, layered over the existing
v1 machinery (`ir_datasets.util` downloads, `ir_datasets.formats` parsers,
`ir_datasets.indices` docstores) so the design can be exercised on real data
without reimplementing any of it. **v1 is untouched.**

This one package holds both the generic graph machinery (`Node`, `Edge`,
`Provider`, `Graph`, `Generator`, `freeze`/`verify`) and the dataset vocabulary
built on it. The two used to be split across two packages (`ir_graph` +
`ir_datasets.v2`); they were merged back into one once it was clear this paper
only needs one domain — see [Deferred to a later
paper](#deferred-to-a-later-paper) below. The generic machinery still declares
no vocabulary of its own — it's `nodes.py` that declares nine node
types, all owned by the `irds` provider:

- **`Resource`** — bytes, and where to get them
- **`Table`** — a structured set of records, parsed from a source — and its
  five per-entity subtypes, `DocTable`/`QueryTable`/`QrelTable`/`RunTable`/`DocPairTable`,
  each its own declared graph type (a subtype of `Table`, not just the same
  `Table` type distinguished by an `.entity` attribute) — so a type-filtered
  listing can ask for either the specific kind or the whole family
- **`Benchmark`** — docs + queries + qrels (etc.) bundled into an evaluable task,
  plus flat metadata (`citation`, `metrics`)
- **`Suite`** — a named, structural set of related Benchmarks (e.g. BEIR)

Citations-as-papers and metrics-as-measures (each with their own node type and
edges) are explicitly out of scope for this iteration — see
[Deferred to a later paper](#deferred-to-a-later-paper). For now both are
plain metadata.

The design still decentralizes: a third-party package can declare its own
provider — anything satisfying the two-method `Provider` protocol
(`load(name)`, `discover_edges()`); `ManifestProvider` is the batteries-included
implementation most datasets use, not a requirement — its own node types and
edge kinds, and its own entry point in the `ir_datasets.providers` group —
pointing at `irds`'s nodes or vice versa — without either package importing
the other until a node is actually resolved.
There is no default provider: every name has a home, and the home is in the
name — `irds:antique-test`, type `irds:Benchmark`, edge kind `irds:derived_from`.
Types are capitalized (RDF/OWL convention: classes are UpperCamelCase);
edge kinds stay lowercase (properties aren't).
Legacy v1 ids (`antique/test`) resolve as explicit aliases.

ANTIQUE is implemented end to end as the worked example:
[`datasets/antique.py`](datasets/antique.py). MS MARCO (passage) is a second,
larger worked example: [`datasets/msmarco_passage.py`](datasets/msmarco_passage.py)
— multiple corpora-sharing benchmarks, derived (`/judged`, split200, medical)
subsets via `Filter`, and a custom `@transform` pipeline step (fixing the
collection's known double-encoding bug, and reducing "qid did qtext dtext"
scoreddocs files to plain qid/did pairs). It does not cover every v1 subset:
`train/triples-small` (needs a doc/query text->id hash-join, not just a
stream transform), `trec-dl-hard` and its folds (queries drawn from two prior
years plus an external fold list), and `dev/2` (its ids come from
`msmarco-passage-v2`, not yet a v2 family) are left for a future pass.

MS MARCO (document), [`datasets/msmarco_document.py`](datasets/msmarco_document.py),
is a third worked example, and the first case of **cross-file shared
reference**: its TREC-DL 2019/2020 benchmarks reuse the *same QueryTable
objects* `msmarco_passage.py` already registered (`trec_dl_2019_queries`,
imported directly, not re-declared) — one node, two independent Benchmarks in
two different files, exactly the "docs/queries/qrels shared by reference"
story extended across dataset families instead of within one. `trec-dl-hard`
and `anchor-text` are left out for the same reasons as their passage-side
counterparts.

BEIR, [`datasets/beir.py`](datasets/beir.py), is the worked example the
`Suite` node type was designed around: 14 headline zero-shot benchmarks plus
CQADupStack's 12 sub-forums, folded into one `irds:beir` suite rather than
split into per-subgroup suites. NanoBEIR mirrors 13 of BEIR's headline
benchmarks at a smaller scale (`datasets/nano_beir.py`).

BRIGHT, [`datasets/bright.py`](datasets/bright.py), is a fourth worked
example, and the first case of **one Resource serving two Tables**: its
qrels are embedded in the same parquet file as its queries (not a separate
qrels file), so `QueryTable` and `QrelTable` both declare that file as their
`source=` rather than one deriving from the other. It's also a second case
of cross-benchmark shared reference within one file: each of the 8 "long"
document variants reuses the short variant's *same* `QueryTable` object,
differing only in docs (long-form) and qrels (re-derived from the queries
file's `gold_ids_long` field). The 12 short benchmarks form the `irds:bright` suite and the 8 long
variants their own `irds:bright-long` suite.

## Try it

```bash
python -c "import ir_datasets.v2 as v2; print(v2.list_datasets()); print(v2.load('irds:antique-test'))"
```

```bash
python -m ir_datasets.v2.freeze --verify
```

```bash
python -m unittest test.v2_conformance
```

## Layout

| Module | What it holds |
|---|---|
| `protocols.py` | `Node`, `Resource`, `Table`, `Benchmark`, `Suite` as `typing.Protocol`s — the actual contract; no inheritance required to satisfy them |
| `base.py` | `Node` (a convenient, optional base class implementing the protocol above), `Edge`, `Generator`, `Param` — generic, no vocabulary of their own |
| `vocabulary.py` | The process-wide index of declared node types / edge kinds, and which providers own them |
| `context.py` | The `defaults()` stack (`ManifestProvider.defaults()` pushes onto it) |
| `protocols.py` (`Provider`) | The two-method contract a package joins the graph with: `prefix`, `load(name)`, `discover_edges()` — not a class to inherit |
| `registry.py` | `ManifestProvider` — one (batteries-included) implementation of `Provider`: nodes, edges, generators, aliases, manifest, and the vocabulary it declares |
| `graph.py` | `Graph`, `default_graph()`, `discover()` — the union over installed providers, entry-point discovery; touches a provider through nothing but `Provider`'s two methods |
| `freeze.py` | `python -m ir_datasets.v2.freeze [provider] [--verify]` — writes a `ManifestProvider`'s manifest |
| `verify.py` | `Divergence`, `verify()`, `verify_all()` — advisory checks against a frozen manifest, bound to the installed graph |
| `provider.py` | The `irds` `ManifestProvider` instance itself |
| `nodes.py` | `Resource`, `Table` (+ `DocTable`/`QueryTable`/`QrelTable`/`RunTable`/`DocPairTable`), `Benchmark`, `Suite` — the dataset vocabulary, declared on `irds` |
| `formats.py`, `sources.py`, `filters.py` | Format subclasses, download/cache machinery, derivation (`Filter`) |
| `log_utils.py` | `log_download` — the generic, provider-agnostic download-event log every `Resource` fetch appends to (`Resource.path`/`.stream`) |
| `datasets/*.py` | One authored file per dataset family |

## What it demonstrates

| Decision | Where |
|---|---|
| Flat names; no hierarchy (`antique-test-qrels`, not `antique/test/qrels`) | `datasets/antique.py` |
| **No default provider**: nodes, node types, edge kinds all `prefix:name`; legacy ids are explicit aliases | `Graph.provider_for` |
| Node types, edge kinds and defaultable fields are **owned by a provider** (`irds.node_type('Table')`, `irds.edge_kind('derived_from', structural=True)`, `with irds.defaults(...)`) — types capitalized (RDF/OWL convention), kinds lowercase; the generic machinery declares none of its own | `nodes.py` top, `vocabulary.py` |
| **One `derived_from` kind covers both "parsed from these bytes" and "filtered from this other table"** — a table's dependency on a `Resource` and its dependency on a parent `Table` are the same shape of relationship (this node would not exist without that one), so there's no separate `source` kind | `nodes.Table.__init__`, `filters.DerivedTable` |
| **Nine node types**: `Resource`, `Table`, `Benchmark`, `Suite`, and one `Table` subtype per entity — `DocTable`/`QueryTable`/`QrelTable`/`RunTable`/`DocPairTable` — each its own declared graph type, a subtype of `Table` (`.entity` is still a plain attribute, used for format-specific behavior) | `nodes.py` |
| First-class **edges**; structural (DAG) vs informational split | `base.Edge` |
| Nodes declare `structural_edges()`; a `ManifestProvider` stores edges (fwd + indexed reverse) | `ManifestProvider.register` |
| Any provider may add *informational* edges about any node — they ship in *its* manifest; *structural* edges are the owner's alone | `ManifestProvider.add_edge` |
| Graph traversal without imports, from each provider's `discover_edges()`, cached per `Graph` | `Graph` |
| Every node named and directly loadable — files and tables included, every registered node listed (no hidden/internal tier) | `nodes.py`, `Graph.list` |
| Single-type nodes; facets by reference, never multi-type | `Benchmark.edge` |
| **`Provider`** is a two-method protocol (`load`, `discover_edges`), not a registry to inherit; **`ManifestProvider`** is the batteries-included registry most datasets actually use — `irds.register(*roots)` is explicit and pulls in dependencies | `protocols.Provider`, `registry.ManifestProvider` |
| **`Graph`** = the union over installed providers (entry points + on-import subscription): routing, indexed reverse lookups, validation — built entirely from `Provider.load`/`discover_edges`, nothing more | `graph.Graph` |
| Construction is inert — ad-hoc/test nodes never touch any registry | `base.Node` |
| **Node/Resource/Table/Benchmark/Suite are `Protocol`s, not base classes** — `ManifestProvider.register()` only ever checks the shape (a `hasattr` check on the three methods it calls), never `isinstance` against a concrete class; a node needs no particular ancestor | `protocols.py`, `TestV2Protocols` |
| A Resource's `sources` accept bare URLs or `Source`/`Source.mirror()`/`Source.external()` for headers, auth, or manual acquisition | `sources.py` |
| **Multi-algorithm integrity**: `hashes=['sha256:...', 'md5:...']` (OCI/pip's `algo:hexdigest` convention; `md5=` stays as shorthand for the common single-hash case). Checked in one streamed pass at `verify` time, against the *author's declared* value — not a frozen row, unlike `Table` | `nodes.parse_hash`, `Resource.verify`, `TestV2ResourceHashes` |
| Declarative stream pipeline (`.member().gunzip().pipe()`) over a `Readable` — distinct from a Resource's `Source`s (where to fetch bytes) | `sources.Readable` |
| `derived_from` + `Filter` replaces hand-built Filtered\* stacks | `filters.py` |
| Parametric / dynamic families as `Generator` (rule, not expansion) | `base.Generator` |
| `freeze` replaces `docs/*.yaml`, `etc/downloads.json`, `etc/metadata.json`, and per-dataset tests; **what an attestation is belongs to the node type** (`Table.attest/verify`) | `freeze.py`, `nodes.Table` |
| Lazy, additive verification registry — advisory, not a gate | `verify.py` |
| **`Suite`**: a named, structural set of Benchmarks (e.g. BEIR); membership pulls its benchmarks (and their tables/files) in transitively, and a benchmark may belong to more than one suite | `nodes.Suite` |
| **`citation`** is a generic field on `Node` — any node can carry one — and `metrics` is plain metadata on `Benchmark`; neither is an edge (yet) | `nodes.Benchmark` |
| `Table` is the base type of docs/queries/qrels/… (schema + rows + key) | `nodes.Table` |
| v1's "beta" API is the **default** (`len(ds.docs)`, `ds.docs[:10]`, `ds.docs.lookup(...)`); legacy methods still provided | `nodes.Table` |
| Import-free discovery: manifest rows, one module imported per `load()`; derived facets register with their benchmark (no special case) | `ManifestProvider.__getitem__`, `Benchmark.structural_edges` |
| Legacy v1 ids as permanent aliases, discovered as an `irds:alias` edge like any other — no separate alias API on `Graph` | `ManifestProvider.alias`, `registry.ALIAS_KIND` |
| Existing caches read **in place** — nothing moved, nothing re-downloaded | `cache_path` in `datasets/antique.py` |
| Every actual fetch (never a cache hit) appends one entry — timestamp, node name, the *actual* md5 of the bytes that landed, and where on disk — to one shared, append-only, gzip'd JSON-lines audit log under the ir_datasets home directory, regardless of provider | `log_utils.log_download`, called from `Resource.path`/`.stream` |

## One authored file replaces four

ANTIQUE in v1 needed:

```
ir_datasets/datasets/antique.py      wiring
ir_datasets/docs/antique.yaml        desc / bibtex
ir_datasets/etc/downloads.json       urls / md5 / size       <- central file
test/integration/antique.py          counts / sample records
```

In v2 there is one authored file, plus `manifest.json` generated by `freeze`.
The two central files are gone, which is what makes extension packages possible
at all — a third-party package cannot edit core's shared JSON.

## Known gaps

This is a prototype, not a migration. Not yet addressed:

- Citations-as-papers and metrics-as-measures (their own node types, resolved
  and validated, linked by edges) are deferred to a later, broader knowledge
  graph paper — see [Deferred to a later paper](#deferred-to-a-later-paper).
- `Generator` exists but has no live example in this scope (no dynamic/
  parametric family is currently registered) — kept for the motivating cases
  this paper doesn't need yet (a ClueWeb-style family, a HuggingFace wrapper).
- The manifest is JSON; a larger deployment would want SQLite for faceted
  queries over thousands of nodes, plus a merged per-user `registry.db`.
- Entry points are declared (`pyproject.toml`, group `ir_datasets.providers`)
  and `discover()` reads them, but this checkout is not installed, so the
  `irds` provider reaches the default graph via the on-import subscription in
  `ir_datasets/v2/__init__.py` instead. Both paths are meant to coexist.
- No XDG cache-home move (`$XDG_CACHE_HOME/ir_datasets`) and no legacy-path
  resolver; `cache_path` currently points at v1 locations explicitly.
- `Filter` covers `queries_with_qrels` and explicit id sets; no general
  predicate, no `Concat`, no doc-level views (`html_extractor`).
- Derived facets are auto-named `<benchmark>-<entity>`. Worth deciding whether
  that naming should be explicit. A derived benchmark whose parent is given as
  a *name* resolves it at registration (one import of the parent's module).
- No CLI (`export`/`lookup`/`list`) over v2 nodes.
- No `Suite` example over real data yet (BEIR would be the natural first one);
  `Suite` is exercised only by synthetic nodes in the test suite so far.

## Deferred to a later paper

An earlier iteration of this prototype was structured as two packages: a
generic `ir_graph` kernel that knew nothing about datasets, and this package
subscribing to it — motivated by other domains (papers, evaluation measures,
runs) eventually becoming providers too, in a broader knowledge graph. That
iteration also modeled papers and measures as their own node types (`Article`,
`Measure`), with citations and official measures as edges to them (a `dblp:`
dynamic provider fetched BibTeX on demand; an `ir-measures:` provider resolved
measure specs).

Both the two-package split and the richer citation/measure model are future
work, not this paper's scope — see git history for a working reference
implementation to build on when that broader graph is in scope. For now,
everything lives in this one package, and citations/metrics are plain
metadata.
