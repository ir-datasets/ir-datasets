# Attestation Registry: a public, multi-party validation layer on top of the v2 graph

## Context

The v2 graph already has one layer of trust: `Node.attest()`/`freeze`/`freeze --verify` produces
`manifest.json`, a **single attester's** (whoever ran freeze), **single point-in-time** record, PR-reviewed
alongside the provider's own code. `verify.py`/`Divergence` compares a live materialization against that
one frozen record — advisory, never a gate. Project memory already commits to the underlying philosophy:
*"No content-hash identity — too hard to get perfect. Instead a registry of verifications: additive
attestations of what materializing a dataset yields... warn on mismatch, don't hard-fail; surface
divergence across attesters."*

What's missing is the multi-party piece: a second person, lab, or bot re-materializing a dataset and
reporting what they got — hash confirmations, or effectiveness numbers from running a pipeline against a
benchmark — without needing commit access to the provider's repo. Sean wants to design this as a new,
additive layer, with anonymous and signed submission, extensible to new kinds of attestations, and possibly
reused for access/popularity counting. This plan is a **design document to discuss and refine**, not an
implementation task yet.

## Recommended design (see full agent report earlier in the session for detailed reasoning)

**Vocabulary**: call the whole thing the **Attestation Registry**, and a submission an **attestation** —
deliberately reusing `Node.attest()`'s existing word rather than coining a new one (Sean's call). The
existing `freeze`/`manifest.json` path is a **local attestation** (one attester, one point in time, baked
into the provider's own repo); this new layer holds **registry attestations** (many parties, out-of-band,
accumulating over time). Same underlying concept, different scope — not two competing vocabularies.

### 1. Envelope + payload
One flat JSON envelope for every kind (`schema`, `kind`, `target` [qualified_name], optional `target_ref`
[pins the specific hash/version being confirmed], `submitted_at`, `submitter` [anonymous or keyed],
`payload` [kind-specific], `signature`). `kind` is the sole extension point — new kinds need no envelope
or storage change. Payload vocabulary for `hash_confirmation` deliberately mirrors `Resource.attest`/
`Table.attest`'s own field names (`hashes_confirmed`, `content_sha256`, `hash_scheme`, `count`, `samples`)
so a thin adapter can feed a registry attestation into the *existing* `node.verify()` comparison code — but
the envelope itself is NOT literally local `attest()`'s return dict; it needs fields (`submitter`,
`signature`, `target_ref`) that don't belong in a manifest row, and conflating "registry wire format" with
"provider manifest row" would blur two different trust levels.

Three concrete kinds specified: `hash_confirmation`, `effectiveness_report` (targets a `Benchmark`;
pipeline/paper/metric provenance; deliberately has no "correct" value to diverge against — aggregation and
display, not comparison), `access` (shape shown for generality, but see popularity section — NOT
recommended as a signed-attestation mechanism in practice; a separate telemetry path instead).

### 2. Identity & signing — and how this actually reaches the graph
`attest` is a real **provider**, claiming the `attest:` prefix the normal way: **one, one-time PR to core**
to register the prefix (same governance as any other official namespace claim) — nothing per-identity or
per-attestation ever goes through review after that.

After the prefix is claimed, `attest` is a **live-resolver provider**, the same shape `hf_provider.py`
already uses for Hugging Face (nodes aren't statically enumerated in code; they're resolved on demand
against a live external source — there, the HF API; here, the attestation-registry's own database/API).
Concretely:
- An `Identity` node (`attest:sean-macavaney`) is dynamically resolved from a row someone created via a
  self-service "register a key" API call against the registry — never a package release, never a PR. New
  identity = new row in the registry's DB, immediately visible to anything that resolves the `attest`
  provider.
- Each `Identity` holds `keys: [{key_id, algo, public_key, added_at, revoked_at}]` — ed25519, multiple
  keys/rotation native, revocation doesn't retroactively invalidate old attestations (matches "advisory not
  a gate"). Signature covers the canonical envelope (reuse `nodes._canonical()` exactly — one
  canonicalization convention in the codebase, not two) minus the `signature` field itself.
- **Each attestation is reified as its own node** (Sean's design): `attest:<uuid>`, a new `Attestation`
  node type resolved live from the registry's stored rows. This is the piece that makes anonymous
  attestations fall out for free — no special-casing, no fake shared node needed:
  - `attest:<uuid> --attest:attests--> <target qualified_name>` — what the attestation is about (from the
    envelope's `target`).
  - `attest:[identity] --attest:attests--> attest:<uuid>` — present **only when signed**; the identity
    vouching for/authoring the attestation. Same edge kind reused in both positions (an attestation attests
    a target; an identity attests an attestation) — consistent with how `derived_from` is already reused
    across different node-type pairs elsewhere in the graph. An anonymous attestation simply has no
    incoming edge here — "signed or not" is then just "does this node have an inbound `attest:attests`
    edge from an `Identity`," not a separate flag to maintain.
  - The attestation node's own metadata carries `kind`, `schema`, `submitted_at`, `signature`/`key_id`
    (null if anonymous), and the kind-specific payload itself (hashes confirmed, effectiveness numbers,
    etc.) — **confirmed: kept as node metadata**, not a separate edge/literal graph target. Considered and
    rejected making payload a literal edge target (RDF-object-style): it would be the only place in the
    graph where an edge points at inline data rather than a qualified node, rippling through `edge_rows()`,
    the manifest edge schema, `graph.check()`'s dangling-edge logic, and webapp edge rendering — schema
    surface metadata already avoids entirely. The dedup/citability argument for a literal doesn't really
    apply either: attestation payloads are per-submission observations (naturally differ by timestamp/
    submitter), not shared downloadable bytes like a `Resource`'s content-addressed sources, where dedup
    actually pays off. Metadata is self-contained (one fetch gets everything) and consistent with every
    other node type in the graph.
  - `Attestation` nodes are **type-scoped out of default enumeration**, the same precedent already
    established for `File` — reachable by traversal (from a target's reverse edges, or an identity's
    forward edges), not listed by default. This is what keeps a popular dataset's full attestation history
    from bloating any default listing, without needing to pre-aggregate/summarize submissions the way an
    earlier version of this plan proposed (superseded — full fidelity per submission is fine once nodes are
    traversal-only).
  - Summary counts for a UI ("5 signed, 14 anonymous, 0 divergent") become ordinary graph traversal —
    reverse-edges into a target, then check which of those attestation nodes have an inbound signer edge —
    not a separately computed/maintained aggregate.
- Because `attest` is a provider like any other, **`ir-datasets.com` gets this for free** — the webapp
  already unions edges across every registered provider, so no new integration work is needed there
  specifically for this. `attest`'s own `freeze` just becomes "what the registry said as of this run" — one
  more point-in-time snapshot, same additive/advisory posture as everything else, not a new trust tier.

Trust is **multiplicity-based, not cryptography-based**: signed only proves attribution, not correctness.
Three tiers — anonymous (lowest, always shown), signed/unknown key (attributable, slightly higher), signed/
known-good key (a small curated, PR-reviewed `trusted_keys.json` allowlist — explicitly NOT web-of-trust,
which is over-engineering at this scale). No reputation-scoring algorithm; raw "N confirm, M diverge" is
more honest than a collapsed score.

Sybil resistance: light rate-limiting (per-IP, generous) plus an opaque, non-identifying, server-issued
session token for anonymous submitters, for `hash_confirmation`/`effectiveness_report`. Popularity counting
gets a **structurally different** mechanism (see below) rather than reusing this.

### 3. Storage & wire API
Not a git file in `ir_datasets` core (would force every third-party attestation through this repo's PR
review — defeats the point) and not `manifest.json`. Reference deployment: a small SQLite-backed service
(matches the README's own already-flagged gap: *"a larger deployment would want... a merged per-user
registry.db"*). Append-only (never mutable-latest-wins) — a correction is a new attestation, not an edit —
with content-addressed ids (`sha256` of the canonical envelope, dedup-friendly). Minimal REST surface:
`POST /v1/attestations`, `GET /v1/attestations?target=&kind=&since=&submitter=`,
`GET /v1/attestations/<id>`; `/v1/summary` is an optional convenience, not part of the required contract.

Deliberately borrows Sigstore/Rekor's *content-addressing and append-only* ideas, but explicitly skips
Merkle inclusion proofs / cross-log witnessing — that machinery defends against a malicious log operator
lying differently to different clients, which is more than this project needs for its own reference
deployment (see below — no federation for now, so there's currently only one operator to trust).

### 4. Popularity/access counting — explicitly NOT the same mechanism as attestations
Per-event signed attestations are close to the worst shape for this: privacy-sensitive (a timestamped
per-dataset access log tied to identity/IP even if "anonymous"), and the one signal with a real incentive to
game. Sean's call: **default ON** is acceptable (precedent: Hugging Face tracks downloads this way too),
*conditioned on the signal staying anonymous* — no per-event identity, no per-event timestamp/IP retained.
Recommendation: **separate, batched, low-trust-bar telemetry**, distinct from the attestation mechanism —
client aggregates load counts locally, flushes periodic anonymous batch totals to a distinct
`/v1/telemetry` endpoint, surfaced only as coarse relative signals, never presented with false precision.
Treat "stays anonymous" as a hard requirement when this piece is actually designed/built, not just a
preference. This is a deliberate descope from the attestation mechanism, not a smaller version of it —
bolting popularity onto per-attestation signing would bake in the wrong primitive and require a breaking
migration later.

### 5. Relation to `verify.py` (sketch only, not built now)
`Divergence` gains an optional `source` field (default `'frozen'` = today's local-manifest path; a registry
URL or key_id otherwise). A new `verify_against_registry()` fetches registry attestations, adapts `payload`
into the same shape `frozen` already has, and calls the *existing* `node.verify(frozen)` — zero new
comparison logic needed in `nodes.py`. Purely additive; the registry is independently useful without this
integration.

### 6. MVP (build first)
`hash_confirmation` + `effectiveness_report` kinds; anonymous (session-token rate-limited) + signed
(`Identity`/ed25519) submissions; SQLite-backed reference service with the `POST/GET /v1/attestations`
surface; a curated `trusted_keys.json`. Explicitly deferred: popularity/telemetry (different mechanism,
needs its own design, though default-on-anonymous is now decided in principle), key-revocation ceremonies
(manual PR-based revocation is adequate for now), `verify.py` integration, and any reputation-scoring
(permanently descoped, not just deferred).

## Decisions (2026-09-24)

1. **Identity & attestation representation**: `Identity` and `Attestation` are new graph node types,
   resolved dynamically by the `attest` provider (a live-resolver provider, like `hf_provider.py` — not
   statically enumerated). Registering a new identity is a self-service API call against the registry,
   never a PR or package release; only the `attest:` prefix itself was claimed via a one-time PR. Each
   attestation is **reified as its own node** (`attest:<uuid>`), with `attest:attests` edges to its target
   and, when signed, from the signing `Identity` into the attestation node itself — anonymous attestations
   simply lack that inbound edge, no special-casing needed. `Attestation` nodes are type-scoped out of
   default enumeration (same precedent as `File`) — reachable by traversal, not listed by default, which
   keeps popular datasets' full attestation history from bloating listings without needing to pre-summarize
   submissions.
2. **Terminology**: submissions are called **attestations** (reusing `Node.attest()`'s existing word), the
   whole layer is the **Attestation Registry**, and its provider prefix is **`attest`** — short and
   prefixable, matching the existing `prefix:name` convention (e.g. `attest:sean-macavaney` for a
   `Identity`). The `access_ping` kind is renamed **`access`**.
3. **No federation.** Single reference registry, no multi-registry/alternate-registry design for now —
   dropped from scope entirely rather than deferred as a "later" section.
4. **Telemetry default**: **default ON**, not off — Sean's call, citing Hugging Face's download-tracking as
   precedent, *conditioned on the signal staying anonymous* (hard requirement, not just a preference, when
   this piece is eventually designed/built).
5. **Repo location**: separate repo/package for the reference registry service, depending on `ir_datasets`
   for the `Node`/`Identity` vocabulary. Keeps `pip install ir_datasets` lightweight and decouples deploy
   lifecycles.
6. **`target_ref`**: optional, implicitly "current declared state" when absent — keeps anonymous/casual
   submission frictionless.

Still genuinely open (not asked yet, lower stakes / can be settled later): where `trusted_keys.json`
governance lives (registry-server repo vs. giving dataset-provider maintainers a say).

## Verification
This is a design/discussion task — no code changes were made. This plan file is the record of the design
and the decisions above; next step (separate future task) would be to turn this into an implementation
plan for the MVP (envelope + `hash_confirmation`/`effectiveness_report` kinds + `Identity` node type +
reference SQLite-backed service), likely in a new repo per decision #5.
