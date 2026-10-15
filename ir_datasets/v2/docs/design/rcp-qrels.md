# RcpQrels: calibrated-probability relevance labels (Rubric-Calibrated Preferences)

Status: NanoBEIR part implemented (`datasets/rcp.py`, `formats.RcpQrels`). BRIGHT, TREC-DL, ViDoRe v3 and
the human-grade repo are not.

Departures from the design below, after checking the real data:

- **Qrels only.** The HF repo's corpus and queries were compared with `nano-beir`'s for all 13 tasks: same
  ids, same text, titles all empty. So benchmarks reuse the `nano-beir-<task>` docs and queries and add only
  the qrels. Nodes: `nano-beir-<task>-rcp-qrels`, `nano-beir-<task>-rcp`, suite `nano-beir-rcp`.
- **Record** is `RcpQrel(query_id, doc_id, relevance, theta)`. The source's integer `score` (the original
  human label, already in `nano-beir`) is not carried.
- The "Unknowns" below about the parquet schema and the NanoBEIR license are resolved for NanoBEIR (card
  states CC-BY-4.0 for the derived tasks); the other repos remain unchecked.

## Context

RCP (Schmidt, Crisostomi, Lassance, Reimers, Cohere; arXiv 2609.35739) is an LLM-judge labeling method.
It releases **no new documents or queries**. It labels the top-150 candidate pool of existing benchmarks
(13 NanoBEIR, 12 BRIGHT, 8 ViDoRe v3 corpora, TREC-DL 2019/2020) with:

- `g` in [0,1], the calibrated pass probability used as the gain in RCP-nDCG, and
- `theta`, the calibrated latent ability on a logit scale shared across queries.

Data is hosted on the HF Hub, not in the GitHub repo (`cohere-ai/rcp-ndcg`, Apache-2.0 for the code; each
dataset card states its own data license). Five repos: `rcp-ndcg-{nanobeir,bright,vidore-v3,trecdl}` plus
`rcp-ndcg-external-validation` (7,080 human grades over 311 contests, plus external judge outputs).

## Why this needs a new record type

No existing qrel type in v1 or v2 has a non-integer `relevance` (surveyed: `TrecQrel`, `GenericQrel`,
Touche/ClueWeb12/Highwire/NQ/CodeSearchNet variants; all int or, formerly, a mis-typed str).
`TrecPrel.iprob` is the only float on a qrel-like record and it is a sampling weight, not a label.
`GenericScoredDoc.score` is a float but belongs to the `RunTable` entity. So:

- Reusing `TrecPrel` would mislabel `g` as an inclusion probability. Do not.
- Reusing `RunTable` is wrong: these are judgments of a fixed pool, not a system's output, and the
  entity decides which graph type `list(type=...)` returns.

## Design

### Record

```python
class RcpQrel(NamedTuple):
    query_id: str
    doc_id: str
    relevance: float      # g(theta): calibrated pass probability in [0, 1]
    theta: float          # calibrated ability on the shared logit scale
```

`relevance` keeps the `QrelTable` contract (the field name `score_field` and downstream tools look for),
but its meaning is a probability. `theta` is a second signal, kept because `g` saturates at 0/1 and
`theta` preserves ordering there (Proposition 3.1 in the paper).

Open: whether to expose `theta` as a field on every row or only when the released files carry it. See
"Unknowns".

### Table node

`RcpQrels(QrelTable)` in `ir_datasets/v2/formats.py`, next to `TrecQrels`, with a `Parser` that builds a
v1-style handler (same wrapper pattern as `_TrecQrelsParser`). A small v1 handler class,
`RcpQrels(BaseQrels)`, goes beside `NanoBeirQrels` (it iterates the parquet and yields `RcpQrel`;
`qrels_cls()` returns `RcpQrel`; `qrels_defs()` returns `{}`).

Two overrides on the table:

- `score_field = None`. `freeze --verify` tallies `score_counts` per distinct value
  (`nodes.py:884-902`). A dense float field makes one bucket per qrel, which is noise. Binning is a later
  refinement, not needed now.
- `defs` stays empty. A continuous scale has no grade definitions. The node `desc` states the range.

### Edges, not names

Per the v2 naming rule (names do not encode lineage), tables are named for what they are, with the
relationship carried by edges:

| Node | Type | derived_from |
|---|---|---|
| `rcp-nanobeir-<task>-qrels` (x13) | `RcpQrels` | the matching `nano-beir/<task>` docs + queries |
| `rcp-bright-<task>-qrels` (x12) | `RcpQrels` | the matching `bright/<task>` docs + queries |
| `rcp-vidore-v3-<corpus>-qrels` (x8) | `RcpQrels` | the matching ViDoRe v3 corpus |
| `rcp-trec-dl-<year>-qrels` (x2) | `RcpQrels` | `trec-dl` passage docs + that year's queries |

Name stems are provisional and need checking against `nano_beir.py`, `bright.py` and `trec_dl.py`, plus
whatever ViDoRe v3 coverage v2 has (I did not confirm one exists).

Each pairs with its source docs/queries as a `Benchmark` (docs, queries, qrels) through the same
`Benchmark(...)` pattern used in `nano_beir.py`, so `irds:<name>` resolves to a runnable
docs+queries+judgments triple. An aggregate `Suite` (`rcp-ndcg`) groups them.

### Sources

Each table's source is the HF repo's parquet, as a `Resource`. Two options:

1. **Direct `File` nodes** with url + md5 + size per parquet, as `nano_beir.py` does (`SOURCES` dict).
   Pinned, reproducible, plugs into the existing `freeze` path. **Recommended.**
2. **`HfDataset`** (a `GitRepo` at a pinned commit, `hf_provider.py:248`) and `.file(path)` into it. Less
   to transcribe, and commit pinning gives identity for free. Worth it if the five repos have many small
   files, since `snapshot_download` fetches the whole repo.

Either way, pin to the revision the paper's `experiments/fetch_data.py` uses rather than `main`.

### Non-trivial cases

- **BRIGHT:** the paper removes each query's excluded documents from every ranking. The labels are over
  the post-exclusion pool, so a consumer needs the same exclusion applied. State this in the node
  `desc`, and check that v2's BRIGHT node already exposes exclusions.
- **NanoBEIR/ArguAna:** the query's own argument is removed from rankings. Same handling.
- **TREC-DL:** the pool joins the 150 retrieved candidates with every NIST-judged passage, so it is
  larger than 150 per query (about 287 on average). Do not assume `K=150`.
- **ViDoRe v3:** documents are page images; the judge read OCR text. Doc ids must match whichever v2
  ViDoRe node the labels attach to. If no such node exists, that corpus is deferred.
- **Judge identity:** the primary judge is Qwen3.5-397B, and gpt-oss-120b (and, for TREC-DL, Qwen3.6-27B)
  are alternates. If the repos ship several, make one table per judge and put the judge in the `desc`.
  Do not hide it in the name.

### Evaluation

Standard integer-grade tooling will not apply `g` as a gain. This design delivers the labels only. RCP-
nDCG itself stays in `cohere-ai/rcp-ndcg`. A follow-up could document an `ir_measures` recipe, or ship
a thin wrapper that exposes `RcpQrels` as float qrels.

### Human grades (`rcp-ndcg-external-validation`)

Out of scope for the first pass. It is a different shape (annotator contests, per-annotator grades), so
it likely needs its own record type and probably a `QrelTable` per-annotator or a dedicated node. I'd
add it after the core tables land, and only if the benchmark use case (validating future LLM judges) is
wanted in the catalog.

## Unknowns to resolve before implementing

1. **Parquet schema.** I could not see the column names or whether `theta` is released. The repo's
   `docs/data.md` says it describes the layout. This decides the record shape and the parser.
2. **Per-dataset licenses.** Each dataset card states its own, and the arXiv metadata says CC-BY-SA-4.0.
   v2 records licenses per node (see the "license verified" comments in `trec_cast.py`), so each
   needs checking individually, not inferred from the GitHub repo.
3. **Pool membership.** Whether judged-but-unlabeled documents are distinguishable. For nDCG semantics,
   unlabeled means gain 0, but that differs from "not in the pool".
4. **Which judges are released** (one, or several), and at which pinned commit.
5. **ViDoRe v3 and BRIGHT coverage in v2.** Not confirmed.

## Implementation order

1. Read `docs/data.md` and the dataset cards. Settle the record fields and licenses.
2. Add `RcpQrel` + v1-style handler + `RcpQrels` table (`formats.py`).
3. `datasets/rcp_ndcg.py`: NanoBEIR tables first (the docs/queries already exist), then BRIGHT, then
   TREC-DL, then ViDoRe if a docs node exists.
4. `freeze` and `freeze --verify`, and add the manifest entries.
5. Unit test the parser on a tiny fixture. Skip integration downloads unless asked.
