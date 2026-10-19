"""MS MARCO -- the family as one bundle.

The five corpora (passage and document, each in its v1 and v2 form, plus QnA) and every
benchmark defined over them are separate nodes in their own modules; this registers an
``Overview`` (a bundle with a summary) that holds them together and carries the summary of what they are and how they
relate. It adds no data of its own. (TREC Deep Learning judges these corpora against its own
query sets, so it stays in ``trec_dl.py``; the summary points to it.)
"""
from ir_datasets.v2 import Benchmark, DocTable, Overview, irds
# The members register themselves on import; the bundle is built from what they registered.
from ir_datasets.v2.datasets import (  # noqa: F401
    msmarco_document, msmarco_document_v2, msmarco_passage, msmarco_passage_v2, msmarco_qna,
)
from ir_datasets.v2.datasets.msmarco_passage import CITATION

SUMMARY = """\
MS MARCO (Microsoft MAchine Reading COmprehension) is a family of large-scale retrieval
datasets built from real queries that people sent to Bing. It is the standard training and
evaluation resource for neural ranking: the relevance judgments are sparse (usually one
passage or document per query, labeled by crowd workers), so a system is judged on whether it
ranks that one item near the top, and the usual measure is reciprocal rank at 10 (RR@10).

## What is in it

| Corpus | What it is | Node |
|---|---|---|
| Passage (v1) | About 8.8 million short passages (8,841,823) | [[msmarco-passage]] |
| Passage (v2) | A much larger passage corpus, roughly 138 million passages | [[msmarco-passage-v2]] |
| Document (v1) | About 3.2 million web documents | [[msmarco-document]] |
| Document (v2) | A larger document corpus, roughly 12 million documents | [[msmarco-document-v2]] |
| QnA | The question-answering release: passages, with human-written answers for the queries | [[msmarco-qna]] |

Each corpus has its own queries and judgments, split into the benchmarks listed under
Contents. The usual splits are `train`, `dev` and `eval`; the `eval` queries have no public
judgments (they are scored on the leaderboard), so they cannot be evaluated locally.

## Benchmarks you will meet

- **train**: the official training queries with their judgments. For the v1 passage corpus
  there are also training triples (a query with a relevant and a non-relevant passage),
  in two releases, and variants: only the queries that have a judgment (`-judged`), a
  200-query held-out validation split (`-split200-*`), and a medical subset.
- **dev** and **dev-small**: the official development set ([[msmarco-passage-dev]]), and the
  smaller 6,980-query set most papers report on ([[msmarco-passage-dev-small]]).
- **eval** and **eval-small**: the leaderboard queries (judgments hidden).
- **Document benchmarks**: judged by inheritance. A document counts as relevant if it
  contains a passage that was labeled relevant in the matching passage data, so these
  judgments are indirect.
- **v2 corpora**: `dev1` and `dev2` development sets. The v2 passage judgments are mapped over
  from the v1 data.
- **ORCAS** ([[msmarco-document-orcas]]): queries and judgments for the document corpus taken
  from user clicks, not from crowd workers.
- **QnA**: train, dev and eval splits whose queries also have answers.

## Things to know

- The data is distributed under Microsoft's data usage agreement, which you must accept
  before downloading. Each node records the agreement and the licence terms.
- Judgments are sparse: most relevant passages have no judgment, so effectiveness numbers
  are best read as comparisons between systems, not as absolute scores.
- The TREC Deep Learning track (2019 to 2023) reuses the passage and document corpora with
  its own, much more thoroughly judged queries; those benchmarks are defined separately, for
  example [[trec-dl-2019-passage]] and [[trec-dl-2019-document]].

Cite [[dblp:conf/nips/NguyenRSGTMD16|the original paper]] when using any of these.
"""

_members = [name for name, node in irds.nodes.items()
            if name.startswith('irds:msmarco') and isinstance(node, (DocTable, Benchmark))]

irds.register(Overview(
    'msmarco', parts=_members, pretty_name='MS MARCO',
    desc='MS MARCO: passage, document and QnA retrieval datasets built from Bing queries.',
    summary=SUMMARY, citation=CITATION))
