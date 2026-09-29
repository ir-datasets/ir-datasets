"""TREC CAsT (Conversational Assistance Track) -- a v2 dataset family, migrated
with **degraded provenance** by deliberate choice (see the "Known follow-up"
note this replaces in ``todo-v2.txt``).

Scope: v1's currently-registered subsets only -- ``v0``, ``v0/train``,
``v1``, ``v1/2019``, ``v1/2020`` (each with a ``/judged`` derived variant for
the two evaluation-year benchmarks and the training benchmark). v1's own
``v2``/``2021`` and ``v3``/``2022`` subsets are commented out in
``ir_datasets/datasets/trec_cast.py`` itself and are not migrated here either
-- there is nothing registered upstream to migrate.

The provenance tradeoff
------------------------
CAsT's corpora (``docs_v0`` = Washington Post v2 + MS MARCO passage (v1) +
TREC CAR v2.0; ``docs_v1`` = MS MARCO passage (v1) + TREC CAR v2.0) are built
by v1's own cross-dataset ``PrefixedDocs``/``PrefixedDocsSpec``/``LazyDocs``/
``IterDocs`` machinery (``ir_datasets.util.docs.multiple``/``.lazy``), which
prefixes doc ids per sub-corpus and dispatches lookups by prefix, calling back
into v1's own (not v2's) ``ir_datasets.load(...)`` for each piece -- including
a from-scratch WaPo-paragraph-splitting converter and two dedup files
(``wapo_dupes``/``marco_dupes``) that only make sense wired up exactly as v1
wires them.

Rather than reimplement that dispatch as a new v2 union/prefixed Table type
(rejected -- not worth it for one family) or hand-port the WaPo-paragraph
converter and dedup logic into a v2 pipeline (a lot of surface area to get
subtly wrong for a corpus most people can't even download, since v0 needs the
LDC/NIST-gated WaPo v2 file), this file reuses v1's already-registered
``trec-cast/v0``/``trec-cast/v1`` dataset objects *wholesale*: each v2
``docs`` Table's ``Parser`` just calls v1's own ``ir_datasets.load('trec-cast/
v0').docs_handler()`` (the plain v1 registry lookup, not the v2 graph -- no
circularity even once this module's own ``trec-cast/v0`` legacy id is aliased
back to the v2 node it produces) and returns whatever ``PrefixedDocs`` v1
already built.

The cost: these two ``DocTable`` nodes (``trec-cast-v0-docs``,
``trec-cast-v1-docs``) have **no structural ``derived_from`` edges** at all --
not into ``wapo.py``'s ``docs_v2``, ``msmarco_passage.py``'s ``docs``, or
``car.py``'s ``docs_v20``, even though all three are exactly what backs them
(``docs_v0`` = WaPo v2 + MS MARCO passage + CAR v2.0; ``docs_v1`` = MS MARCO
passage + CAR v2.0 -- noted here in prose for a human reading the graph, since
there's no clean way to say it as an edge). This is unlike every other
cross-file reuse in this codebase (e.g. ``kilt.py`` importing ``codec.py``'s
actual queries Table, ``touche.py`` importing ``clueweb12.py``'s actual docs
Table) -- there, the imported object *is* a v2 node, so the edge is real;
here, v1's ``PrefixedDocs`` bottoms out in v1's own registry, which has no v2
node to point at (``LazyDocs("msmarco-passage")`` is a v1-internal callback,
not a reference to this package's ``msmarco_passage.py`` module). A source
Resource-based Table would show its provenance in the graph automatically; a
``Parser`` whose ``build()`` reaches into a different loader entirely cannot.

Queries/qrels/scoreddocs have no such problem -- each subset's topic file and
qrels file is a plain, single-source download, wired up as ordinary v2
``Resource``s below, same as any other TREC track file. ``CastQueries``
(JSON topic files, imported from v1 unmodified -- it only ever calls
``.stream()`` on what it's given, same "thin layer" reuse as ``car.py``'s
``CarQueries``) is reused as a v2 ``Parser`` wrapper; ``TrecQrels``/
``TrecScoredDocs`` are v2's own general-purpose formats.

The ``/judged`` variants (``v0/train/judged``, ``v1/2019/judged``,
``v1/2020/judged``) are v1's own hand-built "queries restricted to those with
>= 1 qrel" derivation -- exactly what ``Filter(queries_with_qrels=True)``
already does (see ``msmarco_passage.py``'s ``train_judged``), so they are
expressed that way here rather than reused from v1.
"""
import ir_datasets
from ir_datasets.datasets.trec_cast import (
    Cast2019Query, Cast2020Query, CastQueries as _V1CastQueries,
    QRELS_DEFS, QRELS_DEFS_TRAIN,
)
from ir_datasets.v2 import (
    Benchmark, DocTable, Filter, QueryTable, Resource, Source, TrecQrels,
    TrecScoredDocs, irds,
)
from ir_datasets.v2.formats import Parser

NAME = 'trec-cast'

CITATION_2019 = 'dblp:journals/corr/abs-2003-13624'
CITATION_2020 = 'dblp:conf/trec/0001XC20'


class _TrecCastDocsParser(Parser):
    """Degraded-provenance shim: hands back whatever v1's own (already
    registered, unmodified) ``PrefixedDocs`` handler built for this v1 id --
    see the module docstring for why this can't carry a real ``derived_from``
    edge to wapo.py/msmarco_passage.py/car.py's own v2 nodes."""
    name = 'TrecCastDocs'

    def __init__(self, v1_id):
        self.v1_id = v1_id

    def build(self, source, node):
        # NOTE: v1's own top-level ir_datasets.load(), not ir_datasets.v2's --
        # this resolves purely within v1's own registry regardless of whether
        # this v1 id is later aliased back to a v2 node below.
        return ir_datasets.load(self.v1_id).docs_handler()


class _CastQueriesParser(Parser):
    name = 'CastQueries'

    def __init__(self, query_type):
        self.query_type = query_type

    def build(self, source, node):
        return _V1CastQueries(source, self.query_type)


# Files
# -----------------------------------------
train_queries_file = Resource('trec-cast-v0-train-queries.json',
    sources=['https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/training/train_topics_v1.0.json'],
    md5='2017389f5bbea04478574c6e84d65482',
    size=32_800,
)
train_qrels_file = Resource('trec-cast-v0-train-qrels.txt',
    sources=['https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/training/train_topics_mod.qrel'],
    md5='84af27620dfc009f1f76a58e3d9d6c40',
    size=80_964,
)
train_scoreddocs_file = Resource('trec-cast-v0-train-scoreddocs.teIn',
    sources=['https://huggingface.co/datasets/macavaney/trec-cast-files/resolve/main/train_topics.teIn'],
    md5='83bdb720e0c469390004598091021901',
    size=14_766_212,
)

eval_2019_queries_file = Resource('trec-cast-v1-2019-queries.json',
    sources=['https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json'],
    md5='362283885194feefcab8441d2bb24f7c',
    size=57_204,
)
eval_2019_qrels_file = Resource('trec-cast-v1-2019-qrels.txt',
    sources=['https://trec.nist.gov/data/cast/2019qrels.txt', Source.mirror()],
    md5='aab238105020c4cd55fae60dedfa9f1e',
    size=1_138_032,
)
eval_2019_scoreddocs_file = Resource('trec-cast-v1-2019-scoreddocs.teIn',
    sources=['https://huggingface.co/datasets/macavaney/trec-cast-files/resolve/main/test_topics.teIn'],
    md5='4c3958c09edab1b46474a337590c1ecd',
    size=27_691_171,
)

eval_2020_queries_file = Resource('trec-cast-v1-2020-queries.json',
    sources=['https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json'],
    md5='98ae2be2c82e294895a83e76b4133e19',
    size=78_998,
)
eval_2020_qrels_file = Resource('trec-cast-v1-2020-qrels.txt',
    sources=['https://trec.nist.gov/data/cast/2020qrels.txt', Source.mirror()],
    md5='de6a8406217945bdbf1da304214ef60c',
    size=1_563_427,
)

# Tables
# -----------------------------------------
# docs: source=[] (no Resources) is deliberate -- see the module docstring on
# why these two nodes carry no derived_from edges at all.
docs_v0 = DocTable('trec-cast-v0-docs',
    source=[], parser=_TrecCastDocsParser('trec-cast/v0'),
    lang='en', count_hint=47_696_605,
    desc='WaPo (v2) + MS MARCO passage (v1) + TREC CAR (v2.0), id-prefixed and '
         'combined via v1\'s PrefixedDocs -- see the module docstring on why '
         'this has no derived_from edges to wapo.py/msmarco_passage.py/car.py.')
docs_v1 = DocTable('trec-cast-v1-docs',
    source=[], parser=_TrecCastDocsParser('trec-cast/v1'),
    lang='en', count_hint=38_622_444,
    desc='MS MARCO passage (v1) + TREC CAR (v2.0), id-prefixed and combined '
         'via v1\'s PrefixedDocs -- see the module docstring on why this has '
         'no derived_from edges to msmarco_passage.py/car.py.')

train_queries = QueryTable('trec-cast-v0-train-queries',
    source=train_queries_file, parser=_CastQueriesParser(Cast2019Query),
    lang='en', count_hint=269)
train_qrels = TrecQrels('trec-cast-v0-train-qrels',
    source=train_qrels_file, defs=QRELS_DEFS_TRAIN, count_hint=2_399)
train_scoreddocs = TrecScoredDocs('trec-cast-v0-train-scoreddocs',
    source=train_scoreddocs_file, count_hint=269_000)

eval_2019_queries = QueryTable('trec-cast-v1-2019-queries',
    source=eval_2019_queries_file, parser=_CastQueriesParser(Cast2019Query),
    lang='en', count_hint=479)
eval_2019_qrels = TrecQrels('trec-cast-v1-2019-qrels',
    source=eval_2019_qrels_file, defs=QRELS_DEFS, count_hint=29_350)
eval_2019_scoreddocs = TrecScoredDocs('trec-cast-v1-2019-scoreddocs',
    source=eval_2019_scoreddocs_file, count_hint=479_000)

eval_2020_queries = QueryTable('trec-cast-v1-2020-queries',
    source=eval_2020_queries_file, parser=_CastQueriesParser(Cast2020Query),
    lang='en', count_hint=216)
eval_2020_qrels = TrecQrels('trec-cast-v1-2020-qrels',
    source=eval_2020_qrels_file, defs=QRELS_DEFS, count_hint=40_451)

# Benchmarks
# -----------------------------------------
v0_train = Benchmark('trec-cast-v0-train',
    docs=docs_v0, queries=train_queries, qrels=train_qrels, scoreddocs=train_scoreddocs,
    citation=CITATION_2019,
    desc='TREC CAsT 2019 training set.')
v0_train_judged = Benchmark('trec-cast-v0-train-judged',
    derived_from=v0_train, filter=Filter(queries_with_qrels=True),
    desc='trec-cast-v0-train restricted to queries with >= 1 qrel.')

v1_2019 = Benchmark('trec-cast-v1-2019',
    docs=docs_v1, queries=eval_2019_queries, qrels=eval_2019_qrels, scoreddocs=eval_2019_scoreddocs,
    citation=CITATION_2019,
    desc='Official evaluation set for TREC CAsT 2019.')
v1_2019_judged = Benchmark('trec-cast-v1-2019-judged',
    derived_from=v1_2019, filter=Filter(queries_with_qrels=True),
    desc='trec-cast-v1-2019 restricted to queries with >= 1 qrel.')

v1_2020 = Benchmark('trec-cast-v1-2020',
    docs=docs_v1, queries=eval_2020_queries, qrels=eval_2020_qrels,
    citation=CITATION_2020,
    desc='Official evaluation set for TREC CAsT 2020.')
v1_2020_judged = Benchmark('trec-cast-v1-2020-judged',
    derived_from=v1_2020, filter=Filter(queries_with_qrels=True),
    desc='trec-cast-v1-2020 restricted to queries with >= 1 qrel.')


# Registration
# -----------------------------------------
irds.register(docs_v0, docs_v1,
              v0_train, v0_train_judged,
              v1_2019, v1_2019_judged,
              v1_2020, v1_2020_judged)
