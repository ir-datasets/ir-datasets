"""DPR-w100 -- a v2 dataset family (the Dense Passage Retrieval Wikipedia
dump, split into 100-word passages, paired with four Q&A benchmarks: Natural
Questions dev/train and Trivia QA dev/train).

The corpus (``psgs_w100.tsv.gz``) is an ordinary ``doc_id<TAB>text<TAB>title``
TSV, so it uses v2's general-purpose ``TsvDocs`` format node directly (no
custom ``Parser`` needed for it) -- same as e.g. ``antique.py``/
``nfcorpus.py``. ``DprW100Doc`` (the record type) is still imported from v1's
``ir_datasets.datasets.dpr_w100`` -- a type import, not a behavioral one.

The four query+qrel subsets are a different story: v1's ``DprW100Manager`` is
a *stateful* handler, the same shape as ``AolManager`` in ``aol_ia.py`` --
its ``build()`` method reads one raw JSON download (a list of
question/positive-passage/hard-negative/negative records) and derives two
files on disk as a side effect (``queries.tsv``, a plain trec-style
``qrels``) the first time either is needed, kept lazy via v1's own
``_ManagedDlc`` shim. Rather than reimplement any of that (JSON walking,
qid assignment, relevance-level assignment from BM25 score / negative-ctx
bucket), this module reuses ``DprW100Manager``/``DprW100Queries`` wholesale,
one manager instance per subset, constructed exactly as v1's ``_init()``
does -- just with a v2 ``Resource``/pipeline standing in for v1's
``dlc[...]`` entry (a ``Resource`` satisfies the same ``.path()``/
``.stream()`` duck type v1's dlcs do, per ``aol_ia.py``).

Because each subset has its own manager deriving its own two files, the
queries and qrels ``Parser``s below are small and per-manager (closing over
one ``DprW100Manager`` each) rather than one shared parser -- unlike
``aol_ia.py``'s single shared ``_MANAGER``. Queries need a custom ``Parser``
wrapping v1's ``DprW100Queries`` (a narrow v1 class with no v2 format node of
its own, same situation as ``aol_ia.py``'s query handler). Qrels, however,
only need a custom ``Parser`` for the same *reason* queries do -- not because
the qrels themselves are non-standard (the manager writes plain
trec-qrels-shaped lines, which v2's general ``TrecQrels`` format node could
read) but because the ``Table``'s ``source=`` must stay the raw v2
``Resource`` (for correct ``derived_from`` provenance edges -- see
``source_resources`` in ``nodes.py``, which does not know how to walk into a
v1 ``_ManagedDlc``), while the manager's *build* must run against the
already-downloaded local file, not against ``source`` as handed to
``parser.build()``. So both queries and qrels get a thin local ``Parser``
that ignores the ``source``/``node`` arguments it's given and instead reads
through the manager it was constructed with, exactly as
``aol_ia.py``'s ``_AolIaQueriesParser``/``_AolIaQrelsParser`` do.

**Fixed, v1-style base path.** As in ``aol_ia.py``, each ``DprW100Manager``
is rooted at v1's own path (``ir_datasets.util.home_path()/'dpr-w100'/<subset>``)
and manages two interrelated derived files there together -- not one table's
own cache. Routing that through v2's per-node ``docstore_path`` would require
reworking ``DprW100Manager`` itself to be table-aware, out of scope here; this
module keeps the same fixed v1 base path instead.
"""
import ir_datasets
from ir_datasets.datasets.dpr_w100 import (
    DprW100Doc, DprW100Manager, DprW100Queries, QREL_DEFS,
)
from ir_datasets.formats import TrecQrels as _V1TrecQrels
from ir_datasets.v2 import Benchmark, QrelTable, QueryTable, Resource, TsvDocs, irds
from ir_datasets.v2.formats import Parser

NAME = 'dpr-w100'
BASE_PATH = ir_datasets.util.home_path() / NAME

CITATION_DPR = 'dblp:conf/emnlp/KarpukhinOMLWEC20'
CITATION_NQ = 'dblp:journals/tacl/KwiatkowskiPRCP19; dblp:conf/emnlp/KarpukhinOMLWEC20'
CITATION_TQA = 'dblp:conf/acl/JoshiCWZ17; dblp:conf/emnlp/KarpukhinOMLWEC20'


class _DprW100QueriesParser(Parser):
    name = 'DprW100Queries'

    def __init__(self, manager):
        self.manager = manager

    def build(self, source, node):
        return DprW100Queries(self.manager.file_ref('queries.tsv'))


class _DprW100QrelsParser(Parser):
    name = 'DprW100Qrels'

    def __init__(self, manager):
        self.manager = manager

    def build(self, source, node):
        return _V1TrecQrels(self.manager.file_ref('qrels'), QREL_DEFS)


# Files
# -----------------------------------------
docs_file = Resource('dpr-w100-docs.tsv.gz',
    sources=['https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz'],
    hash='md5:612fe66e0b6b41ee28f806140226c563',
    size=4_694_541_059,
)
nq_dev_file = Resource('dpr-w100-nq-dev.json.gz',
    sources=['https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz'],
    hash='md5:2640483dbe0df7ae29c6da419c551a80',
    size=256_239_282,
)
nq_train_file = Resource('dpr-w100-nq-train.json.gz',
    sources=['https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz'],
    hash='md5:a1c927b5adae71388eb064329387709f',
    size=2_314_892_908,
)
tqa_dev_file = Resource('dpr-w100-tqa-dev.json.gz',
    sources=['https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz'],
    hash='md5:d559dffe09acfe5a6370adea55b8abf2',
    size=207_271_749,
)
tqa_train_file = Resource('dpr-w100-tqa-train.json.gz',
    sources=['https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz'],
    hash='md5:5aa4d3577c91425cd20e239ed89a252b',
    size=1_848_559_940,
)

# One DprW100Manager per subset -- see module docstring. Constructed the same
# way v1's _init() does, just with a v2 Resource/pipeline standing in for
# v1's dlc[...] entry.
_nq_dev_manager = DprW100Manager(nq_dev_file.gunzip(), BASE_PATH / 'nq-dev')
_nq_train_manager = DprW100Manager(nq_train_file.gunzip(), BASE_PATH / 'nq-train')
_tqa_dev_manager = DprW100Manager(tqa_dev_file.gunzip(), BASE_PATH / 'tqa-dev', passage_id_key='psg_id')
_tqa_train_manager = DprW100Manager(tqa_train_file.gunzip(), BASE_PATH / 'tqa-train', passage_id_key='psg_id')


# Tables
# -----------------------------------------
with irds.defaults(lang='en'):
    docs = TsvDocs('dpr-w100-docs',
        source=docs_file.gunzip(),
        cls=DprW100Doc,
        skip_first_line=True,
        docstore_size_hint=12_827_215_492,
        count_hint=21_015_324,
        citation=CITATION_DPR,
    )

    nq_dev_queries = QueryTable('dpr-w100-natural-questions-dev-queries',
        source=nq_dev_file,
        parser=_DprW100QueriesParser(_nq_dev_manager),
        count_hint=6_515,
    )
    nq_dev_qrels = QrelTable('dpr-w100-natural-questions-dev-qrels',
        source=nq_dev_file,
        parser=_DprW100QrelsParser(_nq_dev_manager),
        defs=QREL_DEFS,
        count_hint=979_893,
    )

    nq_train_queries = QueryTable('dpr-w100-natural-questions-train-queries',
        source=nq_train_file,
        parser=_DprW100QueriesParser(_nq_train_manager),
        count_hint=58_880,
    )
    nq_train_qrels = QrelTable('dpr-w100-natural-questions-train-qrels',
        source=nq_train_file,
        parser=_DprW100QrelsParser(_nq_train_manager),
        defs=QREL_DEFS,
        count_hint=8_856_662,
    )

    tqa_dev_queries = QueryTable('dpr-w100-trivia-qa-dev-queries',
        source=tqa_dev_file,
        parser=_DprW100QueriesParser(_tqa_dev_manager),
        count_hint=8_837,
    )
    tqa_dev_qrels = QrelTable('dpr-w100-trivia-qa-dev-qrels',
        source=tqa_dev_file,
        parser=_DprW100QrelsParser(_tqa_dev_manager),
        defs=QREL_DEFS,
        count_hint=883_700,
    )

    tqa_train_queries = QueryTable('dpr-w100-trivia-qa-train-queries',
        source=tqa_train_file,
        parser=_DprW100QueriesParser(_tqa_train_manager),
        count_hint=78_785,
    )
    tqa_train_qrels = QrelTable('dpr-w100-trivia-qa-train-qrels',
        source=tqa_train_file,
        parser=_DprW100QrelsParser(_tqa_train_manager),
        defs=QREL_DEFS,
        count_hint=7_878_500,
    )

    # Benchmarks
    # -----------------------------------------
    natural_questions_dev = Benchmark('dpr-w100-natural-questions-dev',
        docs=docs, queries=nq_dev_queries, qrels=nq_dev_qrels,
        citation=CITATION_NQ,
        desc='Dev subset from the Natural Questions Q&A collection, using the '
             'full DPR Wikipedia dump and additional filtering (see the DPR paper).')
    natural_questions_train = Benchmark('dpr-w100-natural-questions-train',
        docs=docs, queries=nq_train_queries, qrels=nq_train_qrels,
        citation=CITATION_NQ,
        desc='Training subset from the Natural Questions Q&A collection, using the '
             'full DPR Wikipedia dump and additional filtering (see the DPR paper).')
    trivia_qa_dev = Benchmark('dpr-w100-trivia-qa-dev',
        docs=docs, queries=tqa_dev_queries, qrels=tqa_dev_qrels,
        citation=CITATION_TQA,
        desc='Dev subset from the Trivia QA dataset, using the DPR Wikipedia '
             'dump as the source collection (see the DPR paper).')
    trivia_qa_train = Benchmark('dpr-w100-trivia-qa-train',
        docs=docs, queries=tqa_train_queries, qrels=tqa_train_qrels,
        citation=CITATION_TQA,
        desc='Training subset from the Trivia QA dataset, using the DPR Wikipedia '
             'dump as the source collection (see the DPR paper).')


# Registration
# -----------------------------------------
irds.register(docs,
              nq_dev_queries, nq_dev_qrels,
              nq_train_queries, nq_train_qrels,
              tqa_dev_queries, tqa_dev_qrels,
              tqa_train_queries, tqa_train_qrels,
              natural_questions_dev, natural_questions_train,
              trivia_qa_dev, trivia_qa_train)
