"""Natural Questions -- a v2 dataset family (Google's long/short/yes-no-answer
Q&A dataset, framed here as ad-hoc passage ranking).

v1's handler for this dataset, ``NqManager`` (in
``ir_datasets.datasets.natural_questions``), is the same shape as
``AolManager`` in ``aol_ia.py`` -- model this file on that one first. In
short: ``NqManager.build()`` lazily derives several files on disk (one
docstore plus, per split, a ``queries.tsv``/``qrels.jsonl``/``scoreddocs.tsv``
triple) the first time any of them is needed, by streaming and parsing the 55
raw ``nq-{train,dev}-NN.jsonl.gz`` downloads in one pass. A single module-level
``NqManager`` is shared by every ``Parser`` below (docs + both splits' queries/
qrels/scoreddocs), so building any one of them builds them all, exactly as in
v1.

One wrinkle beyond ``aol_ia.py``: ``NqManager.__init__(self, dlcs, base_path)``
doesn't take individual v1 dlcs, it takes the whole v1 ``DownloadConfig``
context object, and ``build()`` calls ``self._dlcs.contents().keys()`` to
enumerate all 55 registered download names (to sort and split them into
train/dev by filename) before indexing in with ``self._dlcs[file_name]``.
That's a different duck-typing requirement than a single dlc's
``.path()``/``.stream()`` (which v2 ``Resource``s already satisfy): it needs
``.contents()`` (something with ``.keys()``) *and* ``__getitem__`` by name.
``_Dlcs`` below is a trivial ``dict`` subclass that satisfies both -- a dict
already has ``.keys()``/``__getitem__``, so ``.contents(self): return self``
is the whole shim -- mapping each of the 55 download names to its v2
``Resource`` (standing in for v1's ``dlc[...]`` entries, same substitution as
``aol_ia.py``/``clinicaltrials.py``/``highwire.py``).

Deliberate simplification, noted here rather than silently diverging from v1:
v1 marks all 55 downloads ``"stream": true, "skip_local": true`` -- it never
keeps a local copy, always re-streaming from the source URL on each access.
v2 ``Resource`` has no "never cache" equivalent, so this module's Resources
will cache to disk like any other Resource rather than re-streaming every
time; since they're only ever read once (during ``NqManager.build()``, which
memoizes via the docstore's ``.built()`` check), this just trades "never
cached" for "cached once," which is strictly cheaper on repeat use.

Same "fixed v1-style base path, not per-node docstore_path" reasoning as
``aol_ia.py``'s module docstring: ``NqManager`` manages several interrelated
files (the docstore plus six derived per-split files) under one directory,
not just one table's own cache, so this module keeps v1's fixed
``home_path()/'natural-questions'`` base path rather than reworking
``NqManager`` to be table-aware.
"""
import ir_datasets
from ir_datasets.datasets.natural_questions import (
    NqManager, NqPassageDoc, NqQrel, NqQrels as _V1NqQrels,
    NqScoredDocs as _V1NqScoredDocs,
)
from ir_datasets.formats import DocstoreBackedDocs, TsvQueries as _V1TsvQueries
from ir_datasets.v2 import (
    Benchmark, DocTable, QrelTable, QueryTable, Resource, RunTable, irds,
)
from ir_datasets.v2.formats import Parser

NAME = 'natural-questions'
BASE_PATH = ir_datasets.util.home_path() / NAME

QREL_DEFS = {1: 'passage marked by annotator as a "long" answer to the question'}


# Files
# -----------------------------------------
# All 55 raw per-shard downloads follow a fixed URL pattern; md5/size below
# taken from ir_datasets/etc/downloads.json's 'natural-questions' entries.
_DEV_FILES = {
    'nq-dev-00': ('21df324e9d0725c7cbc5ec06a34b630b', 219_593_373),
    'nq-dev-01': ('3de5cd9d66b705f3ef0462c9bcde1c4b', 200_209_706),
    'nq-dev-02': ('d7b6f2e7f296006ad2f3de291d5960ce', 210_446_574),
    'nq-dev-03': ('0d93b1e520328c50e0f4582dc89023c7', 216_859_801),
    'nq-dev-04': ('cc27dc8fc0a2d2753e2ff60a7e6fb976', 220_929_521),
}
_TRAIN_FILES = {
    'nq-train-00': ('22c9c2954ea80ff33f9667a9b398c86c', 858_728_609),
    'nq-train-01': ('2b76373340261ef019434235a2671a49', 891_498_165),
    'nq-train-02': ('785f6ca88ba7f039240218d20644f121', 885_374_316),
    'nq-train-03': ('4386753b84222234e12455cf99f116b9', 885_313_666),
    'nq-train-04': ('e02d403b1bacee5404f1db48fa3250d9', 890_873_425),
    'nq-train-05': ('39c46c3b5ab81ba92766dcb796037fd4', 873_023_109),
    'nq-train-06': ('e2c8f7434a40cc3740be8e0597c0db78', 866_509_301),
    'nq-train-07': ('3fc55e3e1ef3d834fed665840522e841', 838_940_867),
    'nq-train-08': ('660990d45610d4d4f37fcd21965b183e', 902_610_214),
    'nq-train-09': ('f4335cb2b08d3166ef5fa007560d0f95', 883_494_801),
    'nq-train-10': ('66c607ec3dae612422dd44a0a4ead32b', 876_311_133),
    'nq-train-11': ('a9b36187b8ccee70f461e8ae06109cb4', 878_127_326),
    'nq-train-12': ('feeec2103ce408899c721fc937dab83b', 889_257_016),
    'nq-train-13': ('97f3068af15e6a0c1d107bfa6dc50f70', 891_769_129),
    'nq-train-14': ('c4ea6358f617fbb8f9e65760e88b2028', 892_523_839),
    'nq-train-15': ('e3bf5b86c977b41bce825d853fba8e3e', 910_660_095),
    'nq-train-16': ('92abe032b4608bf35f08bcf8295eb1d3', 878_177_689),
    'nq-train-17': ('2c3acddbce0f5221f24e9db4e1d7662a', 872_805_189),
    'nq-train-18': ('a301430662fb1f25a73359f521d1da47', 875_275_428),
    'nq-train-19': ('9e0e96cca9e3594f885d8e6063cbd0c7', 862_034_169),
    'nq-train-20': ('8747331d168160e013a0c81e0323491d', 887_586_358),
    'nq-train-21': ('6de9b253b4a069fdfdc2bc80ddc5c2d1', 890_472_815),
    'nq-train-22': ('7c24574f4bc21ad764b9b8b208b97d64', 888_396_337),
    'nq-train-23': ('eadf9189bd9557943f468323e00de2ff', 900_331_594),
    'nq-train-24': ('f063dec5a57b2701e9318d3c93ad6486', 871_216_444),
    'nq-train-25': ('7db0ce87b997e3d47d4b511924384ec8', 871_166_814),
    'nq-train-26': ('c50093c839d33956d38548e2b44e0c50', 903_385_811),
    'nq-train-27': ('355e504e2a4b5bb6eb05d6976c93517f', 842_966_594),
    'nq-train-28': ('f4da07e59b33d1ce1dc006363f961e9d', 876_393_409),
    'nq-train-29': ('519c4a0349495253ec9f4211c85f8fcb', 872_982_425),
    'nq-train-30': ('96cde3a58f5e9bad4c94510aab7a5db5', 899_739_217),
    'nq-train-31': ('26955c79974141f71cb4729908f9c45b', 875_703_668),
    'nq-train-32': ('91e33f14401adafb3d5cd2c13239531b', 895_840_703),
    'nq-train-33': ('d28bf97bf7d433e64d7c016a477e97a1', 874_713_497),
    'nq-train-34': ('c98c6aa8578b04524a8e7f035aaa4bbe', 872_620_262),
    'nq-train-35': ('3e2b5b6280158f9ba312158a57360512', 854_439_473),
    'nq-train-36': ('b6114ca6d3804c045b6abd4b6e6a9f52', 866_233_094),
    'nq-train-37': ('89d1c93f2425c232882118fc3af92b9d', 894_411_832),
    'nq-train-38': ('785d092d65bb3d815eb5a8027d764c13', 879_967_719),
    'nq-train-39': ('c58b54591127ef977f9b900c05ec44ec', 887_056_754),
    'nq-train-40': ('cd14b09b6fc9ae8a7bda558b020459aa', 873_720_601),
    'nq-train-41': ('bb7415c0fd7cdc52ef8381c44194f2da', 880_452_966),
    'nq-train-42': ('08f384e34c0358ab4e9d72ac18eeb4f2', 856_217_171),
    'nq-train-43': ('c148652671404f7b2796327db542ef90', 908_184_635),
    'nq-train-44': ('245074693e3863f099ad92a2e7e07d88', 891_701_874),
    'nq-train-45': ('8e6d4a0895f5c87ede7cbbd0fe8ec0bb', 870_559_738),
    'nq-train-46': ('6b965fb2b2fdc3772da641aac10a7c7e', 883_791_796),
    'nq-train-47': ('79ac295cf818f1dcdf676772b3863232', 882_109_720),
    'nq-train-48': ('545bb6edf1290d61fd42b13125b5bf2a', 882_241_605),
    'nq-train-49': ('d6aaa8706626d4b6210e4f9bc6ffe265', 863_247_626),
}

_dev_files = {
    name: Resource(f'{name}.jsonl.gz',
        sources=[f'https://storage.googleapis.com/natural_questions/v1.0/dev/{name}.jsonl.gz'],
        md5=md5, size=size)
    for name, (md5, size) in _DEV_FILES.items()
}
_train_files = {
    name: Resource(f'{name}.jsonl.gz',
        sources=[f'https://storage.googleapis.com/natural_questions/v1.0/train/{name}.jsonl.gz'],
        md5=md5, size=size)
    for name, (md5, size) in _TRAIN_FILES.items()
}
_all_files = {**_dev_files, **_train_files}


class _Dlcs(dict):
    """Duck-types v1's ``DownloadConfig`` context object well enough for
    ``NqManager``: ``.contents()`` (returning something with ``.keys()``) and
    ``__getitem__`` by name. A dict already has both of the latter, so the
    only thing to add is ``.contents()`` returning ``self``. See the module
    docstring."""

    def contents(self):
        return self


# One shared, stateful manager -- see module docstring. Constructed the same
# way v1's _init() does, just with a _Dlcs of v2 Resources standing in for
# v1's DownloadConfig-backed dlcs.
_MANAGER = NqManager(_Dlcs(_all_files), BASE_PATH)


class _NqDocsParser(Parser):
    name = 'NqDocs'

    def build(self, source, node):
        return DocstoreBackedDocs(_MANAGER.docs_store, docs_cls=NqPassageDoc,
            namespace=NAME, lang=node.lang)


class _NqQueriesParser(Parser):
    name = 'NqQueries'

    def __init__(self, split):
        self.split = split

    def build(self, source, node):
        return _V1TsvQueries(_MANAGER.file_ref(f'{self.split}.queries.tsv'),
            namespace=NAME, lang=node.lang)


class _NqQrelsParser(Parser):
    name = 'NqQrels'

    def __init__(self, split):
        self.split = split

    def build(self, source, node):
        # NqQrels' qrels_defs() is a fixed dict hardcoded on the class (it
        # doesn't take a defs= constructor arg) -- node.defs (set below, for
        # the frozen manifest/metadata) isn't threaded through here, same
        # "defs fixed rather than forwarded from node.defs" pattern as
        # car.py's/highwire.py's narrow-format qrels parsers.
        return _V1NqQrels(_MANAGER.file_ref(f'{self.split}.qrels.jsonl'))


class _NqScoredDocsParser(Parser):
    name = 'NqScoredDocs'

    def __init__(self, split):
        self.split = split

    def build(self, source, node):
        return _V1NqScoredDocs(_MANAGER.file_ref(f'{self.split}.scoreddocs.tsv'))


# Tables
# -----------------------------------------
with irds.defaults(lang='en'):
    docs = DocTable('natural-questions-docs',
        source=list(_all_files.values()),
        parser=_NqDocsParser(),
        count_hint=28_390_850,
    )

    dev_queries = QueryTable('natural-questions-dev-queries',
        source=list(_dev_files.values()),
        parser=_NqQueriesParser('dev'),
        count_hint=7_830,
    )
    dev_qrels = QrelTable('natural-questions-dev-qrels',
        source=list(_dev_files.values()),
        parser=_NqQrelsParser('dev'),
        defs=QREL_DEFS,
        count_hint=7_695,
    )
    dev_scoreddocs = RunTable('natural-questions-dev-scoreddocs',
        source=list(_dev_files.values()),
        parser=_NqScoredDocsParser('dev'),
        count_hint=973_480,
    )

    train_queries = QueryTable('natural-questions-train-queries',
        source=list(_train_files.values()),
        parser=_NqQueriesParser('train'),
        count_hint=307_373,
    )
    train_qrels = QrelTable('natural-questions-train-qrels',
        source=list(_train_files.values()),
        parser=_NqQrelsParser('train'),
        defs=QREL_DEFS,
        count_hint=152_148,
    )
    train_scoreddocs = RunTable('natural-questions-train-scoreddocs',
        source=list(_train_files.values()),
        parser=_NqScoredDocsParser('train'),
        count_hint=40_374_730,
    )


# Benchmarks
# -----------------------------------------
CITATION = 'Kwiatkowski2019Nq'

dev = Benchmark('natural-questions-dev',
    docs=docs,
    queries=dev_queries,
    qrels=dev_qrels,
    scoreddocs=dev_scoreddocs,
    citation=CITATION,
    desc='Official dev set.',
)
train = Benchmark('natural-questions-train',
    docs=docs,
    queries=train_queries,
    qrels=train_qrels,
    scoreddocs=train_scoreddocs,
    citation=CITATION,
    desc='Official train set.',
)


# Registration
# -----------------------------------------
irds.register(
    docs,
    dev_queries, dev_qrels, dev_scoreddocs,
    train_queries, train_qrels, train_scoreddocs,
    dev, train,
)
