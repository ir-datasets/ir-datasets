"""CodeSearchNet -- a v2 dataset family (semantic code search across six
programming languages).

The corpus is SIX independent per-language zip downloads (python, java, go,
php, ruby, javascript), each extracted via v1's ``ZipExtractCache`` to its
own fixed directory (``<home>/codesearchnet/<lang>``) -- the zip equivalent
of ``istella22.py``'s ``TarExtractAll``, except here there are six separate
archives/directories instead of one, so there is no single shared
``_base_dlc`` the way istella22.py has. v1's ``CodeSearchNetDocs`` then
glob-walks all six extracted directories together to build one combined doc
collection, and v1's ``CodeSearchNetQueries``/``CodeSearchNetQrels`` glob-walk
the SAME six directories filtered to one of three split subdirectories
(train/valid/test). Since ``Table.__init__`` only accepts one ``source=``
value but structural edges need all six raw zip ``Resource``s to be
discoverable, each of the four ``DocTable``/``QueryTable``/``QrelTable``
groups is given the flat list of the same six raw ``Resource``s as
``source=`` (so ``source_resources`` finds a real ``derived_from`` edge to
each one), and each thin ``Parser`` re-wraps that flat list into
``ZipExtractCache(resource, base_path/lang)`` pairs before handing them to
the unmodified v1 handler class -- same "flat list, split-and-rebuild in the
parser" shape ``highwire.py``/``pmc.py`` use for their own multi-Resource
corpora.

The fourth subset, ``challenge``, is a completely different, hand-curated
benchmark that does NOT derive from the six per-language zips: a plain-text
query list (``CodeSearchNetChallengeQueries``, one query per line after a
header row, with the query's *line number* used as its ``query_id`` -- kept
exactly as v1 does it, even though that's a fragile id scheme) and a CSV of
judgments (``CodeSearchNetChallengeQrels``) whose rows are joined to query
ids by looking up each row's query *text* against the queries handler's own
``text -> query_id`` map (built from ``queries_iter()``). That means the
qrels parser needs a *live* queries handler, not just the raw queries
``Resource`` -- the same shape ``kilt.py``'s CODEC-query reuse and
``nyt.py``-style cross-facet reuse rely on. Two ways to get one were
considered:

  (a) construct a second, throwaway ``CodeSearchNetChallengeQueries``
      instance directly inside the qrels ``Parser.build()``. Cheap: its
      ``__init__`` only stores the dlc, doing no I/O or other one-time work,
      so building it twice is harmless.
  (b) close over the ``challenge_queries`` ``QueryTable`` node itself and
      read its ``.handler`` lazily inside ``build()`` (no import-time
      circularity, since both tables are defined in the same module and
      ``build()`` only runs when the qrels table is actually resolved).

Option (b) is used here (mirroring ``kilt.py``): it keeps the live queries
handler in sync with whatever the ``challenge_queries`` table actually
resolves to (e.g. if that table's parser or source ever changes), rather
than silently duplicating today's construction logic in two places.
``QREL_DEFS``/``QREL_DEFS_CHALLENGE``/``CodeSearchNetChallengeQrel`` are
copied from the legacy module unmodified.

``docs_lang()`` is ``None`` in v1 (code, not natural language), so
``lang=None`` on the docs ``DocTable`` too.
"""
import ir_datasets
from ir_datasets.datasets.codesearchnet import (
    CodeSearchNetChallengeQrel, CodeSearchNetChallengeQrels as _V1ChallengeQrels,
    CodeSearchNetChallengeQueries as _V1ChallengeQueries, CodeSearchNetDocs as _V1Docs,
    CodeSearchNetQrels as _V1Qrels, CodeSearchNetQueries as _V1Queries,
    QREL_DEFS, QREL_DEFS_CHALLENGE,
)
from ir_datasets.util import ZipExtractCache
from ir_datasets.v2 import Benchmark, DocTable, QrelTable, QueryTable, Resource, irds
from ir_datasets.v2.formats import Parser

NAME = 'codesearchnet'
BASE_PATH = ir_datasets.util.home_path() / NAME
LANGS = ['python', 'java', 'go', 'php', 'ruby', 'javascript']

CITATION = 'dblp:journals/corr/abs-1909-09436'


def _extracted_dlcs(source):
    """Rebuild the six ``ZipExtractCache``-wrapped per-language directories
    from the flat list of six raw zip ``Resource``s -- see module docstring."""
    return [ZipExtractCache(res, BASE_PATH / lang) for res, lang in zip(source, LANGS)]


class _CodeSearchNetDocsParser(Parser):
    name = 'CodeSearchNetDocs'

    def build(self, source, node):
        return _V1Docs(_extracted_dlcs(source))


class _CodeSearchNetQueriesParser(Parser):
    name = 'CodeSearchNetQueries'

    def __init__(self, split):
        self.split = split

    def build(self, source, node):
        return _V1Queries(_extracted_dlcs(source), self.split)


class _CodeSearchNetQrelsParser(Parser):
    name = 'CodeSearchNetQrels'

    def __init__(self, split):
        self.split = split

    def build(self, source, node):
        return _V1Qrels(_extracted_dlcs(source), self.split)


class _CodeSearchNetChallengeQueriesParser(Parser):
    name = 'CodeSearchNetChallengeQueries'

    def build(self, source, node):
        return _V1ChallengeQueries(source)


class _CodeSearchNetChallengeQrelsParser(Parser):
    """Needs a *live* queries handler to resolve each qrel row's query text
    to a query id (see module docstring) -- closes over the sibling
    ``challenge_queries`` ``QueryTable`` node and reads its ``.handler``
    lazily, so there's no import-time circularity and the queries handler
    stays in sync with whatever that table resolves to."""
    name = 'CodeSearchNetChallengeQrels'

    def build(self, source, node):
        return _V1ChallengeQrels(source, challenge_queries.handler)


# Files
# -----------------------------------------
lang_files = [
    Resource(f'codesearchnet-{lang}.zip',
        sources=[f'https://huggingface.co/datasets/macavaney/codesearchnet-mirror/resolve/main/v2/{lang}.zip'],
        md5=md5, size=size)
    for lang, (md5, size) in {
        'python': ('07b49dd01fbac894fbdae22da6462e4f', 940_909_997),
        'java': ('fea180077275d8f98f42a3386f492837', 1_060_569_153),
        'go': ('c0288db91f067c95bb952577949e7b13', 487_525_935),
        'php': ('62373f85cfae2f5d7422dc1e55fbbb50', 851_894_048),
        'ruby': ('6847c0149666334cc937909b0e2297ae', 111_758_028),
        'javascript': ('7649178a02f7b5c8fbccb64abe7946b6', 1_664_713_350),
    }.items()
]
challenge_queries_file = Resource('codesearchnet-challenge-queries.csv',
    sources=['https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/queries.csv'],
    md5='6041a0c32dff4286859ca76d420d76f4',
    size=2_493,
)
challenge_qrels_file = Resource('codesearchnet-challenge-qrels.csv',
    sources=['https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/annotationStore.csv'],
    md5='9e0a57ae90b3dd0144d59064d0751abd',
    size=677_798,
)

# Tables
# -----------------------------------------
docs = DocTable('codesearchnet-docs',
    source=lang_files,
    parser=_CodeSearchNetDocsParser(),
    lang=None,
    count_hint=2_070_536,
)

train_queries = QueryTable('codesearchnet-train-queries',
    source=lang_files,
    parser=_CodeSearchNetQueriesParser('train'),
    lang='en',
    count_hint=1_880_853,
)
train_qrels = QrelTable('codesearchnet-train-qrels',
    source=lang_files,
    parser=_CodeSearchNetQrelsParser('train'),
    defs=QREL_DEFS,
    count_hint=1_880_853,
)
valid_queries = QueryTable('codesearchnet-valid-queries',
    source=lang_files,
    parser=_CodeSearchNetQueriesParser('valid'),
    lang='en',
    count_hint=89_154,
)
valid_qrels = QrelTable('codesearchnet-valid-qrels',
    source=lang_files,
    parser=_CodeSearchNetQrelsParser('valid'),
    defs=QREL_DEFS,
    count_hint=89_154,
)
test_queries = QueryTable('codesearchnet-test-queries',
    source=lang_files,
    parser=_CodeSearchNetQueriesParser('test'),
    lang='en',
    count_hint=100_529,
)
test_qrels = QrelTable('codesearchnet-test-qrels',
    source=lang_files,
    parser=_CodeSearchNetQrelsParser('test'),
    defs=QREL_DEFS,
    count_hint=100_529,
)

challenge_queries = QueryTable('codesearchnet-challenge-queries',
    source=challenge_queries_file,
    parser=_CodeSearchNetChallengeQueriesParser(),
    lang='en',
    count_hint=99,
)
challenge_qrels = QrelTable('codesearchnet-challenge-qrels',
    source=challenge_qrels_file,
    parser=_CodeSearchNetChallengeQrelsParser(),
    defs=QREL_DEFS_CHALLENGE,
    count_hint=4_006,
)

# Benchmarks
# -----------------------------------------
train = Benchmark('codesearchnet-train',
    docs=docs, queries=train_queries, qrels=train_qrels,
    citation=CITATION,
    desc='CodeSearchNet: official train set, using queries inferred from docstrings.')
valid = Benchmark('codesearchnet-valid',
    docs=docs, queries=valid_queries, qrels=valid_qrels,
    citation=CITATION,
    desc='CodeSearchNet: official validation set, using queries inferred from docstrings.')
test = Benchmark('codesearchnet-test',
    docs=docs, queries=test_queries, qrels=test_qrels,
    citation=CITATION,
    desc='CodeSearchNet: official test set, using queries inferred from docstrings.')
challenge = Benchmark('codesearchnet-challenge',
    docs=docs, queries=challenge_queries, qrels=challenge_qrels,
    citation=CITATION,
    desc='CodeSearchNet: official challenge set, with keyword queries and deep relevance assessments.')


# Registration
# -----------------------------------------
irds.register(docs, train, valid, test, challenge)
