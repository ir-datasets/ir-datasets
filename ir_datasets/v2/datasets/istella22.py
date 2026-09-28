"""Istella22 -- a v2 dataset family (an Italian web-search collection pairing
document/query text with LTR features, from the Istella search engine).

Everything -- the corpus, the ``test`` queries/qrels, and the 5 folds' own
query-id lists -- comes from a single downloadable tar.gz (~26GB), extracted
in full via v1's ``TarExtractAll`` to a fixed directory
(``<home>/istella22/istella22_extracted``), the same shape ``c4.py``/
``clueweb09.py``/``clueweb12.py`` use for their own checkpoint archives. This
is a real, automatically-downloadable ``Resource`` (not a manual/
``Source.local()`` one) -- ``dua=`` on it communicates only that the download
requires having accepted the Istella22 Licence Agreement, same as every other
DUA-gated *downloadable* ``Resource`` in this package.

``TarExtractAll.path()`` is a no-op once the directory already exists (see
``ir_datasets/util/fileio.py``), so it is cheap to reference from every
parser; a single module-level ``_base_dlc`` is still shared by all of them
(docs, queries, qrels, and the 5 fold parsers below), the same "one shared
stateful pipeline object" shape ``aol_ia.py``'s module-level ``_MANAGER``
uses, so there is exactly one place that names the extraction directory.

Individual files are then read off that extracted tree via v1's
``RelativePath``/``GzipExtract``/``TarExtract``, exactly as v1's own
``_init()`` does, wrapped in thin ``Parser``s -- there is no general-purpose
v2 ``JsonlDocs``/``JsonlQueries`` format node in ``formats.py`` the way there
is for ``TsvDocs``, so docs/queries go through local ``Parser``s that reuse
v1's ``JsonlDocs``/``JsonlQueries`` handlers directly, same "thin layer over
v1 machinery" as ``codec.py``. Qrels get a local ``Parser`` too, even though
v2's general-purpose ``TrecQrels`` format node could read the manager's
output directly once extracted+gunzipped -- not because the format is
unusual, but for the same reason docs/queries do: ``QrelTable.source=`` must
stay the raw ``source_file`` Resource (so ``source_resources`` can find it
and record a real ``derived_from`` edge -- a v1 ``GzipExtract(RelativePath(...))``
pipeline has no ``._parent`` for it to walk, so handing that straight to the
builtin ``TrecQrels`` node's ``source=`` would silently produce *no* edge at
all), while the actual extraction path is read through ``_base_dlc`` inside
the parser, same split every other table here uses.

The 5 folds (``test/fold1`` .. ``test/fold5``) are each the ``test``
benchmark filtered down to the query ids listed in one member of
``queries.test.folds.tar.gz`` (one id per line). This is exactly the
``derived_from`` + ``Filter`` shape ``codec.py``'s per-domain benchmarks use,
except simpler: no per-fold ``QueryTable`` is built by hand here (unlike
codec's per-domain queries, which need their own parser to *select* the
domain out of a single combined queries file) -- passing no explicit
``queries=``/``qrels=`` to each fold ``Benchmark`` lets ``Filter`` derive both
automatically from the parent ``test`` benchmark's own tables, naming them
``irds:istella22-test-fold<N>-queries``/``-qrels`` (see
``Benchmark.structural_edges``/``Filter.apply``).

The id source is a file, not another node's query ids, so ``Filter`` is given
a ``query_ids=`` callable rather than ``ids_of(...)`` -- and deliberately not
``ids_from_lines(...)`` either: v2's ``ids_from_lines`` reads one raw id per
line, with no preprocessing hook, but v1's own fold-id logic
(``fold_qids_factory`` in the legacy module) additionally does
``.lstrip('0')`` on every line -- confirmed necessary here, not just
faithfulness for its own sake: the integration test fixture
(``test/integration/istella22.py``) shows the corpus's own query ids with no
leading zeros (e.g. ``'480'``, ``'263'``), so a fold file whose raw lines carry
leading zeros would silently filter to nothing (or the wrong ids) without the
strip. So this module defines its own small ``_fold_query_ids`` lazy
id-set callable, modeled on ``ids_from_lines`` (same streamed-read shape, same
``depends_on`` contract for the ``derived_from`` edge -- pointed at the
underlying tar.gz ``Resource`` directly, since the intermediate v1
``TarExtract``/``RelativePath`` pipeline objects are not v2 ``_Pipe``s and so
``source_resources`` cannot walk them the way it walks a ``.member().gunzip()``
chain), but with the extra ``.lstrip('0')`` step.
"""
import ir_datasets
from ir_datasets.datasets.istella22 import Istella22Doc, QREL_DEFS
from ir_datasets.util import GzipExtract, RelativePath, TarExtract, TarExtractAll
from ir_datasets.v2 import Benchmark, DocTable, Filter, QrelTable, QueryTable, Resource, irds
from ir_datasets.v2.formats import Parser

NAME = 'istella22'
BASE_PATH = ir_datasets.util.home_path() / NAME

DUA = ("To use the Istella22 dataset, you must read and accept the Istella22 Licence "
       "Agreement, found here: <https://istella.ai/data/istella22-dataset/>")

CITATION = 'Dato2022Istella'


class _Istella22DocsParser(Parser):
    name = 'JsonlDocs'

    def build(self, source, node):
        from ir_datasets.formats import JsonlDocs
        docs_src = GzipExtract(RelativePath(_base_dlc, 'istella22/docs.jsonl.gz'))
        return JsonlDocs(docs_src, doc_cls=Istella22Doc, lang=node.lang,
                          count_hint=node.count_hint, docstore_path=str(node.docstore_path))


class _Istella22QueriesParser(Parser):
    name = 'JsonlQueries'

    def build(self, source, node):
        from ir_datasets.formats import JsonlQueries
        queries_src = GzipExtract(RelativePath(_base_dlc, 'istella22/queries.test.jsonl.gz'))
        return JsonlQueries(queries_src, lang=node.lang)


class _Istella22QrelsParser(Parser):
    """Same reasoning as the docs/queries parsers above: ``source=`` on the
    ``QrelTable`` stays the raw ``source_file`` Resource (for a correct
    ``derived_from`` edge -- a v1 ``GzipExtract(RelativePath(...))`` pipeline
    has no ``._parent`` for ``source_resources`` to walk, so using one
    directly as ``source=`` would silently produce no edge at all), while the
    parser ignores it and reads through the shared extracted tree instead."""
    name = 'TrecQrels'

    def build(self, source, node):
        from ir_datasets.formats import TrecQrels as _V1TrecQrels
        qrels_src = GzipExtract(RelativePath(_base_dlc, 'istella22/qrels.test.gz'))
        return _V1TrecQrels(qrels_src, node.defs or {})


def _fold_query_ids(fold):
    """Lazily read one id per line out of the fold's member of
    ``queries.test.folds.tar.gz``, stripping leading zeros -- see the module
    docstring for why this can't just be ``ids_from_lines``."""
    def _ids():
        fold_member = TarExtract(RelativePath(_base_dlc, 'istella22/queries.test.folds.tar.gz'),
                                  f'./test.queries.{fold}')
        with fold_member.stream() as stream:
            return {qid.decode().strip().lstrip('0') for qid in stream}
    _ids.depends_on = (source_file,)
    return _ids


# Files
# -----------------------------------------
source_file = Resource('istella22-source.tar.gz',
    sources=['https://www.istella.ai/dataset/istella22.tar.gz'],
    md5='c2e49dca9730fbb14164ed890756dc1d',
    size=26_499_490_813,
    dua=DUA,
)

# Shared, extracted-once pipeline -- see module docstring.
_base_dlc = TarExtractAll(source_file, BASE_PATH / 'istella22_extracted')

# Tables
# -----------------------------------------
docs = DocTable('istella22-docs',
    source=source_file,
    parser=_Istella22DocsParser(),
    lang=None,
    count_hint=8_421_456,
)
test_queries = QueryTable('istella22-test-queries',
    source=source_file,
    parser=_Istella22QueriesParser(),
    lang='it',
    count_hint=2_198,
)
test_qrels = QrelTable('istella22-test-qrels',
    source=source_file,
    parser=_Istella22QrelsParser(),
    defs=QREL_DEFS,
    count_hint=10_693,
)

# Benchmarks
# -----------------------------------------
test = Benchmark('istella22-test',
    docs=docs, queries=test_queries, qrels=test_qrels,
    citation=CITATION,
    desc='Istella22: official test query set.')

_FOLD_COUNTS = {
    'fold1': 2_164,
    'fold2': 2_140,
    'fold3': 2_197,
    'fold4': 2_098,
    'fold5': 2_094,
}

fold_benchmarks = {}
for _fold in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:
    fold_benchmarks[_fold] = Benchmark(f'istella22-test-{_fold}',
        derived_from=test,
        filter=Filter(query_ids=_fold_query_ids(_fold), mode='include'),
        citation=CITATION,
        desc=f'Istella22: fold {_fold[-1]} of the official test query set.')


# Registration
# -----------------------------------------
irds.register(docs, test, *fold_benchmarks.values())
