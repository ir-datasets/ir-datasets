"""AOL-IA -- a v2 dataset family (the AOL query log, paired with documents
re-crawled from the Internet Archive for the URLs the log references).

There is exactly one dataset here (no subsets), so this file is smaller than
most v2 families, but v1's handler for it (``AolManager``, in
``ir_datasets.datasets.aol_ia``) is unusually stateful: its ``build()``
method reads the ten raw AOL log-line files and derives *three* artifacts on
disk as a side effect -- ``queries.tsv``, a plain trec-style ``qrels`` file,
and ``log.pkl.lz4`` (the pickled query-log stream) -- the first time any of
them is needed. v1 keeps this lazy via ``_ManagedDlc``, a duck-typed dlc
whose ``.path()``/``.stream()`` call ``manager.build()`` before touching the
file. Rather than reimplement any of that (log parsing, sessionization,
query/doc-id hashing, the three-derived-files dance), this module reuses
``AolManager`` wholesale -- constructed exactly as v1's ``_init()`` does,
just with v2 ``Resource``s standing in for v1's ``dlc[...]`` entries (a
``Resource``, and the ``.member()``/``.gunzip()`` pipeline views over one,
satisfy the same ``.path()``/``.stream()`` duck type v1's dlcs do -- see
``clinicaltrials.py``/``highwire.py`` for the same substitution). A single
module-level ``AolManager`` is shared by all four ``Parser``s below, so
that -- as in v1 -- building queries, qrels, docs, or the log via any one of
them also produces the others.

The ``qlogs`` table (the raw query log itself, v1's ``AolQlogs``) is a
``QlogTable`` -- registered alongside ``docs``/``queries``/``qrels`` as a
standalone artifact, not as a ``Benchmark`` facet (there is no ``Benchmark``
in this file at all: v1 never bundles AOL's docs/queries/qrels into one
evaluable task either, just exposes the four as siblings of one ``Dataset``).
Its ``Parser`` reuses v1's own ``AolQlogs`` wrapper around
``_MANAGER.file_ref('log.pkl.lz4')``, same as the other three tables reuse
their own v1 counterparts.

One deliberate simplification, noted here rather than silently diverging
from v1:

* **Fixed, v1-style base path.** v1 roots ``AolManager`` at
  ``ir_datasets.util.home_path()/'aol-ia'`` and has it manage several
  interrelated files there (the three derived artifacts above, plus the doc
  docstore) under one directory -- not just one table's own cache. Routing
  that through v2's per-node ``docstore_path`` (as ``gov.py`` does for a
  single docstore) would require reworking ``AolManager`` itself to be
  table-aware, which is out of scope here. This module keeps the same fixed
  v1 base path instead, the same "reuse v1's own path/caching logic
  unmodified" precedent as ``clinicaltrials.py``/``highwire.py``.

The document contents themselves have no v1 ``Resource``/download-config
entry at all: ``AolManager._build_docs`` requires the user to have already
run an external tool (`aolia-tools <https://github.com/terrierteam/aolia-tools>`_)
to populate ``downloaded_docs/*.jsonl.lz4`` files under the base path, and
raises with instructions if it finds them missing. That manual prerequisite
is preserved as-is (through ``AolManager``); the docs table's ``source=`` is
still declared as the two log/id2wb Resources for provenance purposes (the
manager is constructed from them), even though the doc *contents* do not
themselves come from either.
"""
import ir_datasets
from ir_datasets.datasets.aol_ia import AolIaDoc, AolManager, AolQlogs, QREL_DEFS
from ir_datasets.formats import DocstoreBackedDocs
from ir_datasets.formats import TrecQrels as _V1TrecQrels
from ir_datasets.formats import TsvQueries as _V1TsvQueries
from ir_datasets.v2 import DocTable, QlogTable, QrelTable, QueryTable, Resource, irds
from ir_datasets.v2.formats import Parser

NAME = 'aol-ia'
BASE_PATH = ir_datasets.util.home_path() / NAME


# Files
# -----------------------------------------
logs_file = Resource('aol-ia-logs.tar.gz',
    sources=['http://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/aol-data.tar.gz'],
    md5='31cd27ce12c3a3f2df62a38050ce4c0a',
    size=460_331_537,
)
id2wb_file = Resource('aol-ia-id2wb.tsv.gz',
    sources=['https://macavaney.us/aol.id2wb.tsv.gz'],
    md5='afbf9b03e1a0fabc9f3fdd5105e6ae5a',
    size=40_099_187,
)

# The ten raw per-file logs, each a member of the tar.gz, individually
# gzipped again within it -- same two-step pipeline as v1's
# GzipExtract(TarExtract(dlc['logs'], '.../user-ct-test-collection-NN.txt.gz')).
_log_members = [
    logs_file.member(f'AOL-user-ct-collection/user-ct-test-collection-{i:02d}.txt.gz')
        .gunzip()
    for i in range(1, 11)
]

# One shared, stateful manager -- see module docstring. Constructed the same
# way v1's _init() does, just with v2 Resources/pipelines standing in for
# v1's dlc[...] entries.
_MANAGER = AolManager(_log_members, id2wb_file.gunzip(), BASE_PATH)


class _AolIaDocsParser(Parser):
    name = 'AolIaDocs'

    def build(self, source, node):
        return DocstoreBackedDocs(_MANAGER.docs_store, docs_cls=AolIaDoc,
            namespace=NAME, lang=node.lang)


class _AolIaQueriesParser(Parser):
    name = 'AolIaQueries'

    def build(self, source, node):
        return _V1TsvQueries(_MANAGER.file_ref('queries.tsv'), lang=node.lang)


class _AolIaQrelsParser(Parser):
    name = 'AolIaQrels'

    def build(self, source, node):
        return _V1TrecQrels(_MANAGER.file_ref('qrels'), QREL_DEFS)


class _AolIaQlogsParser(Parser):
    name = 'AolQlogs'

    def build(self, source, node):
        return AolQlogs(_MANAGER.file_ref('log.pkl.lz4'))


# Tables
# -----------------------------------------
with irds.defaults(lang='en'):
    docs = DocTable('aol-ia-docs',
        source=[logs_file, id2wb_file],
        parser=_AolIaDocsParser(),
        count_hint=1_525_586,
    )
    queries = QueryTable('aol-ia-queries',
        source=[logs_file, id2wb_file],
        parser=_AolIaQueriesParser(),
        count_hint=9_966_939,
    )
    qrels = QrelTable('aol-ia-qrels',
        source=[logs_file, id2wb_file],
        parser=_AolIaQrelsParser(),
        defs=QREL_DEFS,
        count_hint=19_442_629,
    )
    qlogs = QlogTable('aol-ia-qlogs',
        source=[logs_file, id2wb_file],
        parser=_AolIaQlogsParser(),
        count_hint=36_389_567,
    )


# Registration
# -----------------------------------------
irds.register(docs, queries, qrels, qlogs)
