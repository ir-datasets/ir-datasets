"""NYT (The New York Times Annotated Corpus) -- a v2 dataset family.

The corpus is LDC-gated and not automatically downloadable -- like
``disks45.py``, a ``Source.external()``-backed Resource at
``<home>/external/nyt.tgz`` (v1's old ``<home>/nyt/nyt.tgz`` still read as
fallback, so an existing v1 placement is picked up in place). Unlike
``disks45.py`` though, the source is a *single file* (``nyt.tgz``), not a
directory tree, so this is a plain ``Resource`` (not a ``Directory``), same
shape as ``c4.py``'s ``trec_misinfo_2021_queries_file``.

``NytDocs`` (a nested tar-of-tars of NYT XML articles, parsed with
BeautifulSoup) is v1's own handler, reused entirely unmodified via a thin
``Parser`` -- same "thin layer over v1 machinery" precedent as every other
migrated family. ``NytDoc`` is imported from legacy ``nyt.py`` as a record
type only.

**No ``Migrator``.** v1's ``_init()`` wraps the collection in
``Migrator(base_path/'irds_version.txt', 'v2', affected_files=[base_path/
'nyt.tgz.pklz4'], ...)`` -- a purely v1-internal cache-busting mechanism, to
force existing users to rebuild their docstore after a fix to extract full
body text (the ``'v2'`` version string there is v1's own internal migration
counter, unrelated to this package's "v2" architecture -- an unfortunate
naming collision). That migrator exists only to invalidate a *stale*
``nyt.tgz.pklz4`` built by an *older* release of v1's own ``NytDocs``; it says
nothing about this module. ``NytDocs.docs_store()`` still computes its own
cache path internally (``f'{self.docs_path()}.pklz4'``, unchanged here, same
"reuse v1's own path management unmodified" precedent as ``gov.py``/
``c4.py``/``istella22.py``), so a v1 install that already ran the migrator
(as any install using a current v1 release will have) already has a
docstore in the post-migration format, and this module simply picks it up.
Wrapping it in ``Migrator`` again here would just re-run v1's own
already-applied bookkeeping for no benefit -- v2 has no format change of its
own to guard against for this table, since the handler and its cache path
are both reused byte-for-byte from v1. (The one gap: a install that somehow
has a *pre-migration* ``nyt.tgz.pklz4`` lying around, without ever having run
a v1 release new enough to contain the migrator, would need to delete it by
hand before using this module -- judged unlikely enough, this long after the
migration landed in v1, not to be worth carrying ``Migrator`` forward.)

``trec-core-2017``'s benchmark now lives in ``trec_core.py`` (alongside
``wapo.py``'s 2018 year of the same track) -- moved out of here, since a
TREC track's home shouldn't depend on how many corpora happen to be migrated
for it yet, only on whether the track name itself is corpus-agnostic; see
that module's docstring.

``wksup`` (weak supervision) is entirely *derived from the corpus itself*:
v1's ``NytQueries``/``NytQrels`` generate one query per doc (text = headline)
and one self-referential qrel per doc (``doc_id -> doc_id``, relevance 1) by
iterating ``collection.docs_iter()`` -- no separate download. Reused
unmodified via thin ``Parser``s that close over the module-level ``docs``
``DocTable`` and read ``docs.handler`` *inside* ``build()`` (not at
Parser-construction time), so the lazy build in ``Table.handler`` avoids any
import-time ordering/circularity concern. ``source=`` on these two tables is
still ``docs_file`` (the same underlying Resource ``docs`` itself is sourced
from), for a real ``derived_from`` edge back to it -- same shape as
``istella22.py``'s docs/queries/qrels tables all naming ``source_file``
directly even though their parsers actually read through extracted
sub-paths.

``wksup/train`` and ``wksup/valid`` are ``wksup`` filtered by ``VALID_IDS``,
a ~3000-id literal set defined in legacy ``nyt.py`` -- imported directly
(not re-transcribed) and passed straight as ``Filter(query_ids=VALID_IDS,
...)``, which accepts a plain set (see ``filters.py``'s ``Filter``
docstring). No explicit per-split ``QueryTable``/``QrelTable`` is built by
hand; ``Filter`` derives ``irds:nyt-wksup-train-queries``/``-qrels`` (and the
``valid`` equivalents) automatically from ``wksup``'s own tables, same
pattern as ``codec.py``'s/``istella22.py``'s derived benchmarks.
"""
from ir_datasets.datasets.nyt import (
    NytDocs as _V1NytDocs, NytQrels as _V1NytQrels,
    NytQueries as _V1NytQueries, QREL_DEFS, VALID_IDS,
)
from ir_datasets.v2 import (
    Benchmark, DocTable, Filter, QrelTable, QueryTable, Resource, Source, irds,
)
from ir_datasets.v2.formats import Parser

NAME = 'nyt'

DUA = ("The New York Times Annotated Corpus is distributed by the LDC "
       "<https://catalog.ldc.upenn.edu/LDC2008T19> under a data usage "
       "agreement filed with LDC. Many organizations already have an LDC "
       "subscription; check with your library for access details.")

SOURCE_INSTRUCTIONS = (
    "The New York Times Annotated Corpus. It is available from the LDC via: "
    "<https://catalog.ldc.upenn.edu/LDC2008T19>.\n"
    "More details about the procedure can be found here: "
    "<https://ir-datasets.com/nyt.html#DataAccess>.\n"
    "The source file is nyt_corpus_LDC2008T19.tgz.\n"
    "To proceed, place or symlink the source file at: {path}"
)

CITATION_WKSUP = 'dblp:conf/sigir/MacAvaneyYHF19'


class _NytDocsParser(Parser):
    """Wraps v1's ``NytDocs`` directly, unmodified -- no ``Migrator``; see
    the module docstring for why."""
    name = 'NytDocs'

    def build(self, source, node):
        return _V1NytDocs(source)


class _NytQueriesParser(Parser):
    """One query per doc (headline as the query text), reusing v1's
    ``NytQueries`` unmodified. It needs something with a ``docs_iter()``,
    which the sibling ``docs`` table's own v1 handler provides -- referenced
    here inside ``build()`` (not at Parser-construction time) so ``Table.
    handler``'s lazy build has already had a chance to run by the time this
    is called. See the module docstring."""
    name = 'NytQueries'

    def build(self, source, node):
        return _V1NytQueries(docs.handler)


class _NytQrelsParser(Parser):
    """Self-referential qrels (``doc_id -> doc_id``, relevance 1), same
    v1-handler-reuse shape as ``_NytQueriesParser`` above."""
    name = 'NytQrels'

    def build(self, source, node):
        return _V1NytQrels(docs.handler)


# Files
# -----------------------------------------
docs_file = Resource('nyt-source.tgz',
    sources=[Source.external('nyt.tgz', old_locations=[f'{NAME}/nyt.tgz'], instructions=SOURCE_INSTRUCTIONS)],
    hash='md5:67a1bcf200c448424bf0fba34cef17b0',
    dua=DUA,
)
# Tables
# -----------------------------------------
docs = DocTable('nyt',
    source=docs_file,
    parser=_NytDocsParser(),
    lang='en',
    count_hint=1_864_661,
    citation='Sandhaus2008Nyt',
)

wksup_queries = QueryTable('nyt-wksup-queries',
    source=docs_file,
    parser=_NytQueriesParser(),
    lang='en',
    count_hint=1_864_661,
)
wksup_qrels = QrelTable('nyt-wksup-qrels',
    source=docs_file,
    parser=_NytQrelsParser(),
    defs=QREL_DEFS,
    count_hint=1_864_661,
)

# Benchmarks
# -----------------------------------------
wksup = Benchmark('nyt-wksup',
    docs=docs, queries=wksup_queries, qrels=wksup_qrels,
    citation=CITATION_WKSUP,
    desc='Weak supervision from the NYT collection: one query per doc '
         '(headline), self-referential qrels.')

wksup_train = Benchmark('nyt-wksup-train',
    derived_from=wksup,
    filter=Filter(query_ids=VALID_IDS, mode='exclude'),
    citation=CITATION_WKSUP,
    desc='nyt-wksup, excluding the held-out nyt-wksup-valid ids.')

wksup_valid = Benchmark('nyt-wksup-valid',
    derived_from=wksup,
    filter=Filter(query_ids=VALID_IDS, mode='include'),
    citation=CITATION_WKSUP,
    desc='nyt-wksup, held-out validation split.')


# Registration
# -----------------------------------------
irds.register(docs, wksup, wksup_train, wksup_valid)
