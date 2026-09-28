"""GOV -- a v2 dataset family (the corpus judged by the TREC Web Track
2002-2004).

The corpus (a .gov web crawl, not to be confused with the much larger
``gov2``) is distributed by the University of Glasgow on physical hard
drives under a data usage agreement, same shape as ``disks45.py``: a raw
directory tree (``G00``, ``G01``, ...) handed over as loose files, not a
single archive -- ``Directory`` + ``Source.local()`` pointed at the same
well-known path v1 used.

Unlike ``disks45-nocr``, the per-document format here is not TREC SGML read
through v2's built-in ``TrecDocs``/BS4-or-sax parser: v1 wrote its own
hand-rolled ``<DOC>``/``<DOCNO>``/``<DOCHDR>`` block scanner (``GovDocs``,
in ``ir_datasets/datasets/gov.py``) for speed, and it consumes the directory
by walking ``G??/*.gz`` itself rather than through ``TrecDocs``'
``path_globs``. So this reuses v1's ``GovDocs`` handler directly as a v2
``parser=`` wrapper -- same "thin layer over v1 machinery" as
``car.py``/``codec.py``/``clinicaltrials.py`` -- subclassed locally
(``_GovDocs``) only to point its docstore at ``node.docstore_path`` (v1's
own ``docs_store`` hardcodes ``<docs_dlc.path>.pklz4``, which would put the
index inside the DUA-gated source tree) and to take the node's own
``count_hint`` instead of v1's fixed per-``NAME`` global lookup.

The TREC Web Track queries/qrels that judge this corpus live in their own
file, ``trec_web.py``, which imports ``docs``/``DUA`` from here by
reference rather than duplicating them -- same cross-file pattern as
``trec_adhoc.py`` importing ``docs``/``DUA`` from ``disks45.py``.
"""
import ir_datasets
from ir_datasets.datasets.gov import GovDocs as _V1GovDocs
from ir_datasets.indices import DEFAULT_DOCSTORE_OPTIONS, PickleLz4FullStore
from ir_datasets.v2 import DocTable, Directory, Source, irds
from ir_datasets.v2.formats import Parser

DUA = ("Please confirm you agree to the data usage terms found at "
       "<http://ir.dcs.gla.ac.uk/test_collections/access_to_data.html>")

DATA_ACCESS = (
    "GOV is distributed by the University of Glasgow "
    "<http://ir.dcs.gla.ac.uk/test_collections/access_to_data.html> as a hard drive shipment, "
    "under an individual or organizational data usage agreement filed with UoG. Once obtained, "
    "copy or symlink the G00, G01, G02, ... directories here: {path}"
)

DOCS_LOCAL_PATH = ir_datasets.util.home_path() / 'gov' / 'corpus'


class _GovDocs(_V1GovDocs):
    """v1 GovDocs, but with the docstore where the node wants it and a
    count_hint that comes from the node rather than v1's fixed per-``NAME``
    global lookup (see the module docstring)."""

    def __init__(self, docs_dlc, count_hint=None, store_path=None):
        super().__init__(docs_dlc)
        self._v2_count_hint = count_hint
        self._store_path = store_path

    def docs_store(self, field='doc_id', options=DEFAULT_DOCSTORE_OPTIONS):
        return PickleLz4FullStore(
            path=str(self._store_path),
            init_iter_fn=self._docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=['doc_id'],
            count_hint=self._v2_count_hint,
            options=options,
        )


class _GovDocsParser(Parser):
    name = 'GovDocs'

    def build(self, source, node):
        # source is the Directory node itself -- GovDocs only ever calls
        # .path(force) on it (see ir_datasets/datasets/gov.py), same as
        # disks45's TrecDocs.path_globs walking the Directory directly.
        return _GovDocs(source, count_hint=node.count_hint, store_path=node.docstore_path)


# Files
# -----------------------------------------
docs_file = Directory('gov.dir',
    sources=[Source.local(DOCS_LOCAL_PATH, instructions=DATA_ACCESS)],
    dua=DUA,
)

# Tables
# -----------------------------------------
docs = DocTable('gov',
    source=docs_file,
    parser=_GovDocsParser(),
    lang='en',
    count_hint=1_247_753,
)


# Registration
# -----------------------------------------
irds.register(docs)
