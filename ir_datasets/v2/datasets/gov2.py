"""GOV2 -- a v2 dataset family (the corpus judged by the TREC Terabyte and
Million Query tracks).

The corpus is the .gov web crawl used by the TREC Terabyte tracks
(2004-2006), distributed by the University of Glasgow on physical hard
drives under an individual/organizational data agreement -- not
downloadable, same shape as ``disks45.py``: a ``Directory`` Resource pointed
at by ``Source.local()``, at the same well-known path v1 used, so an
existing v1 placement is picked up in place. Unlike disks45, the on-disk
layout is v1's own bespoke ``GOV2_data/GX???/*.gz`` tree with a companion
``GOV2_extras/url2id.gz`` used to build a per-file document-count index (for
efficient random slicing) -- not a set of ``TrecDocs``-compatible SGML files
walked by ``path_globs``. v1's ``Gov2Docs``/``Gov2Docstore``/
``Gov2DocCountFile`` handler classes are reused directly as a ``parser=``
wrapper (same "thin layer over v1 machinery" as ``car.py``), with
``Gov2Docs`` subclassed locally only to point its docstore at
``node.docstore_path`` instead of v1's fixed ``<source path>.cache`` -- same
reason ``car.py`` subclasses ``CarDocs``.

The TREC Terabyte (2004-2006) and Million Query (2007-2008) queries/qrels
that judge this corpus live in their own files, ``trec_tb.py`` and
``trec_mq.py``, which import ``docs``/``DUA`` from here by reference rather
than duplicating them -- same cross-file pattern as ``trec_adhoc.py``
importing ``docs``/``DUA`` from ``disks45.py``.
"""
import ir_datasets
from ir_datasets.datasets.gov2 import Gov2DocCountFile
from ir_datasets.datasets.gov2 import Gov2Docs as _V1Gov2Docs
from ir_datasets.datasets.gov2 import Gov2Docstore as _V1Gov2Docstore
from ir_datasets.indices import DEFAULT_DOCSTORE_OPTIONS, CacheDocstore
from ir_datasets.v2 import DocTable, Directory, Source, irds
from ir_datasets.v2.formats import Parser

DUA = ("Please confirm you have (or your organization has) filed a data usage "
       "agreement with the University of Glasgow for GOV2, as described at "
       "<http://ir.dcs.gla.ac.uk/test_collections/access_to_data.html>")

DATA_ACCESS = (
    "GOV2 is distributed by the University of Glasgow on hard drives shipped "
    "under a data usage agreement (individual or organizational, depending "
    "on whether your organization already has one on file); see "
    "<http://ir.dcs.gla.ac.uk/test_collections/access_to_data.html>. Once "
    "obtained, copy or symlink the GOV2_data (and GOV2_extras) directories "
    "here: {path}"
)

DOCS_LOCAL_PATH = ir_datasets.util.home_path() / 'gov2' / 'corpus'
DOCCOUNT_LOCAL_PATH = ir_datasets.util.home_path() / 'gov2' / 'corpus.doccounts'


class _Gov2Docs(_V1Gov2Docs):
    """v1 Gov2Docs, but with the docstore where the node wants it (see the
    module docstring -- v1's fixed ``<source path>.cache`` would collide
    across providers/versions the way ``car.py``'s ``_CarDocs`` explains)."""

    def __init__(self, *args, store_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._store_path = store_path

    def docs_store(self, options=DEFAULT_DOCSTORE_OPTIONS):
        docstore = _V1Gov2Docstore(self)
        return CacheDocstore(docstore, str(self._store_path), options=options)


class _Gov2DocsParser(Parser):
    name = 'Gov2Docs'

    def build(self, source, node):
        doccount = Gov2DocCountFile(str(DOCCOUNT_LOCAL_PATH), source)
        return _Gov2Docs(source, doccount, store_path=node.docstore_path)


# Files
# -----------------------------------------
docs_file = Directory('gov2.dir',
    sources=[Source.local(DOCS_LOCAL_PATH, instructions=DATA_ACCESS)],
    dua=DUA,
)

# Tables
# -----------------------------------------
docs = DocTable('gov2',
    source=docs_file,
    parser=_Gov2DocsParser(),
    lang='en',
    count_hint=25_205_179,
)


# Registration
# -----------------------------------------
irds.register(docs)
