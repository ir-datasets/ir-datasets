"""PMC -- a v2 dataset family (corpus for TREC Clinical Decision Support;
the benchmarks themselves live in ``trec_cds.py`` -- see that module's
docstring for why they're no longer bundled here).

Reuses v1's ``PmcDocs``/``PmcDoc`` handler classes directly as a v2
``parser=`` wrapper, same "thin layer over v1 machinery" every other v2
family uses (see ``highwire.py``/``clinicaltrials.py``). There are two
corpus versions, each its own ``DocTable``:

- v1 (733,111 docs; feeds TREC CDS 2014/2015): 4 tar.gz part-files, plus 2
  "duplicate file name" list-files that v1's ``PmcDocs`` uses to skip
  known-duplicate articles.
- v2 (1,255,260 docs; feeds TREC CDS 2016): the same 4-part shape, but no
  duplicate-file filtering (v1's ``_init()`` leaves ``duplicate_dlcs=[]``
  for this version).

``Table.__init__`` only accepts one ``source=`` value, so (same as
``highwire.py``'s 58 corpus zips + legalspans file) each table's Resources
are passed as a single flat list -- structural edges are just "every
Resource this source expression bottoms out in" (``nodes.source_resources``
walks lists but not dicts) -- and a ``Parser`` splits the flat list back
into ``(dlcs, duplicate_dlcs)`` before handing it to v1. One ``Parser``
instance is used per table, parameterized by how many trailing entries in
its own ``source=`` list are duplicate-file Resources (2 for v1, 0 for v2),
so each table's ``source=`` list and its own ``Parser`` stay in sync.

v1's ``PmcDocs`` takes its own ``path`` positional argument (not just a
name) and manages that docstore path entirely itself -- kept unmodified
here, same "reuse v1's own path management" precedent as
``clinicaltrials.py``/``codec.py``, rather than routing through v2's
``DocTable.docstore_path``.

"""
import ir_datasets
from ir_datasets.datasets.pmc import PmcDocs as _V1PmcDocs
from ir_datasets.v2 import DocTable, Resource, irds
from ir_datasets.v2.formats import Parser

NAME = 'pmc'

class _PmcDocsParser(Parser):
    name = 'PmcDocs'

    def __init__(self, path, n_duplicates=0):
        self.path = path
        self.n_duplicates = n_duplicates

    def build(self, source, node):
        if self.n_duplicates:
            dlcs, duplicate_dlcs = source[:-self.n_duplicates], source[-self.n_duplicates:]
        else:
            dlcs, duplicate_dlcs = source, []
        return _V1PmcDocs(dlcs, self.path, duplicate_dlcs=duplicate_dlcs, count_hint=node.count_hint)


with irds.defaults(lang='en'):
    # Files
    # -----------------------------------------
    v1_source_files = [
        Resource('pmc-v1-text-00.tar.gz',
            sources=['https://ceb.nlm.nih.gov/~simpsonmatt/pmc-text-00.tar.gz'],
            md5='3448bc2967913dd46c771d55b3b3ccae',
            size=1_958_971_361),
        Resource('pmc-v1-text-01.tar.gz',
            sources=['https://ceb.nlm.nih.gov/~simpsonmatt/pmc-text-01.tar.gz'],
            md5='6f4bee841dcf307ce1f4cd137f65e822',
            size=3_038_214_436),
        Resource('pmc-v1-text-02.tar.gz',
            sources=['https://ceb.nlm.nih.gov/~simpsonmatt/pmc-text-02.tar.gz'],
            md5='03375c91a74220c59071c7aa5f78deee',
            size=3_133_121_435),
        Resource('pmc-v1-text-03.tar.gz',
            sources=['https://ceb.nlm.nih.gov/~simpsonmatt/pmc-text-03.tar.gz'],
            md5='270648367547f746cf1e2e4323a46aa9',
            size=2_161_685_233),
    ]
    v1_duplicate_files = [
        Resource('pmc-v1-duplicates-1.txt',
            sources=['http://www.trec-cds.org/duplicates-1.txt'],
            md5='7a93656ca21c1749bf0b71a03be01cf3',
            size=3_836),
        Resource('pmc-v1-duplicates-2.txt',
            sources=['http://www.trec-cds.org/duplicates-2.txt'],
            md5='8d5f1004ea4d00cfd5a96b40c6419be1',
            size=2_640),
    ]
    v2_source_files = [
        Resource('pmc-v2-text-00.tar.gz',
            sources=['https://ceb.nlm.nih.gov/~robertske/pmc-00.tar.gz'],
            md5='d66f2e243cb697138753e622f0a1867a',
            size=3_875_701_879),
        Resource('pmc-v2-text-01.tar.gz',
            sources=['https://ceb.nlm.nih.gov/~robertske/pmc-01.tar.gz'],
            md5='e59422e5d21ef7e1be307a0de61626f7',
            size=3_590_186_941),
        Resource('pmc-v2-text-02.tar.gz',
            sources=['https://ceb.nlm.nih.gov/~robertske/pmc-02.tar.gz'],
            md5='145d194643818ea09a4947ba6b1c91c7',
            size=4_991_488_701),
        Resource('pmc-v2-text-03.tar.gz',
            sources=['https://ceb.nlm.nih.gov/~robertske/pmc-03.tar.gz'],
            md5='231de843bb5334c3c885d75f2ca3240b',
            size=6_443_033_629),
    ]

    # Tables
    # -----------------------------------------
    v1_docs = DocTable('pmc-v1-docs',
        source=[*v1_source_files, *v1_duplicate_files],
        parser=_PmcDocsParser(ir_datasets.util.home_path() / NAME / 'v1' / 'corpus', n_duplicates=2),
        count_hint=733_111,
    )
    v2_docs = DocTable('pmc-v2-docs',
        source=v2_source_files,
        parser=_PmcDocsParser(ir_datasets.util.home_path() / NAME / 'v2' / 'corpus', n_duplicates=0),
        count_hint=1_255_260,
    )


# Registration
# -----------------------------------------
irds.register(v1_docs, v2_docs)
