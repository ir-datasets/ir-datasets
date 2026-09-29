"""AQUAINT -- a v2 dataset family.

The corpus (Xinhua, New York Times, and AP Worldstream English newswire) is
LDC-gated (LDC2002T31) and not automatically downloadable -- same shape as
``trec_arabic.py``/``trec_mandarin.py``: a single tgz ``Resource`` pointed at
by ``Source.external()``, at ``<home>/external/aquaint.tgz`` (the old
v1 location ``<home>/aquaint/aquaint_comp_LDC2002T31.tgz`` is still read as a fallback). Unlike ``disks45.py``'s corpus, NIST
distributes this one as a single archive (not a loose directory tree), so a
plain ``Resource`` is the right node here, not ``Directory``.

TREC Robust 2005 (a 50-"hard"-topic subset of TREC Robust 2004's query set,
judged against this corpus) lives in ``trec_robust.py`` alongside TREC
Robust 2004, and imports ``docs`` from here by reference rather than
duplicating it -- same cross-file pattern as ``trec_adhoc.py``/
``trec_robust.py`` importing ``docs`` from ``disks45.py``.
"""
import ir_datasets
from ir_datasets.v2 import Resource, Source, TrecDocs, irds

DATA_ACCESS = [
    "Find the AQUAINT collection in the LDC catalog: <https://catalog.ldc.upenn.edu/LDC2002T31> (aquaint_comp_LDC2002T31.tgz).\nMany organizations already have an LDC subscription; check with your library for access.",
    "Download aquaint_comp_LDC2002T31.tgz.",
]

DOCS_MD5 = 'ac623257d8dd35326c9d500d5f6834e5'
DOCS_DEFAULT_PATH = 'aquaint.tgz'
DOCS_OLD_LOCATION = 'aquaint/aquaint_comp_LDC2002T31.tgz'

# Files
# -----------------------------------------
docs_file = Resource('aquaint.tgz',
    sources=[Source.external(DOCS_DEFAULT_PATH, old_locations=[DOCS_OLD_LOCATION], instructions=DATA_ACCESS)],
    hash=f'md5:{DOCS_MD5}',
    size=1_073_936_915,
)

# Tables
# -----------------------------------------
docs = TrecDocs('aquaint',
    source=docs_file,
    encoding='utf8',
    path_globs=['aquaint_comp/apw/*/*.gz', 'aquaint_comp/nyt/*/*.gz', 'aquaint_comp/xie/*/*.gz'],
    lang='en',
    count_hint=1_033_461,
    citation='Graff2002Aquaint',
)


# Registration
# -----------------------------------------
irds.register(docs)
