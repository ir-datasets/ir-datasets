"""TREC Disks 4 and 5 -- a v2 dataset family.

The corpus (Financial Times, Federal Register, Foreign Broadcast Information
Service, Los Angeles Times -- the Congressional Record is not yet included,
hence "nocr") is LDC/NIST-gated and not automatically downloadable, same
shape as ``trec_arabic.py``/``trec_mandarin.py``: a ``Directory`` Resource
pointed at by ``Source.external()``, at ``<home>/external/trec-disks-4-5`` (old v1
location ``<home>/disks45/corpus`` still read as a fallback).

Unlike those two families, the source is a raw directory tree handed to NIST
as loose files (not a single archive) -- ``Directory`` (not a bare
``Resource``/``File``) is the right node here, and ``TrecDocs``'
``path_globs`` walks it directly (v1's own ``TrecDocs.docs_iter`` already
branches on "is this a directory or a tar" -- see ``formats.py``'s
``_TrecDocsParser``), rather than going through a ``.member()`` pipeline
step.

Only ``disks45-nocr`` is defined here -- the DUA-gated corpus itself.
The TREC 7/8 ad hoc benchmarks and TREC Robust 2004 (all of which use this
corpus but add their own queries/qrels) live in their own files,
``trec_adhoc.py`` and ``trec_robust.py``, which import ``docs`` from here by
reference rather than duplicating it -- same cross-file pattern as
``trec_dl.py`` importing ``docs`` from ``msmarco_document.py``/
``msmarco_passage.py``.
"""
import ir_datasets
from ir_datasets.v2 import Directory, Resource, Source, TrecDocs, irds

DUA = ("Please confirm you agree to the TREC data usage agreement found at "
       "<https://trec.nist.gov/data/cd45/index.html>")

DATA_ACCESS = [
    "TREC Disks 4 and 5 are distributed by NIST: <https://trec.nist.gov/data/cd45/index.html>.\nFile a data usage agreement with NIST (skip if your organization already has one on file).",
    "Gather the FBIS, FR94, FT, and LATIMES directories from the source (the Congressional Record is not used) into one directory.",
]

DOCS_DEFAULT_PATH = 'trec-disks-4-5'
DOCS_OLD_LOCATION = 'disks45/corpus'

# Files
# -----------------------------------------
# Every document file (path, size, sha256) of the FBIS/FR94/FT/LATIMES tree, checked on
# first access by path and size -- see directory_manifest.py. On the mirror
# only (generated from a copy of the corpus, not published upstream).
docs_manifest = Resource('disks45-manifest.jsonl.gz',
    sources=[Source.mirror()],
    hash='md5:952c2a0db862c6982995c5bfbb9a28fb',
    size=110_381,
)

docs_file = Directory('disks45.dir',
    sources=[Source.external(DOCS_DEFAULT_PATH, old_locations=[DOCS_OLD_LOCATION], instructions=DATA_ACCESS)],
    manifest=docs_manifest,
    size=1_997_002_586,  # total of the manifest's files
    dua=DUA,
)

# Tables
# -----------------------------------------
docs = TrecDocs('disks45-nocr',
    source=docs_file,
    path_globs=['**/FBIS/FB*', '**/FR94/??/FR*', '**/FT/*/FT*', '**/LATIMES/LA*'],
    parser='sax',
    expected_file_count=2295,
    lang='en',
    count_hint=528_155,
    citation='Voorhees1996Disks45',
)


# Registration
# -----------------------------------------
irds.register(docs)
