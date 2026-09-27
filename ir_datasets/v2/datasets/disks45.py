"""TREC Disks 4 and 5 -- a v2 dataset family.

The corpus (Financial Times, Federal Register, Foreign Broadcast Information
Service, Los Angeles Times -- the Congressional Record is not yet included,
hence "nocr") is LDC/NIST-gated and not automatically downloadable, same
shape as ``trec_arabic.py``/``trec_mandarin.py``: a ``Directory`` Resource
pointed at by ``Source.local()``, at the same well-known path v1 used, so an
existing v1 placement is picked up in place.

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
from ir_datasets.v2 import Directory, Source, TrecDocs, irds

DUA = ("Please confirm you agree to the TREC data usage agreement found at "
       "<https://trec.nist.gov/data/cd45/index.html>")

DATA_ACCESS = (
    "TREC Disks 4 and 5 are distributed by NIST <https://trec.nist.gov/data/cd45/index.html> "
    "under a data usage agreement filed with NIST (an individual or organizational agreement, "
    "depending on whether your organization already has one on file). ir_datasets needs the "
    "FBIS, FR94, FT, and LATIMES directories from the source (the Congressional Record is not "
    "used). Once obtained, copy or symlink them (so this path contains FBIS/, FR94/, FT/, and "
    "LATIMES/ subdirectories) here: {path}"
)

DOCS_LOCAL_PATH = ir_datasets.util.home_path() / 'disks45' / 'corpus'

# Files
# -----------------------------------------
docs_file = Directory('disks45.dir',
    sources=[Source.local(DOCS_LOCAL_PATH, instructions=DATA_ACCESS)],
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
)


# Registration
# -----------------------------------------
irds.register(docs)
