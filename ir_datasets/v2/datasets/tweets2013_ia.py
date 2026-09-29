"""tweets2013-ia -- a v2 dataset family (corpus for TREC Microblog
2013/2014, using tweets re-distributed by the Internet Archive rather than
the original, long-since-restricted Twitter streaming API dumps; the
benchmarks themselves live in ``trec_mb.py`` -- see that module's docstring
for why they're no longer bundled here).

The corpus is fetched from two large Internet Archive tar files (February and
March 2013) and reassembled into a from-scratch, lookup-optimized on-disk
layout (bucketed by minute, lz4-compressed, with short JSON checkpoint records
enabling skip-ahead binary-search-style lookups) -- see v1's own
``Tweets2013IaDocs`` module docstring for the full rationale. That handler
(``ir_datasets.datasets.tweets2013_ia.Tweets2013IaDocs``) is reused completely
unmodified here via a thin ``Parser`` wrapper, same "thin layer over v1
machinery" pattern as ``clinicaltrials.py``/``highwire.py``/``aol_ia.py`` --
it manages its own on-disk build/cache/docstore state at a fixed, v1-style
base path (``<home>/tweets2013-ia/corpus``) rather than through v2's
per-node ``docstore_path``, the same "keep the existing stateful path/caching
logic as-is" precedent those modules set. The two source tar files are large
archive.org downloads with no declared DUA (unlike TREC's own restricted
Twitter corpora), so they are ordinary (if huge, and not ``--verify``'d here
for the same reason) ``Resource``s.
"""
import ir_datasets
from ir_datasets.datasets.tweets2013_ia import Tweets2013IaDocs as _V1Tweets2013IaDocs
from ir_datasets.v2 import DocTable, Resource, irds
from ir_datasets.v2.formats import Parser

NAME = 'tweets2013-ia'
BASE_PATH = ir_datasets.util.home_path() / NAME


class _Tweets2013IaDocsParser(Parser):
    name = 'Tweets2013IaDocs'

    def build(self, source, node):
        return _V1Tweets2013IaDocs(source, str(BASE_PATH / 'corpus'))


# Files
# -----------------------------------------
docs_feb_file = Resource('tweets2013-ia-docs-feb.tar',
    sources=['https://archive.org/download/archiveteam-twitter-stream-2013-02/archiveteam-twitter-stream-2013-02.tar'],
    md5='e82916d37116c781afff750e2127156f',
)
docs_mar_file = Resource('tweets2013-ia-docs-mar.tar',
    sources=['https://archive.org/download/archiveteam-twitter-stream-2013-03/archiveteam-twitter-stream-2013-03.tar'],
    md5='486817a372e03298162daa0e80dc6399',
)

# Tables
# -----------------------------------------
docs = DocTable('tweets2013-ia',
    source=[docs_feb_file, docs_mar_file],
    parser=_Tweets2013IaDocsParser(),
    lang=None,  # multiple languages
    count_hint=252_713_133,
    citation='dblp:conf/sigir/SequieraL17',
)


# Registration
# -----------------------------------------
irds.register(docs)
