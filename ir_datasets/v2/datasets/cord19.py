"""CORD-19 -- a v2 dataset family (corpus for TREC-COVID; the benchmarks
themselves live in ``trec_covid.py`` -- see that module's docstring for why
they're no longer bundled here).

CORD-19 is a corpus of COVID-19-related scientific articles, released by
Semantic Scholar as a series of dated snapshots throughout 2020.

Reuses v1's ``Cord19Docs``/``Cord19Doc``/``Cord19FullTextDoc``/
``Cord19FullTextSection`` completely unmodified via a thin ``Parser``
wrapper, same "thin layer over v1 machinery" pattern as every other migrated
family (see ``clinicaltrials.py``/``csl.py``). ``Cord19Docs`` only ever calls
``.stream()``/``.path()`` on the ``streamer`` it's given (confirmed by
re-reading ``docs_path()``/``_docs_iter()`` in ``ir_datasets/datasets/
cord19.py``), so a v2 ``Resource`` satisfies it exactly the way a v1 dlc
entry did -- no subclassing needed (unlike ``car.py``'s ``_CarDocs``, which
overrides ``docs_store()`` to avoid a v1 path collision; ``Cord19Docs``
already keys its docstore off ``streamer.path()``, which differs per
snapshot/mode, so there's no collision to avoid here).

``Cord19Docs`` has two distinct modes, controlled by ``include_fulltext``:

* metadata-only (the default): a single ``metadata.csv`` is streamed
  directly (no local extraction) via ``csv.DictReader`` -- cheap, and what
  every dated snapshot below except the full-text one uses.
* full-text (``include_fulltext=True``): the (much bigger) corpus tar.gz is
  first extracted to a fixed local directory (``extr_path``) if not already
  present, then ``metadata.csv`` plus per-article JSON bodies are read from
  that extracted tree. Every snapshot has a ``-fulltext`` table (an
  ``irds:alternative_of`` its metadata-only one; v1 only offered 2020-07-16) --
  ``extr_path`` is a plain local directory under
  ``ir_datasets.util.home_path()``, same convention as ``clueweb09.py``'s
  ``corpus.chk`` / ``c4.py``'s checkpoints extraction.

Five distinct metadata-only corpus snapshots are registered, by date
(2020-04-10, 2020-05-01, 2020-05-19, 2020-06-19, 2020-07-16), each with a
full-text variant -- named after their date
rather than the TREC-COVID round number that (soley) used to judge them
(v1's own ``round1``-``round5`` subset names), since the corpus itself has
no notion of "rounds" -- that's purely a TREC-COVID judging concept, now
that the benchmarks live in ``trec_covid.py``.
"""
import ir_datasets
from ir_datasets.datasets.cord19 import Cord19Docs as _V1Cord19Docs
from ir_datasets.v2 import DocTable, Resource, irds
from ir_datasets.v2.formats import Parser

NAME = 'cord19'

BASE_PATH = ir_datasets.util.home_path() / NAME

CITATION_CORD19 = 'dblp:journals/corr/abs-2004-10706'


class _Cord19DocsParser(Parser):
    name = 'Cord19Docs'

    def __init__(self, extr_path, date, include_fulltext=False):
        self.extr_path = extr_path
        self.date = date
        self.include_fulltext = include_fulltext

    def build(self, source, node):
        return _V1Cord19Docs(source, self.extr_path, self.date,
            include_fulltext=self.include_fulltext, count_hint=node.count_hint)


# license verified 2026-09-30: https://github.com/allenai/cord19/blob/master/LICENSE (COVID Dataset License Agreement; custom, text-and-data-mining only)
with irds.defaults(lang='en', license='https://github.com/allenai/cord19/blob/master/LICENSE'):
    # Files
    # -----------------------------------------
    docs_2020_07_16_metadata_file = Resource('cord19-2020-07-16-metadata.csv',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/metadata.csv'],
        hash='md5:80d664e496b8b7e50a39c6f6bb92e0ef',
        size=269_219_095,
    )
    docs_2020_07_16_file = Resource('cord19-2020-07-16.tar.gz',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-07-16.tar.gz'],
        hash='md5:018c4bc4d76d4ae072a26ac28c8b456b',
        size=3_662_861_028,
    )
    docs_2020_04_10_metadata_file = Resource('cord19-2020-04-10-metadata.csv',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-04-10/metadata.csv'],
        hash='md5:42a21f386be86c24647a41bedde34046',
        size=77_323_567,
    )
    docs_2020_05_01_metadata_file = Resource('cord19-2020-05-01-metadata.csv',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-05-01/metadata.csv'],
        hash='md5:b1d2e409026494e0c8034278bacd1248',
        size=89_290_114,
    )
    docs_2020_05_19_metadata_file = Resource('cord19-2020-05-19-metadata.csv',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-05-19/metadata.csv'],
        hash='md5:e3c5c8af3a078e19cb179e630c345959',
        size=189_687_667,
    )
    docs_2020_06_19_metadata_file = Resource('cord19-2020-06-19-metadata.csv',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-06-19/metadata.csv'],
        hash='md5:4e8788b6e44f3428ff9ab1d4bfdfb6ab',
        size=228_730_850,
    )
    # (the 2020-04-10 and 2020-05-01 releases are bzip2, despite their .tar.gz URLs)
    docs_2020_04_10_file = Resource('cord19-2020-04-10.tar.bz2',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-04-10.tar.gz'],
        hash='md5:f4c3e742af7a6d6907ac86b1ca9f5312',
        size=1_516_963_708,
    )
    docs_2020_05_01_file = Resource('cord19-2020-05-01.tar.bz2',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-05-01.tar.gz'],
        hash='md5:e8c56920c612b89e20b54f9f5b02c992',
        size=1_742_677_535,
    )
    docs_2020_05_19_file = Resource('cord19-2020-05-19.tar.gz',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-05-19.tar.gz'],
        hash='md5:6424de9c3bdf74b889df74f9b71d5cbc',
        size=2_793_628_709,
    )
    docs_2020_06_19_file = Resource('cord19-2020-06-19.tar.gz',
        sources=['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-06-19.tar.gz'],
        hash='md5:47b61215768b6fa72d7a152757d38d96',
        size=3_336_210_521,
    )

    # Tables
    # -----------------------------------------
    docs_2020_07_16 = DocTable('cord19-2020-07-16',
        source=docs_2020_07_16_metadata_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-07-16', '2020-07-16'),
        count_hint=192_509,
        citation=CITATION_CORD19,
    )
    docs_2020_07_16_fulltext = DocTable('cord19-2020-07-16-fulltext',
        source=docs_2020_07_16_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-07-16.fulltext', '2020-07-16', include_fulltext=True),
        count_hint=192_509,
        citation=CITATION_CORD19,
        metadata={'alternative_note': 'adds article full text'},
    )
    docs_2020_04_10 = DocTable('cord19-2020-04-10',
        source=docs_2020_04_10_metadata_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-04-10', '2020-04-10'),
        count_hint=51_078,
        citation=CITATION_CORD19,
    )
    docs_2020_05_01 = DocTable('cord19-2020-05-01',
        source=docs_2020_05_01_metadata_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-05-01', '2020-05-01'),
        count_hint=59_887,
        citation=CITATION_CORD19,
    )
    docs_2020_05_19 = DocTable('cord19-2020-05-19',
        source=docs_2020_05_19_metadata_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-05-19', '2020-05-19'),
        count_hint=128_492,
        citation=CITATION_CORD19,
    )
    docs_2020_06_19 = DocTable('cord19-2020-06-19',
        source=docs_2020_06_19_metadata_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-06-19', '2020-06-19'),
        count_hint=158_274,
        citation=CITATION_CORD19,
    )

    docs_2020_04_10_fulltext = DocTable('cord19-2020-04-10-fulltext',
        source=docs_2020_04_10_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-04-10.fulltext', '2020-04-10', include_fulltext=True),
        count_hint=51_078,
        citation=CITATION_CORD19,
        metadata={'alternative_note': 'adds article full text'},
    )
    docs_2020_05_01_fulltext = DocTable('cord19-2020-05-01-fulltext',
        source=docs_2020_05_01_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-05-01.fulltext', '2020-05-01', include_fulltext=True),
        count_hint=59_887,
        citation=CITATION_CORD19,
        metadata={'alternative_note': 'adds article full text'},
    )
    docs_2020_05_19_fulltext = DocTable('cord19-2020-05-19-fulltext',
        source=docs_2020_05_19_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-05-19.fulltext', '2020-05-19', include_fulltext=True),
        count_hint=128_492,
        citation=CITATION_CORD19,
        metadata={'alternative_note': 'adds article full text'},
    )
    docs_2020_06_19_fulltext = DocTable('cord19-2020-06-19-fulltext',
        source=docs_2020_06_19_file,
        parser=_Cord19DocsParser(BASE_PATH / '2020-06-19.fulltext', '2020-06-19', include_fulltext=True),
        count_hint=158_274,
        citation=CITATION_CORD19,
        metadata={'alternative_note': 'adds article full text'},
    )

# Registration
# -----------------------------------------
irds.register(docs_2020_07_16, docs_2020_07_16_fulltext,
              docs_2020_04_10, docs_2020_05_01, docs_2020_05_19, docs_2020_06_19,
              docs_2020_04_10_fulltext, docs_2020_05_01_fulltext,
              docs_2020_05_19_fulltext, docs_2020_06_19_fulltext)

# The full-text docs share doc_ids (cord_uid) with the metadata-only ones, so
# benchmarks over the latter (trec-covid, its rounds) can swap them in -- see
# ``Benchmark.replace`` / ``irds:alternative_of``.
for _date in ('2020-04-10', '2020-05-01', '2020-05-19', '2020-06-19', '2020-07-16'):
    _v = _date.replace('-', '_')
    irds.add_edge(globals()[f'docs_{_v}_fulltext'], 'alternative_of', globals()[f'docs_{_v}'])
