"""Mr. TyDi -- a v2 dataset family.

11 languages, one downloaded tar per language (v1's own per-language tarball,
unpacked by v1 via ``TarExtractAll`` into a persistent directory). Here each
member (the doc collection, and one query/qrels file pair per split) is read
straight off the tar with ``.member(...)`` rather than extracting the whole
thing to a side directory first -- the tar itself is already the one thing
downloaded and cached (see ``nfcorpus.py``'s identical use of ``.member(...)``
off one archive for the same reason: a Resource's ``.stream()``/``.path()``
cache the *download*, so repeated ``.member()`` calls reopen the local tar
rather than re-fetching it).

Reuses v1's ``TsvQueries``/``TrecQrels``/``GenericDoc`` machinery indirectly:
docs are ``GenericDoc``-shaped JSONL (``id``/``contents`` keys, unlike
``miracl.py``'s three-field records), so a small dataset-local Parser wraps
v1's ``JsonlDocs`` with the matching field mapping instead of a new record
type; queries/qrels are exactly the existing v2 ``TsvQueries``/``TrecQrels``
format nodes, no wrapper needed.

v1 also has a bare (whole-topic-set, no single qrels) ``mr-tydi/{lang}``
subset per language, alongside the ``mr-tydi/{lang}/{split}`` splits; not
carried over here (same reasoning as ``beir.py``'s bare BEIR benchmarks --
no single qrels makes it something other than a usable benchmark), so only
the splits are registered, named ``mrtydi-{lang}-{split}`` (the family's own
name is already ``mrtydi`` here, dropping v1's punctuation-heavy ``mr-tydi``,
which reads oddly as a name segment once hyphen-flattened elsewhere).
"""
from typing import NamedTuple

from ir_datasets.formats import JsonlDocs as _V1JsonlDocs

from ir_datasets.v2 import (
    Benchmark, DocTable, Parser, Resource, TrecQrels, TsvQueries, irds,
)

CITATION = ('Zhang et al., 2021, "Mr. TyDi: A Multi-lingual Benchmark for '
           'Dense Retrieval" (arXiv:2108.08787)')

QREL_DEFS = {
    1: "Passage identified within Wikipedia article from top Google search results",
}

#: lang -> (tar's own top-level dir name, url, md5, size) -- verbatim from
#: ir_datasets.datasets.mr_tydi's ``langs`` dict plus etc/downloads.json.
LANGS = {
    'ar': ('mrtydi-v1.0-arabic', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-arabic.tar.gz', 'a0dd1e06c27486b09762c033bde42b70', 321_016_840),
    'bn': ('mrtydi-v1.0-bengali', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-bengali.tar.gz', '06ed183ce7c407f851e5f07370fdbbbb', 59_707_233),
    'en': ('mrtydi-v1.0-english', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-english.tar.gz', '031277b7a7912aedb3a5ae58b93ec2c0', 4_964_684_589),
    'fi': ('mrtydi-v1.0-finnish', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-finnish.tar.gz', '92579475f609bc986aa3eb8260a7328a', 268_373_209),
    'id': ('mrtydi-v1.0-indonesian', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-indonesian.tar.gz', '708610c85ce2953ab281fd66c34d4f84', 168_175_031),
    'ja': ('mrtydi-v1.0-japanese', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-japanese.tar.gz', 'feff865aada3a55cafb8756bd2bf89af', 1_054_574_801),
    'ko': ('mrtydi-v1.0-korean', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-korean.tar.gz', 'ccf88f800e87cb62b735cb283ab6f50c', 222_544_514),
    'ru': ('mrtydi-v1.0-russian', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-russian.tar.gz', 'fab64459133bc93a0bec2f0559bfb423', 1_549_150_495),
    'sw': ('mrtydi-v1.0-swahili', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-swahili.tar.gz', '1042af38a358bd3a60e0f548f9986f8a', 10_452_957),
    'te': ('mrtydi-v1.0-telugu', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-telugu.tar.gz', 'a2174529c3154fe9fa50179cb1584a0d', 73_416_550),
    'th': ('mrtydi-v1.0-thai', 'https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-thai.tar.gz', '351d17d7e8614447f2d350bb736ea718', 112_677_400),
}

SPLITS = ('train', 'dev', 'test')


class MrTydiDoc(NamedTuple):
    doc_id: str
    text: str
    def default_text(self):
        return self.text


class _MrTydiDocsParser(Parser):
    name = 'JsonlDocs'

    def build(self, source, node):
        return _V1JsonlDocs(source, doc_cls=MrTydiDoc,
                            mapping={'doc_id': 'id', 'text': 'contents'},
                            lang=node.lang, count_hint=node.count_hint,
                            docstore_path=str(node.docstore_path))


_benchmarks = []
_aliases = {}

for _lang, (_dir, _url, _md5, _size) in LANGS.items():
    _tar = Resource(f'mrtydi-{_lang}.tar.gz', sources=[_url], md5=_md5, size=_size)

    _docs = DocTable(f'mrtydi-{_lang}-docs',
        source=_tar.member(f'{_dir}/collection/docs.jsonl.gz').gunzip(),
        parser=_MrTydiDocsParser(), lang=_lang,
        desc=f'The Mr. TyDi {_lang} Wikipedia passage corpus.')

    for _split in SPLITS:
        _name = f'mrtydi-{_lang}-{_split}'
        _benchmarks.append(Benchmark(_name,
            docs=_docs,
            queries=TsvQueries(f'{_name}-queries',
                source=_tar.member(f'{_dir}/topic.{_split}.tsv'), lang=_lang),
            qrels=TrecQrels(f'{_name}-qrels',
                source=_tar.member(f'{_dir}/qrels.{_split}.txt'), defs=QREL_DEFS),
            citation=CITATION,
            desc=f'Mr. TyDi {_lang}, {_split} split.'))
        _aliases[f'mr-tydi/{_lang}/{_split}'] = _name


# Registration
# -----------------------------------------
irds.register(*_benchmarks)


# Aliases (old ir-datasets ID mapping)
# -----------------------------------------
irds.alias(_aliases)
