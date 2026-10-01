"""HC4 -- a v2 dataset family.

Three CLIR test collections (Chinese, Persian, Russian Common Crawl web
pages), evaluated against English topics. Each language's documents are a
single gzip'd jsonl downloaded from HuggingFace (``neuclir/hc4``) and read
through ``.gunzip()`` -- hashed with sha256 (the hub's git-LFS digest) rather
than md5.

Topics are one shared file per split (train/dev/test), not per language --
``ExctractedCCQueries``' ``subset_lang=`` picks each language's own
title/description (and any human/machine translations) out of the same
record, exactly as v1 does; reused directly as a v2 ``parser=`` wrapper.

The per-language ``*_ids`` Resources at the bottom (gzip'd id lists, no
Table wrapping them -- they are not documents or queries, just an
id-membership index) are HC4's own published artifacts, not consumed by
anything in this file. They exist so ``neuclir.py``'s ``hc4-filtered``
variant (NeuCLIR's own subset that intersects with HC4) can import them
directly -- the same cross-file shared-reference pattern
``msmarco_document.py`` uses for MS MARCO's TREC-DL query tables, just for a
Resource instead of a Table.
"""
import ir_datasets
from ir_datasets.formats import ExctractedCCDocs as _V1ExctractedCCDocs
from ir_datasets.formats import ExctractedCCQueries as _V1ExctractedCCQueries

from ir_datasets.v2 import (
    Benchmark, DocTable, Parser, QueryTable, Resource, TrecQrels, irds,
)

CITATION = 'dblp:conf/ecir/LawrieMOY22'

#: lang -> (docs url, sha256, size), from HuggingFace's neuclir/hc4 (the hub
#: publishes git-LFS sha256 digests, not md5).
_HF = 'https://huggingface.co/datasets/neuclir/hc4/resolve/main/data'
DOCS = {
    'zh': (f'{_HF}/zho-00000-of-00001.jsonl.gz', '06498130be194e758779b73748e66cbbdbd9e5a13361ea05da617722c9643c15', 576_134_453),
    'fa': (f'{_HF}/fas-00000-of-00001.jsonl.gz', 'b15f68b9e0b530f3ba5ad6c586a3a8b4288ce833468d87af9ed2ce07ca8202d1', 518_142_839),
    'ru': (f'{_HF}/rus-00000-of-00001.jsonl.gz', 'bfa3d80f404c0bacfa3321628a605c05ee965ebf7f70951e666b26dcd52e1211', 4_138_605_072),
}

#: v1's 2-letter -> 3-letter language code (also HC4's own github path segment).
LANG3 = {'zh': 'zho', 'fa': 'fas', 'ru': 'rus'}

DOC_COUNTS = {'zh': 646_305, 'fa': 486_486, 'ru': 4_721_064}

QREL_DEFS = {
    3: 'Very-valuable. Information in the document would be found in the lead paragraph of a report that is later written on the topic.',
    1: 'Somewhat-valuable. The most valuable information in the document would be found in the remainder of such a report.',
    0: 'Not-valuable. Information in the document might be included in a report footnote, or omitted entirely.',
}

_GH = 'https://raw.githubusercontent.com/hltcoe/HC4/main/resources/hc4'

#: split -> (topics md5, size) -- one shared file per split, all 3 languages.
TOPICS = {
    'train': ('cf3a43c4085e28ce03f37704771e0e36', 104_898),
    'dev': ('4c34c546c3e90de4733c3c9411ce9c6d', 81_444),
    'test': ('a311237913a7335d45fe261e9ff7f11e', 964_434),
}

#: (lang, split) -> (qrels md5, size)
QRELS = {
    ('fa', 'train'): ('4a343957837ce996a7275a71a30ed806', 5_152),
    ('fa', 'dev'): ('d66b62e3733a66b151f895e2f57ee144', 24_574),
    ('fa', 'test'): ('842c307e3e01a688897fadefc7a9672f', 113_490),
    ('zh', 'train'): ('67d956dcc5b373ae8b6ce5f360b03987', 15_686),
    ('zh', 'dev'): ('f252fb5edeee1fa38ccc8ee1c2a6e6f0', 20_126),
    ('zh', 'test'): ('1bc5cfcefc49805884142b8b32f1a6ea', 123_795),
    ('ru', 'train'): ('2e2c52e404a6ee0fe676cb88d63c26bb', 4_232),
    ('ru', 'dev'): ('24305749fd39fb0392be171e76ef2510', 11_532),
    ('ru', 'test'): ('1d1112350a11289496426ea558c321a2', 133_650),
}

#: lang -> [(md5, size), ...] id-list shard(s) (russian ships 8; the others 1).
IDS = {
    'fa': [('553e510633c30ce783c22ed37471ed3a', 21_890_148)],
    'zh': [('d4ee35f9ca55c0416fe439d4f41a9e2a', 29_051_136)],
    'ru': [
        ('4763df966f6ea953c731ef2d572044e5', 26_794_412),
        ('c19fb0dd1aceb0f6fd02f92818fa55b7', 26_784_412),
        ('41d6db2ae68b8a4a1e2b371e4f5fe7a8', 26_771_449),
        ('e3d20167c9fdce77e633b3ea0421cb51', 26_765_684),
        ('54db61aec1a4585ce172c39111725be7', 26_790_863),
        ('ba8a7bace2df0be82f80f7ae84f736d5', 26_802_522),
        ('9ada14526d375e2c7aaf95be80f8a043', 26_793_985),
        ('8555423b846aaf097527017cf8eda94c', 23_266_384),
    ],
}


class _Hc4DocsParser(Parser):
    name = 'ExctractedCCDocs'

    def __init__(self, lang):
        self.lang = lang

    def build(self, source, node):
        return _V1ExctractedCCDocs(source, subset_lang=self.lang,
                                   count=DOC_COUNTS[self.lang],
                                   docstore_path=str(node.docstore_path))


class _Hc4QueriesParser(Parser):
    name = 'ExctractedCCQueries'

    def __init__(self, lang):
        self.lang = lang

    def build(self, source, node):
        return _V1ExctractedCCQueries(source, subset_lang=self.lang)


TOPICS_FILES = {
    split: Resource(f'hc4-{split}-topics.jsonl',
        sources=[f'{_GH}/{split}.topics.v1-0.jsonl'], hash=f'md5:{md5}', size=size)
    for split, (md5, size) in TOPICS.items()
}

def _ids_url(lang, i, n_shards):
    suffix = '' if n_shards == 1 else f'.{i}'
    return f'{_GH}/{LANG3[lang]}/ids{suffix}.jsonl.gz'


#: lang -> [Resource, ...] -- exported for neuclir.py's hc4-filtered variant.
IDS_FILES = {
    lang: [
        Resource(f'hc4-{lang}-ids-{i}.jsonl.gz',
            sources=[_ids_url(lang, i, len(shards))], hash=f'md5:{md5}', size=size)
        for i, (md5, size) in enumerate(shards)
    ]
    for lang, shards in IDS.items()
}

#: (lang, split) -> Resource -- exported for neuclir.py's hc4-filtered variant
#: (its qrels are HC4's own dev+test qrels, restricted to the shared doc ids).
QRELS_FILES = {
    (lang, split): Resource(f'hc4-{lang}-{split}-qrels.txt',
        sources=[f'{_GH}/{LANG3[lang]}/{split}.qrels.v1-0.txt'], hash=f'md5:{md5}', size=size)
    for (lang, split), (md5, size) in QRELS.items()
}

_benchmarks = []

for _lang, (_url, _sha, _size) in DOCS.items():
    # license verified 2026-09-30: https://huggingface.co/datasets/neuclir/hc4 (card metadata: license odc-by); applies to the document corpus only
    _docs_file = Resource(f'hc4-{_lang}-docs.jsonl.gz', sources=[_url], hash=f'sha256:{_sha}', size=_size,
        license='ODC-By-1.0')
    _docs = DocTable(f'hc4-{_lang}-docs',
        source=_docs_file.gunzip(), parser=_Hc4DocsParser(_lang), lang=_lang, license='ODC-By-1.0',
        desc=f'The HC4 {_lang} Common Crawl document corpus.')

    for _split in ('train', 'dev', 'test'):
        _name = f'hc4-{_lang}-{_split}'
        _queries = QueryTable(f'{_name}-queries',
            source=TOPICS_FILES[_split], parser=_Hc4QueriesParser(_lang))
        _qrels = TrecQrels(f'{_name}-qrels',
            source=QRELS_FILES[(_lang, _split)], defs=QREL_DEFS)
        _benchmarks.append(Benchmark(_name,
            docs=_docs, queries=_queries, qrels=_qrels,
            citation=CITATION,
            desc=f'HC4 {_lang}, {_split} split.'))


# Registration
# -----------------------------------------
irds.register(*_benchmarks, *TOPICS_FILES.values(),
              *[r for shards in IDS_FILES.values() for r in shards])

