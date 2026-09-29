"""mMARCO -- a v2 dataset family.

MS MARCO passage, machine-translated in two successive efforts: the
original ("Helsinki"-model) release, 8 languages, and a "v2" (Google
Translate) release covering 13 (5 overlapping with v1's set, 8 new). Only
``train`` qrels are the exact same file as msmarco-passage's own (same md5)
and so are shared by reference (``from .msmarco_passage import train``);
everything else here -- docs, queries, train docpairs, dev qrels, dev
"small" scoreddocs -- is mMARCO's own resource, mirroring v1's own choice of
what to share (it reuses only ``train``'s qrels handler, nothing else).

Two languages (pt, zh) additionally shipped a "v1.1" re-release of their
queries/scoreddocs (and, for pt, train queries too) after the original
files turned out to need re-generating -- both live side by side here, same
as v1, as their own named benchmarks rather than replacing the originals.

"dev-small" (and its v1.1 counterparts) is a filtered view of "dev" whose
query set is exactly the ids judged by the small qrels file. That filtering
is expressed with ``Filter.apply`` called directly (not through
``Benchmark(derived_from=..., filter=...)``) because -- unlike the
"/judged" pattern used elsewhere in this catalog -- its qrels and (where
present) scoreddocs are each their own distinct downloaded resource, not a
filtered view of the parent's.

The per-language, docs-only node (v1's own bare ``mmarco/{lang}`` /
``mmarco/v2/{lang}``) is just the docs table itself, not a Benchmark
wrapper: a corpus with no queries/qrels isn't an evaluable task, same
convention ``hc4.py`` uses for its own per-language, docs-only nodes.
"""
import ir_datasets
from ir_datasets.v2 import (
    Benchmark, Filter, Resource, TrecQrels, TrecScoredDocs, TsvDocPairs, TsvDocs, TsvQueries, irds,
)
from ir_datasets.v2.datasets.msmarco_passage import train as passage_train

CITATION = 'dblp:journals/corr/abs-2108-13897'
QRELS_DEFS = {1: 'Labeled by crowd worker as relevant'}

BASE = ir_datasets.util.home_path() / 'mmarco'

V1_LANGS = ['es', 'fr', 'pt', 'it', 'id', 'de', 'ru', 'zh']
V2_LANGS = ['ar', 'zh', 'dt', 'fr', 'de', 'hi', 'id', 'it', 'ja', 'pt', 'ru', 'es', 'vi']

# lang -> (url, md5, size)
V1_DOCS = {
    'es': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/collections/spanish_collection.tsv', 'c82d3e5998f4cefb5a730a337680fac0', 3382637867),
    'fr': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/collections/french_collection.tsv', '941a8e717efb1ab74b9017976c08f73a', 3469351128),
    'pt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/collections/portuguese_collection.tsv', '77260081f0332befa6aa3e6c922b8fb9', 3182982454),
    'it': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/collections/italian_collection.tsv', 'c87e107fdd99f78a8789c984c96f3e46', 3318354977),
    'id': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/collections/indonesian_collection.tsv', '5e9fd243bfcf160a1177c898796ae53e', 3118483579),
    'de': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/collections/german_collection.tsv', 'b0d40bb296c3ec903926243e8397560e', 3417960916),
    'ru': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/collections/russian_collection.tsv', '27f64ca50b1862d285c53a1bcdc26793', 5500129241),
    'zh': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/collections/chinese_collection.tsv', 'd5672e88206038d8a3d261987b3bd788', 2227628877),
}
V1_QUERIES_DEV = {
    'es': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/dev/spanish_queries.dev.tsv', '1359d9516bf5dd9bf732012a25e7f536', 5294892),
    'fr': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/dev/french_queries.dev.tsv', 'a86c46314a0abfb090c542e672bb66e2', 5523530),
    'pt': ('https://macavaney.us/files/mmarco/262ce189e3d57059b2795c16db44bb81', '262ce189e3d57059b2795c16db44bb81', 4962136),
    'it': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/dev/italian_queries.dev.tsv', 'd9df4b10d892288c81611e7ed74e1549', 5707295),
    'id': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/dev/indonesian_queries.dev.tsv', '3d5f2261edb985d4fe052ed9e379a42a', 4693330),
    'de': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/dev/german_queries.dev.tsv', '8d6383b34cd332f5f21c70c3e6a97579', 5039578),
    'ru': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/dev/russian_queries.dev.tsv', '4d803553d69f967764360570c3e66c84', 8539431),
    'zh': ('https://macavaney.us/files/mmarco/83820cac1d3e27a7c911d5116ebce558', '83820cac1d3e27a7c911d5116ebce558', 4404468),
}
V1_QUERIES_TRAIN = {
    'es': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/train/spanish_queries.train.tsv', '676e7011d020383422556d2c4b39b67d', 41890459),
    'fr': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/train/french_queries.train.tsv', '4c9f45b9c7497d6efb44e593d2f5af4d', 43700586),
    'pt': ('https://macavaney.us/files/mmarco/df2ed4ef0bdb93405ba276a92530fc03', 'df2ed4ef0bdb93405ba276a92530fc03', 39231205),
    'it': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/train/italian_queries.train.tsv', '2aa7e48decfbfbaa327aa01f0d16bc3f', 45148915),
    'id': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/train/indonesian_queries.train.tsv', 'e74bc49a64adf32d105eeace23dc1a58', 37170090),
    'de': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/train/german_queries.train.tsv', '711b9e9c2163fe07468d6303bbf038f8', 39894960),
    'ru': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/train/russian_queries.train.tsv', 'e4c09d563124569a632780c3ed1179b2', 67849583),
    'zh': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/train/chinese_queries.train.tsv', '21649d058877379bc8184923ab8ec408', 35231242),
}
# v1's own top1000 dev runs -- not published for pt/zh (see PT_ZH_V1_1 below).
V1_SCOREDDOCS_DEV = {
    'es': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/runs/run.bm25_spanish-trec.txt', 'fe545532f65b952d538ac6bda169c196', 272702362),
    'fr': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/runs/run.bm25_french-trec.txt', '44df2483ab31dd269f99e0f2e925df29', 272726176),
    'it': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/runs/run.bm25_italian-trec.txt', 'b1635aa57fb329b441cab2bda35c6883', 280099315),
    'id': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/runs/run.bm25_indonesian-trec.txt', '4c8b58801af5d691f489b3bb4765ec79', 275468922),
    'de': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/runs/run.bm25_german-trec.txt', '2e9fd71f8bb9770ef86971cacf7e9119', 264681166),
    'ru': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/runs/run.bm25_russian-trec.txt', '1f8daf67b9624146dbe5a392000f78f3', 279699867),
}
# pt/zh's "v1.1" re-release: newer queries/scoreddocs (and, for pt, train queries).
PT_ZH_V1_1_QUERIES_DEV = {
    'pt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/dev/portuguese_queries.dev.tsv', '4210db124ff1e3b7c803b9cb666c5e44', 4958298),
    'zh': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/dev/chinese_queries.dev.tsv', '9ea0d3e92aaf87d65c07c297893d0ff6', 4410653),
}
PT_ZH_V1_1_SCOREDDOCS_DEV = {
    'pt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/runs/run.bm25_portuguese-trec.txt', '8f8e6ecd4761bd2355f126ab289f57ca', 280708499),
    'zh': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/runs/run.bm25_chinese-trec.txt', '36655599b6d0d0ae95dd49baa2e15acc', 41505364),
}
PT_V1_1_QUERIES_TRAIN = ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/helsinki/queries/train/portuguese_queries.train.tsv', 'c253c476ff1ad1d51bae169cea180acd', 39210147)

V2_DOCS = {
    'ar': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/arabic_collection.tsv', 'b73406ce3a3d31edea240603d031be7a', 4664307196),
    'zh': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/chinese_collection.tsv', 'd176049b56f65bd49248003b9ea8b2b0', 2720255044),
    'dt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/dutch_collection.tsv', '4a29599f160d0a696c7c3d3010da1912', 3362632528),
    'fr': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/french_collection.tsv', '44fd86303b47d2c8c2f9f547cd67686f', 3656218984),
    'de': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/german_collection.tsv', 'dfedd22fef7e7e44966994e06cd7ec57', 3488187566),
    'hi': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/hindi_collection.tsv', '4551a71b468cc109b1f985f6b1c3afe0', 7649320531),
    'id': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/indonesian_collection.tsv', 'd1ba5ff9788f9b11b497e3f75749409c', 3297783371),
    'it': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/italian_collection.tsv', '9c3aa4c4342074e8d37d75c9ffe5f22e', 3444092853),
    'ja': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/japanese_collection.tsv', '85614bd4dac3aa221c0b657d5cc71695', 3924982422),
    'pt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/portuguese_collection.tsv', 'dde80fa2cc5782cae4c40d1127e51958', 3431011785),
    'ru': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/russian_collection.tsv', '9a0cef71748039a6112b0775592eb84d', 5769514997),
    'es': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/spanish_collection.tsv', 'a9eac6b39239121795171da8c86db932', 3571559558),
    'vi': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/vietnamese_collection.tsv', 'dd68020237857a508e7abe2250dad28b', 4140054533),
}
V2_QUERIES_DEV = {
    'ar': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/arabic_queries.dev.tsv', 'd93efe298008c35389afacb3d9fedb06', 6545729),
    'zh': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/chinese_queries.dev.tsv', '30c76201ecddb05d9b125a5f0ef5a6bb', 4002662),
    'dt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/dutch_queries.dev.tsv', '5bcc4ba604106fd6bb7e17030fcdc033', 4828536),
    'fr': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/french_queries.dev.tsv', 'c2a393af030f845041b85b44ca60680a', 5408487),
    'de': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/german_queries.dev.tsv', 'b420bdc83096caa07bd06667658d08c7', 4975836),
    'hi': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/hindi_queries.dev.tsv', '880ec0423ebc6345450d86c030d29f1b', 10389233),
    'id': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/indonesian_queries.dev.tsv', '2f9d37e7baaf7a3834af2d84ddafa376', 4660699),
    'it': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/italian_queries.dev.tsv', '9172fc14d18b1181d7f6c3120a66a8f9', 5044211),
    'ja': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/japanese_queries.dev.tsv', '23c86ba93c63891a95382d8e8198199f', 5823193),
    'pt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/portuguese_queries.dev.tsv', '73fb7009307c6aecd661184bce75cb5d', 4966328),
    'ru': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/russian_queries.dev.tsv', 'b1fe46eea62d4c5c3776c1bc7c38034e', 7853062),
    'es': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/spanish_queries.dev.tsv', 'b2f3c62c6d71700b9af610e7a29fef61', 5241697),
    'vi': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/vietnamese_queries.dev.tsv', '99df771fa60888632dca02431998cec5', 5775517),
}
V2_QUERIES_TRAIN = {
    'ar': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/arabic_queries.train.tsv', '08ab9eed247819db2f1f013f7e06f0d6', 51869220),
    'zh': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/chinese_queries.train.tsv', '166018bdb4d1a279c5083897d7f6752d', 31567548),
    'dt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/dutch_queries.train.tsv', '6c304cf6dbdaff18a876cc6012168d30', 38224557),
    'fr': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/french_queries.train.tsv', 'cbedd13e92ad9d70d049e0262e6956dd', 42857975),
    'de': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/german_queries.train.tsv', '41461df4fd61a98d76d846daa2b797c2', 39404290),
    'hi': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/hindi_queries.train.tsv', 'b41f3f58b7c11ade80966276ceed50a8', 82511017),
    'id': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/indonesian_queries.train.tsv', '5684c5bab64544f3ed62b050bdcf477d', 36857620),
    'it': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/italian_queries.train.tsv', 'd40f32b4a4fb30c938e4b09adc1d1d81', 39923771),
    'ja': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/japanese_queries.train.tsv', 'eeea5ea876f96eeeb66fb7f8d29055c9', 46027536),
    'pt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/portuguese_queries.train.tsv', '816a2b07fe7067d438f785ee9bd8ef88', 39281063),
    'ru': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/russian_queries.train.tsv', '9f9a76edc95fb91477683c610d287327', 62396828),
    'es': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/spanish_queries.train.tsv', '7f54a66db9c928b1245dd0709e6b1caf', 41464870),
    'vi': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/vietnamese_queries.train.tsv', 'af28d1dde86c5bdd27751e3a14a55252', 45651702),
}
V2_SCOREDDOCS_DEV = {
    'ar': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_arabic-msmarco.txt', 'fc130f6ba64e7d0029c7697525ab728c', 130628203),
    'zh': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_chinese-msmarco.txt', 'cad0177b9211526795630801ec219a36', 133131838),
    'dt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_dutch-msmarco.txt', '4263908d9dfcdcd9ddb681cb529ca794', 126028515),
    'fr': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_french-msmarco.txt', '12d5e2d412905dbfa0c188a7c01e2a6a', 130311130),
    'de': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_german-msmarco.txt', '2a7ac8fc322c2ee9f0869f3de055d832', 125641364),
    'hi': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_hindi-msmarco.txt', 'befe2edc072bba1fe02dae27dd91b586', 132794501),
    'id': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_indonesian-msmarco.txt', '50da2b41f7286152aef97279f4a604e6', 129538847),
    'it': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_italian-msmarco.txt', '966ed936d26b0bf54feb4ccd83a8757c', 132615737),
    'ja': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_japanese-msmarco.txt', 'a3908dd2cec7c1d66008eab3b455b4d6', 130027238),
    'pt': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_portuguese-msmarco.txt', '2cb2166103a874f6a7c364764ee5a9fe', 133067761),
    'ru': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_russian-msmarco.txt', 'b0f3de7ab4bb72bea524775816327df8', 132235194),
    'es': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_spanish-msmarco.txt', '35f59b4fd0f6c81099ed36d82ead36e8', 129268683),
    'vi': ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/run.bm25_vietnamese-msmarco.txt', '1b5691e13e7c3053f825fcf47f663ea5', 133077756),
}

TRAIN_TRIPLES = ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/triples.train.ids.small.tsv', 'cc7865df9f2345132dea1c0746a4699c', 905211990)
DEV_QRELS = ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/qrels.dev.tsv', '9157ccaeaa8227f91722ba5770787b16', 1201626)
DEV_QRELS_SMALL = ('https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/qrels.dev.small.tsv', '38a80559a561707ac2ec0f150ecd1e8a', 143300)


def _resource(name, spec):
    url, md5, size = spec
    return Resource(name, sources=[url], md5=md5, size=size)


with irds.defaults(dua=None):
    train_qrels = passage_train.qrels
    train_docpairs = TsvDocPairs('mmarco-train-docpairs', source=_resource('mmarco-train-docpairs.tsv', TRAIN_TRIPLES))
    dev_qrels = TrecQrels('mmarco-dev-qrels', source=_resource('mmarco-dev-qrels.tsv', DEV_QRELS), defs=QRELS_DEFS)
    dev_small_qrels = TrecQrels('mmarco-dev-small-qrels', source=_resource('mmarco-dev-small-qrels.tsv', DEV_QRELS_SMALL), defs=QRELS_DEFS)

    # Shared across every language: which query ids the small dev qrels judge.
    def _small_dev_qids():
        return {q.query_id for q in dev_small_qrels.qrels_iter()}
    _small_dev_qids.depends_on = (dev_small_qrels,)
    small_dev_filter = Filter(query_ids=_small_dev_qids, mode='include')

    nodes = {}

    for lang in V1_LANGS:
        docs = TsvDocs(f'mmarco-{lang}', source=_resource(f'mmarco-{lang}-docs.tsv', V1_DOCS[lang]), lang=lang)
        nodes[f'mmarco-{lang}'] = docs

        train_queries = TsvQueries(f'mmarco-{lang}-train-queries', source=_resource(f'mmarco-{lang}-train-queries.tsv', V1_QUERIES_TRAIN[lang]), lang=lang)
        nodes[f'mmarco-{lang}-train'] = Benchmark(f'mmarco-{lang}-train',
            docs=docs, queries=train_queries, qrels=train_qrels, docpairs=train_docpairs,
            citation=CITATION, desc=f'mMARCO: msmarco-passage-train, queries and corpus translated to {lang}.')

        dev_queries = TsvQueries(f'mmarco-{lang}-dev-queries', source=_resource(f'mmarco-{lang}-dev-queries.tsv', V1_QUERIES_DEV[lang]), lang=lang)
        nodes[f'mmarco-{lang}-dev'] = Benchmark(f'mmarco-{lang}-dev',
            docs=docs, queries=dev_queries, qrels=dev_qrels,
            citation=CITATION, desc=f'mMARCO: msmarco-passage-dev, queries and corpus translated to {lang}.')

        dev_small_queries = small_dev_filter.apply('queries', dev_queries, None, f'mmarco-{lang}-dev-small')
        dev_small_scoreddocs = TrecScoredDocs(f'mmarco-{lang}-dev-small-scoreddocs', source=_resource(f'mmarco-{lang}-dev-small-scoreddocs.txt', V1_SCOREDDOCS_DEV[lang])) if lang in V1_SCOREDDOCS_DEV else None
        nodes[f'mmarco-{lang}-dev-small'] = Benchmark(f'mmarco-{lang}-dev-small',
            docs=docs, queries=dev_small_queries, qrels=dev_small_qrels, scoreddocs=dev_small_scoreddocs,
            citation=CITATION, desc=f'mMARCO: msmarco-passage-dev-small, queries and corpus translated to {lang}.')

        if lang in PT_ZH_V1_1_QUERIES_DEV:
            v11_dev_queries = TsvQueries(f'mmarco-{lang}-dev-v1.1-queries', source=_resource(f'mmarco-{lang}-dev-v1.1-queries.tsv', PT_ZH_V1_1_QUERIES_DEV[lang]), lang=lang)
            nodes[f'mmarco-{lang}-dev-v1.1'] = Benchmark(f'mmarco-{lang}-dev-v1.1',
                docs=docs, queries=v11_dev_queries, qrels=dev_qrels,
                citation=CITATION, desc=f'mMARCO: {lang} "v1.1" re-release of the dev queries.')

            v11_dev_small_queries = small_dev_filter.apply('queries', v11_dev_queries, None, f'mmarco-{lang}-dev-small-v1.1')
            v11_dev_small_scoreddocs = TrecScoredDocs(f'mmarco-{lang}-dev-small-v1.1-scoreddocs', source=_resource(f'mmarco-{lang}-dev-small-v1.1-scoreddocs.txt', PT_ZH_V1_1_SCOREDDOCS_DEV[lang]))
            nodes[f'mmarco-{lang}-dev-small-v1.1'] = Benchmark(f'mmarco-{lang}-dev-small-v1.1',
                docs=docs, queries=v11_dev_small_queries, qrels=dev_small_qrels, scoreddocs=v11_dev_small_scoreddocs,
                citation=CITATION, desc=f'mMARCO: {lang} "v1.1" re-release, filtered to the small dev qrels.')

        if lang == 'pt':
            v11_train_queries = TsvQueries('mmarco-pt-train-v1.1-queries', source=_resource('mmarco-pt-train-v1.1-queries.tsv', PT_V1_1_QUERIES_TRAIN), lang=lang)
            nodes['mmarco-pt-train-v1.1'] = Benchmark('mmarco-pt-train-v1.1',
                docs=docs, queries=v11_train_queries, qrels=train_qrels, docpairs=train_docpairs,
                citation=CITATION, desc='mMARCO: pt "v1.1" re-release of the train queries.')

    for lang in V2_LANGS:
        docs = TsvDocs(f'mmarco-v2-{lang}', source=_resource(f'mmarco-v2-{lang}-docs.tsv', V2_DOCS[lang]), lang=lang)
        nodes[f'mmarco-v2-{lang}'] = docs

        train_queries = TsvQueries(f'mmarco-v2-{lang}-train-queries', source=_resource(f'mmarco-v2-{lang}-train-queries.tsv', V2_QUERIES_TRAIN[lang]), lang=lang)
        nodes[f'mmarco-v2-{lang}-train'] = Benchmark(f'mmarco-v2-{lang}-train',
            docs=docs, queries=train_queries, qrels=train_qrels, docpairs=train_docpairs,
            citation=CITATION, desc=f'mMARCO v2: msmarco-passage-train, queries and corpus translated to {lang}.')

        dev_queries = TsvQueries(f'mmarco-v2-{lang}-dev-queries', source=_resource(f'mmarco-v2-{lang}-dev-queries.tsv', V2_QUERIES_DEV[lang]), lang=lang)
        nodes[f'mmarco-v2-{lang}-dev'] = Benchmark(f'mmarco-v2-{lang}-dev',
            docs=docs, queries=dev_queries, qrels=dev_qrels,
            citation=CITATION, desc=f'mMARCO v2: msmarco-passage-dev, queries and corpus translated to {lang}.')

        dev_small_queries = small_dev_filter.apply('queries', dev_queries, None, f'mmarco-v2-{lang}-dev-small')
        dev_small_scoreddocs = TrecScoredDocs(f'mmarco-v2-{lang}-dev-small-scoreddocs', source=_resource(f'mmarco-v2-{lang}-dev-small-scoreddocs.txt', V2_SCOREDDOCS_DEV[lang]), negate_score=True)
        nodes[f'mmarco-v2-{lang}-dev-small'] = Benchmark(f'mmarco-v2-{lang}-dev-small',
            docs=docs, queries=dev_small_queries, qrels=dev_small_qrels, scoreddocs=dev_small_scoreddocs,
            citation=CITATION, desc=f'mMARCO v2: msmarco-passage-dev-small, queries and corpus translated to {lang}.')


# Registration
# -----------------------------------------
irds.register(*nodes.values())
