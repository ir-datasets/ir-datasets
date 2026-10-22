"""MAST (Multilingual Agentic Search Track, FIRE 2026) -- a v2 dataset family.

A cross-lingual extension of BrowseComp-Plus (see ``browsecomp_plus.py``): 50
of its 830 queries, translated into 21 languages (15 in the *multilingual*
track, 9 in the *Indic* track; Hindi, Bengali and Tamil are in both). Queries
are asked in one language; the corpus (the same ~100K-document BrowseComp-Plus
web corpus) and the answers are English.

Everything but the queries is BrowseComp-Plus's own. The corpus is the
``browsecomp-plus-docs`` table (the MAST release is byte-identical to it), and
each language's qrels/answers are BrowseComp-Plus's evidence qrels and
ground-truth answers, restricted to the 50 translated queries and with
``query_id`` rewritten to MAST's ``<lang>-<browsecomp-plus id>`` form (e.g.
``zh-798``). MAST publishes no qrels or answers of its own; its leaderboard
scores against exactly these.

Like BrowseComp-Plus, the released queries are obfuscated (same canary / XOR
scheme) and are decrypted at read time, never written to disk.

Names carry the track year (``mast-2026-*``): the organizers plan a 2027 edition
with a multilingual corpus and human-written queries, which will be new nodes
beside these. The suites keep the plain ``mast-multilingual`` / ``mast-indic`` ids
(not ``-ml``, which would collide with the Malayalam benchmark ``mast-2026-ml``).

The Hindi, Bengali and Tamil query files are shared between the two tracks
(same ids and text), so each is one benchmark that belongs to both suites.
"""
import json

from ir_datasets.formats import BaseQueries, GenericQuery
from ir_datasets.v2 import AnswerTable, Benchmark, Overview, Parser, QueryTable, Resource, Suite, TrecQrels, irds, transform
from ir_datasets.v2.formats import GenericAnswer

from . import browsecomp_plus as bcp

METRICS = ['Recall', 'EM']

#: The 50 BrowseComp-Plus queries that were translated; identical for every language.
QUERY_IDS = frozenset((
    '10 50 52 55 58 149 156 159 166 174 186 235 255 262 320 354 362 366 371 391 408 464 469 549 553 560 '
    '582 618 679 684 720 798 865 894 942 943 963 968 1025 1037 1073 1105 1110 1135 1174 1182 1184 1208 '
    '1212 1264').split())

ML_REV = 'fb6614cbd965abeb7735852245fb91fd98eae3c6'
IND_REV = '8136627a451329a860ecde6a67029136c4233f4f'

#: lang -> (language name, sha256, size) of its queries.jsonl. The multilingual track's
#: languages first, then the Indic-only ones (hi/bn/ta are shared with the former).
QUERY_FILES = {
    'ar': ('Arabic', 'd190790d2ae6c6f842a76cea5db090f3aa0384f92f0d6fb1295a8b2b9b1fb3cf', 55537),
    'bn': ('Bengali', '53cee12661cf863af5cb80ed2ac289076e58756e3a07e21b4bafed4e0d424021', 105589),
    'cy': ('Welsh', '736fbb82f172125488873b48d43f99311c6f25c6a142591f522933499cebf06a', 39493),
    'de': ('German', '4c4be295e8292f77a3e747fa782219e7d67e0f8e719a2444f35aef1f62e749da', 43637),
    'en': ('English', '02a6ffecb18abe145dcd2e33a5dbd33bd366ce3afe60ad5a3265a4c2a180b903', 40365),
    'es': ('Spanish', 'a53b5b1275f3d4d8c71d744283143705bc7f76cffb852b73a6d9ea3658b51d2c', 42893),
    'fi': ('Finnish', 'cb0f1bebe4f3e4afd6409728ee668f074d466458d13d48b56eba1cc881b00ad9', 41241),
    'fr': ('French', '4cdca3153fe685364718c6b5a6e39b2215340e182db50aaa1237f81416e84cd1', 42177),
    'hi': ('Hindi', '03b0ce3c324442a7741382657974b626cac87dd8689883a1bf73f8c10e9f2dbd', 95297),
    'ru': ('Russian', 'cafb973832df80c5ba1f023edabd1e5988d0dc9fd8efe17b5ea82ae4473f93e0', 70237),
    'sw': ('Swahili', '8a0a471f3abd3c84ac5c2209d13d77b9ec41b6e837a0e1dce002aa0a465e6031', 42013),
    'ta': ('Tamil', '8c9c115b3c3e7dc3e0c2dd5074bcd98995ab1600fe4ad152401ee37bf08fad6e', 126541),
    'th': ('Thai', '93e361a5d2b30c71f30be53cda6e154722d4d5475127911d913a4954afe82827', 95461),
    'ur': ('Urdu', 'a9b94e672ae24dafb536a25fd9cd0a47cd4a6f2f6d6d3908db83cf7e65b5cebd', 64185),
    'zh': ('Chinese', 'a718cebda842124da90f3926f6c9e496067e9178a50e3fba3b0043e1241dfefc', 34845),
    'gu': ('Gujarati', 'c3bbb712051fc641e5a3a0ed57384a68daf47866738b3a178fbcd123f9c0c7a7', 96149),
    'kn': ('Kannada', '6f656b6142fa7eb2772c615ae810cbea87757bb8b4a380d7e42544901aeb8522', 103549),
    'ml': ('Malayalam', '3a213cb1901c077adae6795ddd4926366e582e9b97be766944fa13b75ed8b3fd', 109885),
    'or': ('Odia', '60057d041f0e460f731830d6490db3c1f7b8c296dd0cb867bc9d63494b4c766d', 102997),
    'pa': ('Punjabi', '7bef9e3dc409f27b7f438e491e2bdbc06587a827e5448a522406af0c7153dcc9', 97333),
    'te': ('Telugu', 'fd1894ca85e1f10a288154ea697e6fcea4ad250638902b627d3a17e3994f6db3', 103433),
}
MULTILINGUAL = ['en', 'zh', 'fr', 'ru', 'es', 'de', 'ar', 'bn', 'fi', 'hi', 'th', 'ur', 'ta', 'sw', 'cy']
INDIC = ['hi', 'bn', 'ta', 'te', 'kn', 'gu', 'pa', 'ml', 'or']
INDIC_ONLY = {'gu', 'kn', 'ml', 'or', 'pa', 'te'}

_URL = 'https://huggingface.co/datasets/mast-benchmark/{repo}/resolve/{rev}/data/{lang}/queries.jsonl'


def _jsonl_iter(path, columns):
    with open(path, 'rb') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                yield [row[c] for c in columns]


class _Queries(BaseQueries):
    def __init__(self, file, lang):
        self._file = file
        self._lang = lang

    def queries_iter(self):
        for qid, query in _jsonl_iter(self._file.path(), ['qid', 'query']):
            yield GenericQuery(qid, bcp._decrypt(query))

    def queries_cls(self):
        return GenericQuery

    def queries_namespace(self):
        return 'mast'

    def queries_lang(self):
        return self._lang


class _QueriesParser(Parser):
    name = 'MastQueries'

    def __init__(self, lang):
        self.lang = lang

    def build(self, source, node):
        return _Queries(source, self.lang)


class _Answers:
    """BrowseComp-Plus's answers, kept for the translated queries and keyed ``<lang>-<id>``."""

    def __init__(self, shards, lang):
        self._shards = shards
        self._lang = lang

    def answers_iter(self):
        for shard in self._shards:
            for q in bcp._parquet_iter(shard.path(), ['query_id', 'answer']):
                if str(q['query_id']) in QUERY_IDS:
                    yield GenericAnswer(f'{self._lang}-{q["query_id"]}', bcp._decrypt(q['answer']))

    def answers_cls(self):
        return GenericAnswer

    def answers_count(self):
        return len(QUERY_IDS)


class _AnswersParser(Parser):
    name = 'MastAnswers'

    def __init__(self, lang):
        self.lang = lang

    def build(self, source, node):
        return _Answers(source, self.lang)


def _qrels_transform(lang):
    @transform
    def prefix_qids(stream):
        for line in stream:
            qid, rest = line.split(None, 1)
            if qid.decode() in QUERY_IDS:
                yield f'{lang}-'.encode() + qid + b' ' + rest
    prefix_qids.name = f'prefix_qids_{lang}'
    return prefix_qids


# license verified 2026-10-21: https://huggingface.co/datasets/mast-benchmark/multilingual-queries-2026 and .../indic-queries-2026 (card metadata: license cc-by-sa-4.0); the corpus is BrowseComp-Plus's (MIT)
# Files / Tables / Benchmarks
# -----------------------------------------
queries = {}
benchmarks = {}
for lang, (language, sha256, size) in QUERY_FILES.items():
    repo, rev = ('indic-queries-2026', IND_REV) if lang in INDIC_ONLY else ('multilingual-queries-2026', ML_REV)
    file = Resource(f'mast-2026-queries-{lang}.jsonl',
        sources=[_URL.format(repo=repo, rev=rev, lang=lang)],
        hash=f'sha256:{sha256}', size=size, license='CC-BY-SA-4.0')
    queries[lang] = QueryTable(f'mast-2026-queries-{lang}', source=file, parser=_QueriesParser(lang),
        count_hint=50, lang=lang, license='CC-BY-SA-4.0')
    qrels = TrecQrels(f'mast-2026-qrels-{lang}', source=bcp.evidence_qrels_file.pipe(_qrels_transform(lang)),
        defs=bcp.EVIDENCE_DEFS, license='MIT')
    answers = AnswerTable(f'mast-2026-answers-{lang}', source=bcp.queries_files, parser=_AnswersParser(lang),
        count_hint=50, lang='en', license='MIT')
    benchmarks[lang] = Benchmark(f'mast-2026-{lang}',
        docs=bcp.docs, queries=queries[lang], qrels=qrels, answers=answers,
        metrics=METRICS, license='CC-BY-SA-4.0')

multilingual = Suite('mast-multilingual', benchmarks=[benchmarks[l] for l in MULTILINGUAL],
    license='CC-BY-SA-4.0')
indic = Suite('mast-indic', benchmarks=[benchmarks[l] for l in INDIC], license='CC-BY-SA-4.0')
overview = Overview('mast', parts=[multilingual, indic, *benchmarks.values()])

# Registration
# -----------------------------------------
irds.register(overview)
