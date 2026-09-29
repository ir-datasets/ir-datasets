"""LoTTE (Long-Tail Topic-stratified Evaluation) -- a v2 dataset family.

Six StackExchange-derived domains (five topics plus a "pooled" mix of all
five), each split into dev/test, each with two independent query sets --
"search" queries and "forum" post titles -- sharing one qrels shape: a jsonl
file mapping a query id to the pooled ids of every StackExchange answer post
that answered it. ``LotteQrels`` (reused from v1 as-is) turns each line's
``answer_pids`` array into one-hot ``TrecQrel`` rows.

The whole family ships as one ~3.6GB tar; every member this file reads
(``collection.tsv``, ``questions.{search,forum}.tsv``, ``qas.{search,forum}.jsonl``)
is extracted with ``.member()`` and cached individually -- same treatment
``msmarco_passage.py`` gives its own single collection member, and for the
same reason: ``TsvDocs``/``TsvQueries`` need a real, seekable file.

No corpus-only "lotte" root node: v1's own bare ``lotte`` id carries no docs
of its own (six independent corpora, one per domain/split), so there is
nothing for a single flat name to point at. Each domain/split *does* have
one -- v1's own ``lotte/{domain}/{split}`` -- and that is just the docs
table itself (``lotte-{domain}-{split}``), not a Benchmark wrapper around
it: a corpus with no queries/qrels isn't an evaluable task, same convention
``hc4.py`` uses for its own per-language, docs-only nodes.
"""
import ir_datasets
from ir_datasets.datasets.lotte import LotteQrels as _V1LotteQrels
from ir_datasets.v2 import Benchmark, QrelTable, Resource, TsvDocs, TsvQueries, irds
from ir_datasets.v2.formats import Parser

CITATION = 'dblp:conf/naacl/SanthanamKSPZ22'
QRELS_DEFS = {1: 'Answer upvoted or accepted on stack exchange'}

BASE = ir_datasets.util.home_path() / 'lotte'

DOMAINS = ['lifestyle', 'recreation', 'science', 'technology', 'writing', 'pooled']


class _LotteQrelsParser(Parser):
    name = 'LotteQrels'

    def build(self, source, node):
        return _V1LotteQrels(source)


with irds.defaults(lang='en'):
    source = Resource('lotte.tar.gz',
        sources=['https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz'],
        md5='3b2e88b1d66933627462950b4c3f5d0f',
        size=3_576_167_599,
    )

    nodes = {}

    for domain in DOMAINS:
        for split in ['dev', 'test']:
            docs = TsvDocs(f'lotte-{domain}-{split}',
                source=source.member(f'lotte/{domain}/{split}/collection.tsv')
                             .cache(BASE / domain / split / 'collection.tsv'))
            nodes[f'lotte-{domain}-{split}'] = docs

            for qtype in ['search', 'forum']:
                queries = TsvQueries(f'lotte-{domain}-{split}-{qtype}-queries',
                    source=source.member(f'lotte/{domain}/{split}/questions.{qtype}.tsv')
                                 .cache(BASE / domain / split / f'questions.{qtype}.tsv'))
                qrels = QrelTable(f'lotte-{domain}-{split}-{qtype}-qrels',
                    source=source.member(f'lotte/{domain}/{split}/qas.{qtype}.jsonl')
                                 .cache(BASE / domain / split / f'qas.{qtype}.jsonl'),
                    defs=QRELS_DEFS,
                    parser=_LotteQrelsParser())
                name = f'lotte-{domain}-{split}-{qtype}'
                nodes[name] = Benchmark(name, docs=docs, queries=queries, qrels=qrels,
                    citation=CITATION,
                    desc=f'LoTTE: {domain} ({split} split), {qtype} queries.')


# Registration
# -----------------------------------------
irds.register(*nodes.values())
