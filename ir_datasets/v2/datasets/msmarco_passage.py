"""MS MARCO (passage) — a v2 dataset family.

Replaces:

    ir_datasets/datasets/msmarco_passage.py    wiring + custom stream fixups
    ir_datasets/docs/msmarco-passage.yaml      desc / bibtex / official measures
    ir_datasets/etc/downloads.json             urls / md5 / size   (was central!)
    test/integration/msmarco-passage.py        counts / sample records

Counts, content hashes and sample records come from
``python -m ir_datasets.v2.freeze --verify``, not from this file.

Not every v1 subset is covered here (see "Known gaps" at the bottom of this
docstring / the package README): ``train/triples-small`` (needs a doc/query
text->id hash-join, not just a stream transform), ``trec-dl-hard`` and its
folds (queries drawn from two prior years plus an external fold list),
and ``dev/2`` (its ids come from ``msmarco-passage-v2``, not yet a v2 family).
"""
import ir_datasets
from ir_datasets.v2 import (
    Benchmark, Filter, Resource, Source, TrecQrels, TrecScoredDocs, TsvDocPairs,
    TsvDocs, TsvQueries, ids_from_lines, irds, transform,
)

#: Where extracted tar members are materialized (see the note on
#: ``.cache()`` below, next to ``docs``). Same layout v1 used, so an
#: existing v1 extraction is reused in place.
BASE = ir_datasets.util.home_path() / 'msmarco-passage'

DUA = ("Please confirm you agree to the MSMARCO data usage agreement found at "
       "<http://www.msmarco.org/dataset.aspx>")

CITATION = 'Bajaj2016Msmarco'
MEASURES = ['RR@10']
TREC_DL_MEASURES = ['nDCG@10', 'RR(rel=2)', 'AP(rel=2)']

QRELS_DEFS = {
    1: 'Labeled by crowd worker as relevant',
}

TREC_DL_QRELS_DEFS = {
    3: "Perfectly relevant: The passage is dedicated to the query and contains the exact answer.",
    2: "Highly relevant: The passage has some answer for the query, but the answer may be a bit "
       "unclear, or hidden amongst extraneous information.",
    1: "Related: The passage seems related to the query but does not answer it.",
    0: "Irrelevant: The passage has nothing to do with the query.",
}

# 200 training queries held out for validation.
SPLIT200_QIDS = {'484694', '836399', '683975', '428803', '1035062', '723895', '267447', '325379', '582244', '148817', '44209', '1180950', '424238', '683835', '701002', '1076878', '289809', '161771', '807419', '530982', '600298', '33974', '673484', '1039805', '610697', '465983', '171424', '1143723', '811440', '230149', '23861', '96621', '266814', '48946', '906755', '1142254', '813639', '302427', '1183962', '889417', '252956', '245327', '822507', '627304', '835624', '1147010', '818560', '1054229', '598875', '725206', '811871', '454136', '47069', '390042', '982640', '1174500', '816213', '1011280', '368335', '674542', '839790', '270629', '777692', '906062', '543764', '829102', '417947', '318166', '84031', '45682', '1160562', '626816', '181315', '451331', '337653', '156190', '365221', '117722', '908661', '611484', '144656', '728947', '350999', '812153', '149680', '648435', '274580', '867810', '101999', '890661', '17316', '763438', '685333', '210018', '600923', '1143316', '445800', '951737', '1155651', '304696', '958626', '1043094', '798480', '548097', '828870', '241538', '337392', '594253', '1047678', '237264', '538851', '126690', '979598', '707766', '1160366', '123055', '499590', '866943', '18892', '93927', '456604', '560884', '370753', '424562', '912736', '155244', '797512', '584995', '540814', '200926', '286184', '905213', '380420', '81305', '749773', '850038', '942745', '68689', '823104', '723061', '107110', '951412', '1157093', '218549', '929871', '728549', '30937', '910837', '622378', '1150980', '806991', '247142', '55840', '37575', '99395', '231236', '409162', '629357', '1158250', '686443', '1017755', '1024864', '1185054', '1170117', '267344', '971695', '503706', '981588', '709783', '147180', '309550', '315643', '836817', '14509', '56157', '490796', '743569', '695967', '1169364', '113187', '293255', '859268', '782494', '381815', '865665', '791137', '105299', '737381', '479590', '1162915', '655989', '292309', '948017', '1183237', '542489', '933450', '782052', '45084', '377501', '708154'}

_MS_HEADERS = {'X-Ms-Version': '2019-12-12'}


# The original document source files contain a double-encoding error that
# produces strange sequences like "å¬" and "ðºð" (should be "公" and "🇺🇸"). This
# reproduces v1's fixup byte-for-byte: find short runs around a "suspicious"
# (0x80-0xff) character, try re-decoding them as latin1->utf8, and keep the
# result only where it collapses to a single, valid character.
@transform
def fix_encoding(stream):
    import re
    SUS = '[\x80-\xff]'
    regexes = [
        re.compile(f'(...{SUS}|..{SUS}.|.{SUS}..|{SUS}...)'),
        re.compile(f'(..{SUS}|.{SUS}.|{SUS}..)'),
        re.compile(f'(.{SUS}|{SUS}.)'),
    ]
    for line in stream:
        line = line.decode('utf8')
        for regex in regexes:
            pos = 0
            while pos < len(line):
                match = regex.search(line, pos=pos)
                if not match:
                    break
                try:
                    fixed = match.group().encode('latin1').decode('utf8')
                    if len(fixed) == 1:
                        line = line[:match.start()] + fixed + line[match.end():]
                except UnicodeError:
                    pass
                pos = match.start() + 1
        yield line.encode()


# The "top1000"/test-year scoreddocs files are "qid did qtext dtext" (to save
# space, they carry the query/doc text instead of a run score). Reduce that
# to the "qid did" pairs TrecScoredDocs already understands as MS MARCO-style.
@transform
def extract_qid_pid(stream):
    for line in stream:
        qid, did, _, _ = line.split(b'\t')
        yield qid + b'\t' + did + b'\n'


with irds.defaults(dua=DUA, lang='en', namespace='msmarco'):
    # Files
    # -----------------------------------------
    collectionandqueries_file = Resource('msmarco-passage-collectionandqueries.tar.gz',
        sources=[Source('https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz', headers=_MS_HEADERS)],
        md5='31644046b18952c1386cd4564ba2ae69',
        size=1_057_717_952,
    )
    queries_file = Resource('msmarco-passage-queries.tar.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz'],
        md5='c177b2795d5f2dcc524cf00fcd973be1',
        size=18_882_551,
    )
    medmarco_ids_file = Resource('msmarco-passage-medmarco-ids.txt',
        sources=['https://raw.githubusercontent.com/Georgetown-IR-Lab/covid-neural-ir/master/med-msmarco-train.txt'],
        md5='dc5199de7d4a872c361f89f08b1163ef',
        size=548_428,
    )
    train_qrels_file = Resource('msmarco-passage-train-qrels.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv'],
        md5='733fb9fe12d93e497f7289409316eccf',
        size=10_589_532,
    )
    train_docpairs_file = Resource('msmarco-passage-train-docpairs.gz',
        sources=[Source('https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.tsv.gz', headers=_MS_HEADERS)],
        md5='215a5204288820672f5e9451d9e202c5',
        size=2_633_557_579,
    )
    train_docpairs_v2_file = Resource('msmarco-passage-train-docpairs-v2.gz',
        sources=[Source('https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz', headers=_MS_HEADERS)],
        md5='219083e80a0a751c08b968c2f31a4e0b',
        size=1_841_693_309,
    )
    train_scoreddocs_file = Resource('msmarco-passage-train-scoreddocs.tar.gz',
        sources=[Source('https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.train.tar.gz', headers=_MS_HEADERS)],
        md5='d99fdbd5b2ea84af8aa23194a3263052',
        size=11_519_984_492,
    )
    dev_qrels_file = Resource('msmarco-passage-dev-qrels.tsv',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv'],
        md5='9157ccaeaa8227f91722ba5770787b16',
        size=1_201_626,
    )
    dev_small_scoreddocs_file = Resource('msmarco-passage-dev-small-scoreddocs.tar.gz',
        sources=[Source('https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz', headers=_MS_HEADERS)],
        md5='8c140662bdf123a98fbfe3bb174c5831',
        size=687_414_398,
    )
    eval_small_scoreddocs_file = Resource('msmarco-passage-eval-small-scoreddocs.tar.gz',
        sources=[Source('https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.eval.tar.gz', headers=_MS_HEADERS)],
        md5='73778cd99f6e0632d12d0b5731b20a02',
        size=673_440_221,
    )
    # Named to match their tables (see the queries/qrels/scoreddocs Tables
    # below): the queries file is shared (no "-passage"; the same bytes a
    # future msmarco-document family would reuse by reference), qrels/
    # scoreddocs are corpus-specific and carry "-passage".
    trec_dl_2019_qrels_file = Resource('trec-dl-2019-passage-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2019qrels-pass.txt', Source.irds()],
        md5='2f4be390198da108f6845c822e5ada14',
        size=187_092,
    )
    trec_dl_2019_queries_file = Resource('trec-dl-2019-queries.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz'],
        md5='eda71eccbe4d251af83150abe065368c',
        size=4_276,
    )
    trec_dl_2019_scoreddocs_file = Resource('trec-dl-2019-passage-scoreddocs.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz'],
        md5='ec9e012746aa9763c7ff10b3336a3ce1',
        size=26_634_062,
    )
    trec_dl_2020_qrels_file = Resource('trec-dl-2020-passage-qrels.txt',
        sources=['https://trec.nist.gov/data/deep/2020qrels-pass.txt', Source.irds()],
        md5='0355ccee7509ac0463e8278186cdd8d1',
        size=218_617,
    )
    trec_dl_2020_queries_file = Resource('trec-dl-2020-queries.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz'],
        md5='00a406fb0d14ed3752d70d1e4eb98600',
        size=4_131,
    )
    trec_dl_2020_scoreddocs_file = Resource('trec-dl-2020-passage-scoreddocs.gz',
        sources=['https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2020-top1000.tsv.gz'],
        md5='aa6fbc51d66bd1dc745964c0e140a727',
        size=26_230_218,
    )

    # Tables
    # -----------------------------------------
    # NOTE: TsvDocs/TsvQueries read their source through io.TextIOWrapper,
    # which calls .seekable() on it -- and tarfile's streaming-mode member
    # objects (as produced directly by .member()) don't support that. So,
    # same as v1, every tar member feeding a Tsv* format is materialized to a
    # real file with .cache() first; TrecQrels/TrecScoredDocs read through
    # codecs.getreader instead and don't need it.
    docs = TsvDocs('msmarco-passage-docs',
        source=collectionandqueries_file.member('collection.tsv').pipe(fix_encoding).cache(BASE / 'collection.tsv'),
        docstore_size_hint=14_373_971_970,
        count_hint=8_841_823,
    )

    train_queries = TsvQueries('msmarco-passage-train-queries',
        source=queries_file.member('queries.train.tsv').cache(BASE / 'train' / 'queries.tsv'))
    train_qrels = TrecQrels('msmarco-passage-train-qrels',
        source=train_qrels_file, defs=QRELS_DEFS)
    train_docpairs = TsvDocPairs('msmarco-passage-train-docpairs',
        source=train_docpairs_file.gunzip())
    train_docpairs_v2 = TsvDocPairs('msmarco-passage-train-docpairs-v2',
        source=train_docpairs_v2_file.gunzip())
    train_scoreddocs = TrecScoredDocs('msmarco-passage-train-scoreddocs',
        source=train_scoreddocs_file.member('top1000.train.txt').pipe(extract_qid_pid))

    dev_queries = TsvQueries('msmarco-passage-dev-queries',
        source=queries_file.member('queries.dev.tsv').cache(BASE / 'dev' / 'queries.tsv'))
    dev_qrels = TrecQrels('msmarco-passage-dev-qrels',
        source=dev_qrels_file, defs=QRELS_DEFS)

    dev_small_queries = TsvQueries('msmarco-passage-dev-small-queries',
        source=collectionandqueries_file.member('queries.dev.small.tsv').cache(BASE / 'dev' / 'small' / 'queries.tsv'))
    dev_small_qrels = TrecQrels('msmarco-passage-dev-small-qrels',
        source=collectionandqueries_file.member('qrels.dev.small.tsv'), defs=QRELS_DEFS)
    dev_small_scoreddocs = TrecScoredDocs('msmarco-passage-dev-small-scoreddocs',
        source=dev_small_scoreddocs_file.member('top1000.dev').pipe(extract_qid_pid))

    eval_queries = TsvQueries('msmarco-passage-eval-queries',
        source=queries_file.member('queries.eval.tsv').cache(BASE / 'eval' / 'queries.tsv'))
    eval_small_queries = TsvQueries('msmarco-passage-eval-small-queries',
        source=collectionandqueries_file.member('queries.eval.small.tsv').cache(BASE / 'eval' / 'small' / 'queries.tsv'))
    eval_small_scoreddocs = TrecScoredDocs('msmarco-passage-eval-small-scoreddocs',
        source=eval_small_scoreddocs_file.member('top1000.eval').pipe(extract_qid_pid))

    # TREC-DL is its own benchmark identity, not msmarco-passage's -- and its
    # queries are shared with msmarco-document's version of the same track
    # (same query set, different docs/qrels/scoreddocs per corpus). So the
    # queries are named without a corpus prefix (shared by reference, once a
    # msmarco-document family exists); qrels/scoreddocs/the benchmark itself
    # are corpus-specific and carry "-passage".
    trec_dl_2019_queries = TsvQueries('trec-dl-2019-queries',
        source=trec_dl_2019_queries_file.gunzip())
    trec_dl_2019_qrels = TrecQrels('trec-dl-2019-passage-qrels',
        source=trec_dl_2019_qrels_file, defs=TREC_DL_QRELS_DEFS)
    trec_dl_2019_scoreddocs = TrecScoredDocs('trec-dl-2019-passage-scoreddocs',
        source=trec_dl_2019_scoreddocs_file.gunzip().pipe(extract_qid_pid))

    trec_dl_2020_queries = TsvQueries('trec-dl-2020-queries',
        source=trec_dl_2020_queries_file.gunzip())
    trec_dl_2020_qrels = TrecQrels('trec-dl-2020-passage-qrels',
        source=trec_dl_2020_qrels_file, defs=TREC_DL_QRELS_DEFS)
    trec_dl_2020_scoreddocs = TrecScoredDocs('trec-dl-2020-passage-scoreddocs',
        source=trec_dl_2020_scoreddocs_file.gunzip().pipe(extract_qid_pid))

    # Benchmarks
    # -----------------------------------------
    train = Benchmark('msmarco-passage-train',
        docs=docs, queries=train_queries, qrels=train_qrels,
        docpairs=train_docpairs, scoreddocs=train_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='Official train set.')
    train_judged = Benchmark('msmarco-passage-train-judged',
        derived_from=train, filter=Filter(queries_with_qrels=True),
        citation=CITATION, metrics=MEASURES,
        desc='msmarco-passage-train restricted to queries with >= 1 qrel.')
    train_triples_v2 = Benchmark('msmarco-passage-train-triples-v2',
        docs=docs, queries=train_queries, qrels=train_qrels,
        docpairs=train_docpairs_v2, scoreddocs=train_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='msmarco-passage-train with the v2 (larger) docpairs release.')
    train_split200_train = Benchmark('msmarco-passage-train-split200-train',
        derived_from=train, filter=Filter(query_ids=SPLIT200_QIDS, mode='exclude'),
        citation=CITATION, metrics=MEASURES,
        desc='msmarco-passage-train without the 200 queries held out in '
             'msmarco-passage-train-split200-valid.')
    train_split200_valid = Benchmark('msmarco-passage-train-split200-valid',
        derived_from=train, filter=Filter(query_ids=SPLIT200_QIDS, mode='include'),
        citation=CITATION, metrics=MEASURES,
        desc='A held-out subset of 200 queries from msmarco-passage-train. '
             'Use with msmarco-passage-train-split200-train.')
    train_medical = Benchmark('msmarco-passage-train-medical',
        derived_from=train,
        filter=Filter(query_ids=ids_from_lines(medmarco_ids_file), mode='include'),
        citation=CITATION, metrics=MEASURES,
        desc='Subset of msmarco-passage-train restricted to medical queries, '
             'as identified by Georgetown-IR-Lab/covid-neural-ir.')

    dev = Benchmark('msmarco-passage-dev',
        docs=docs, queries=dev_queries, qrels=dev_qrels,
        citation=CITATION, metrics=MEASURES,
        desc='Official dev set.')
    dev_judged = Benchmark('msmarco-passage-dev-judged',
        derived_from=dev, filter=Filter(queries_with_qrels=True),
        citation=CITATION, metrics=MEASURES,
        desc='msmarco-passage-dev restricted to queries with >= 1 qrel.')
    dev_small = Benchmark('msmarco-passage-dev-small',
        docs=docs, queries=dev_small_queries, qrels=dev_small_qrels,
        scoreddocs=dev_small_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='Official "small" dev set (6,980 queries).')

    eval_ = Benchmark('msmarco-passage-eval',
        docs=docs, queries=eval_queries,
        citation=CITATION, metrics=MEASURES,
        desc='Official eval set for the MS MARCO leaderboard (qrels hidden).')
    eval_small = Benchmark('msmarco-passage-eval-small',
        docs=docs, queries=eval_small_queries, scoreddocs=eval_small_scoreddocs,
        citation=CITATION, metrics=MEASURES,
        desc='Official "small" eval set (6,837 queries).')

    # Named without the msmarco-passage prefix: TREC-DL is its own benchmark
    # identity (not hierarchical under this family), same as its queries above.
    trec_dl_2019 = Benchmark('trec-dl-2019-passage',
        docs=docs, queries=trec_dl_2019_queries, qrels=trec_dl_2019_qrels,
        scoreddocs=trec_dl_2019_scoreddocs,
        citation=CITATION, metrics=TREC_DL_MEASURES,
        desc='TREC Deep Learning 2019 passage ranking task.')
    trec_dl_2019_judged = Benchmark('trec-dl-2019-passage-judged',
        derived_from=trec_dl_2019, filter=Filter(queries_with_qrels=True),
        citation=CITATION, metrics=TREC_DL_MEASURES,
        desc='trec-dl-2019-passage restricted to queries with >= 1 qrel.')

    trec_dl_2020 = Benchmark('trec-dl-2020-passage',
        docs=docs, queries=trec_dl_2020_queries, qrels=trec_dl_2020_qrels,
        scoreddocs=trec_dl_2020_scoreddocs,
        citation=CITATION, metrics=TREC_DL_MEASURES,
        desc='TREC Deep Learning 2020 passage ranking task.')
    trec_dl_2020_judged = Benchmark('trec-dl-2020-passage-judged',
        derived_from=trec_dl_2020, filter=Filter(queries_with_qrels=True),
        citation=CITATION, metrics=TREC_DL_MEASURES,
        desc='trec-dl-2020-passage restricted to queries with >= 1 qrel.')


# Registration
# -----------------------------------------
irds.register(
    train, train_judged, train_triples_v2, train_split200_train,
    train_split200_valid, train_medical,
    dev, dev_judged, dev_small,
    eval_, eval_small,
    trec_dl_2019, trec_dl_2019_judged,
    trec_dl_2020, trec_dl_2020_judged,
)


# Aliases (old ir-datasets ID mapping)
# -----------------------------------------
irds.alias({
    'msmarco-passage': 'msmarco-passage-docs',
    'msmarco-passage/train': 'msmarco-passage-train',
    'msmarco-passage/train/judged': 'msmarco-passage-train-judged',
    'msmarco-passage/train/triples-v2': 'msmarco-passage-train-triples-v2',
    'msmarco-passage/train/split200-train': 'msmarco-passage-train-split200-train',
    'msmarco-passage/train/split200-valid': 'msmarco-passage-train-split200-valid',
    'msmarco-passage/train/medical': 'msmarco-passage-train-medical',
    'msmarco-passage/dev': 'msmarco-passage-dev',
    'msmarco-passage/dev/judged': 'msmarco-passage-dev-judged',
    'msmarco-passage/dev/small': 'msmarco-passage-dev-small',
    'msmarco-passage/eval': 'msmarco-passage-eval',
    'msmarco-passage/eval/small': 'msmarco-passage-eval-small',
    'msmarco-passage/trec-dl-2019': 'trec-dl-2019-passage',
    'msmarco-passage/trec-dl-2019/judged': 'trec-dl-2019-passage-judged',
    'msmarco-passage/trec-dl-2020': 'trec-dl-2020-passage',
    'msmarco-passage/trec-dl-2020/judged': 'trec-dl-2020-passage-judged',
})
