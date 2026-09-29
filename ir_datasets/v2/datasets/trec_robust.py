"""TREC Robust 2004 & 2005 -- a v2 dataset family.

TREC Robust 2004 judges ``disks45-nocr-docs`` (imported by reference from
``disks45.py``, same cross-file pattern as ``trec_adhoc.py``) against the
track's own 250 topics. The five cross-validation folds from Huston & Croft
(2014) are each a ``Filter(query_ids=..., mode='include')`` derived from the
base benchmark -- same shape as ``msmarco_passage.py``'s split200 train/valid
-- named ``trec-robust-2004-foldN`` rather than chaining ``disks45``/``nocr``
into the name (see the flat-naming rule: a fold's provenance is already the
``derived_from``/``filtered_by`` edges, not the name).

TREC Robust 2005 judges a different corpus (``aquaint``, imported by
reference from ``aquaint.py``) against a 50-"hard"-topic subset of 2004's own
250, using the same ``<num>``/``<title>``/``<desc>``/``<narr>`` topic format
-- close enough in shape to 2004 to share this module (same QREL_DEFS, same
family docstring) rather than live in a same-named-but-separate file.
"""
from ir_datasets.v2 import Benchmark, Filter, Resource, Source, TrecQrels, TrecQueries, irds
from ir_datasets.v2.datasets.aquaint import docs as aquaint_docs
from ir_datasets.v2.datasets.disks45 import DUA, docs as disks45_docs

CITATION_2004 = 'dblp:conf/trec/Voorhees04b'
CITATION_FOLDS = 'dblp:conf/cikm/HustonC14'
CITATION_2005 = 'dblp:conf/trec/Voorhees05a'

QREL_DEFS = {
    2: 'highly relevant',
    1: 'relevant',
    0: 'not relevant',
}

QTYPE_MAP_2005 = {
    '<num> *(Number:)?': 'query_id',
    '<title> *(Topic:)?': 'title',
    '<desc> *(Description:)?': 'description',
    '<narr> *(Narrative:)?': 'narrative',
}

# folds from Huston & Croft 2014 <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.646.7749>
FOLDS = {
    'fold1': {'302', '303', '309', '316', '317', '319', '323', '331', '336', '341', '356', '357', '370', '373', '378', '381', '383', '392', '394', '406', '410', '411', '414', '426', '428', '433', '447', '448', '601', '607', '608', '612', '617', '619', '635', '641', '642', '646', '647', '654', '656', '662', '665', '669', '670', '679', '684', '690', '692', '700'},
    'fold2': {'301', '308', '312', '322', '327', '328', '338', '343', '348', '349', '352', '360', '364', '365', '369', '371', '374', '386', '390', '397', '403', '419', '422', '423', '424', '432', '434', '440', '446', '602', '604', '611', '623', '624', '627', '632', '638', '643', '651', '652', '663', '674', '675', '678', '680', '683', '688', '689', '695', '698'},
    'fold3': {'306', '307', '313', '321', '324', '326', '334', '347', '351', '354', '358', '361', '362', '363', '376', '380', '382', '396', '404', '413', '415', '417', '427', '436', '437', '439', '444', '445', '449', '450', '603', '605', '606', '614', '620', '622', '626', '628', '631', '637', '644', '648', '661', '664', '666', '671', '677', '685', '687', '693'},
    'fold4': {'320', '325', '330', '332', '335', '337', '342', '344', '350', '355', '368', '377', '379', '387', '393', '398', '402', '405', '407', '408', '412', '420', '421', '425', '430', '431', '435', '438', '616', '618', '625', '630', '633', '636', '639', '649', '650', '653', '655', '657', '659', '667', '668', '672', '673', '676', '682', '686', '691', '697'},
    'fold5': {'304', '305', '310', '311', '314', '315', '318', '329', '333', '339', '340', '345', '346', '353', '359', '366', '367', '372', '375', '384', '385', '388', '389', '391', '395', '399', '400', '401', '409', '416', '418', '429', '441', '442', '443', '609', '610', '613', '615', '621', '629', '634', '640', '645', '658', '660', '681', '694', '696', '699'},
}

with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    queries_2004_file = Resource('trec-robust-2004-queries.gz',
        sources=['https://trec.nist.gov/data/robust/04.testset.gz', Source.irds()],
        md5='5eac3d774a2f87da61c08a94f945beff',
        size=34_293,
    )
    qrels_2004_file = Resource('trec-robust-2004-qrels.txt',
        sources=['https://trec.nist.gov/data/robust/qrels.robust2004.txt', Source.irds()],
        md5='123c2a0ba2ec31178cb1050995dcfdfa',
        size=6_543_541,
    )

    # Tables
    # -----------------------------------------
    queries_2004 = TrecQueries('trec-robust-2004-queries', source=queries_2004_file.gunzip(), count_hint=250,
        citation=CITATION_2004)
    qrels_2004 = TrecQrels('trec-robust-2004-qrels', source=qrels_2004_file, defs=QREL_DEFS, count_hint=311_410,
        citation=CITATION_2004)

    # Benchmarks
    # -----------------------------------------
    robust2004 = Benchmark('trec-robust-2004',
        docs=disks45_docs, queries=queries_2004, qrels=qrels_2004,
        citation=CITATION_2004,
        desc='TREC Robust 2004: focuses on improving the consistency of retrieval '
             'technology by targeting poorly-performing topics.')

    folds = {}
    for fold, qids in FOLDS.items():
        folds[fold] = Benchmark(f'trec-robust-2004-{fold}',
            derived_from=robust2004, filter=Filter(query_ids=qids, mode='include'),
            citation=CITATION_FOLDS,
            desc=f'TREC Robust 2004, {fold} of the 5-fold cross-validation split '
                 f'proposed by Huston & Croft (2014).')

# TREC Robust 2005 -- its own topics/qrels, no NIST DUA (unlike 2004's disks45
# corpus), so outside the irds.defaults(dua=...) block above.
# Files
# -----------------------------------------
queries_2005_file = Resource('trec-robust-2005-queries.txt',
    sources=['https://trec.nist.gov/data/robust/05/05.50.topics.txt', Source.irds()],
    md5='c2e722e6bdfd00f088c6f6517db564ce',
    size=25_116,
)
qrels_2005_file = Resource('trec-robust-2005-qrels.txt',
    sources=['https://trec.nist.gov/data/robust/05/TREC2005.qrels.txt', Source.irds()],
    md5='9186021c74090464c50f577d4826e2e2',
    size=944_950,
)

# Tables
# -----------------------------------------
queries_2005 = TrecQueries('trec-robust-2005-queries',
    source=queries_2005_file,
    qtype_map=QTYPE_MAP_2005, lang='en',
    count_hint=50,
    citation=CITATION_2005,
)
qrels_2005 = TrecQrels('trec-robust-2005-qrels',
    source=qrels_2005_file, defs=QREL_DEFS, count_hint=37_798,
    citation=CITATION_2005)

# Benchmarks
# -----------------------------------------
robust2005 = Benchmark('trec-robust-2005',
    docs=aquaint_docs, queries=queries_2005, qrels=qrels_2005,
    citation=CITATION_2005,
    desc='TREC Robust 2005: a 50 "hard" topic subset of TREC Robust 2004\'s '
         'query set, judged against the AQUAINT corpus.')


# Registration
# -----------------------------------------
irds.register(robust2004, *folds.values(), robust2005)
