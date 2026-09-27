"""ANTIQUE — a complete v2 dataset family in one file.

This replaces all four v1 artifacts:

    ir_datasets/datasets/antique.py        wiring
    ir_datasets/docs/antique.yaml          desc / bibtex
    ir_datasets/etc/downloads.json         urls / md5 / size   (was central!)
    test/integration/antique.py            counts / sample records

Counts, content hashes and sample records are NOT written here — they come from
`python -m ir_datasets.v2.freeze --verify`, which runs this module once and
commits a manifest.

Nodes defined (17), all under the ``irds:`` prefix:

    antique-docs.txt, antique-*.txt/.qrel   the raw downloads (named by
                                            extension, not a generic "-file"
                                            suffix -- see sources.py)
    antique-docs                        docs table
    antique-train-queries/-qrels        tables
    antique-test-queries/-qrels         tables
    antique-train, antique-test         benchmarks
    antique-split200-train/-valid       derived benchmarks
    antique-test-non-offensive          derived benchmark
"""
from ir_datasets.v2 import Benchmark, Resource, Filter, TrecQrels, TsvDocs, TsvQueries, irds, ids_from_lines

DUA = ("Please confirm you agree to the authors' data usage agreement found at "
       "<https://ciir.cs.umass.edu/downloads/Antique/readme.txt>")

# Official measures and the paper to cite are plain metadata on each Benchmark
# below -- not edges to paper/measure nodes. That richer model (papers and
# measures as nodes, citations and evaluations as edges) is future work for a
# broader knowledge graph; for now these are just strings.
MEASURES = ['P(rel=3)@10', 'nDCG@10', 'MAP(rel=3)', 'RR(rel=3)']
CITATION = 'dblp:conf/ecir/HashemiAZC20'

# Relevance-level semantics belong to the qrels node, not to the benchmark and
# not to the parser. Verbatim from <https://arxiv.org/pdf/1905.08957.pdf>.
QREL_DEFS = {
    4: "It looks reasonable and convincing. Its quality is on par with or better "
       "than the \"Possibly Correct Answer\". Note that it does not have to "
       "provide the same answer as the \"PossiblyCorrect Answer\".",
    3: "It can be an answer to the question, however, it is notsufficiently "
       "convincing. There should be an answer with much better quality for the "
       "question.",
    2: "It does not answer the question or if it does, it provides "
       "anunreasonable answer, however, it is not out of context. Therefore, "
       "you cannot accept it as an answer to the question.",
    1: "It is completely out of context or does not make any sense.",
}

# 200 training queries held out for validation. An explicit id set like this is
# data, so it stays inline (it defines the node; it is not derived from a file).
VALIDATION_QIDS = {'1158088', '4032777', '1583099', '263783', '4237144', '1097878', '114758', '1211877', '1188438', '2689609', '1191621', '2571912', '1471877', '2961191', '2630860', '4092472', '3178012', '358253', '3913653', '844617', '2764765', '212427', '220575', '11706', '4069320', '3280274', '3159749', '4217473', '4042061', '1037897', '103298', '332662', '752633', '2704', '3635284', '2235825', '3651236', '2155390', '3752394', '2008456', '98438', '511835', '1647624', '3884772', '1536937', '544869', '66151', '2678635', '963523', '1881436', '993601', '3608433', '2048278', '3124162', '1907320', '1970273', '2891885', '2858043', '189364', '397709', '3470651', '3885753', '1933929', '94629', '2500918', '1708787', '2492366', '17665', '278043', '643630', '1727343', '196651', '3731489', '2910592', '1144768', '2573745', '546552', '1341602', '317469', '2735795', '1251077', '3507499', '3374970', '1034050', '1246269', '2901754', '2137263', '1295284', '2180502', '406082', '1443637', '2620488', '3118286', '3814583', '3738877', '684633', '2094435', '242701', '2613648', '2942624', '1495234', '1440810', '2421078', '961127', '595342', '363519', '4048305', '485408', '2573803', '3104841', '3626847', '727663', '3961', '4287367', '2112535', '913424', '1514356', '1512776', '937635', '1321784', '1582044', '1467322', '461995', '884643', '4338583', '2550445', '4165672', '1016750', '1184520', '3152714', '3617468', '3172166', '4031702', '2534994', '2035638', '404359', '1398838', '4183127', '2418824', '2439070', '2632334', '4262151', '3841762', '4400543', '2147417', '514804', '1423289', '2041828', '2776069', '1458676', '3407617', '1450678', '1978816', '2466898', '1607303', '2175167', '772988', '1289770', '3382182', '3690922', '1051346', '344029', '2357505', '1907847', '2587810', '3272207', '2522067', '1107012', '554539', '489705', '3652886', '4287894', '4387641', '1727879', '348777', '566364', '2678484', '4450252', '986260', '4336509', '3824106', '2169746', '2700836', '3495304', '3083719', '126182', '1607924', '1485589', '3211282', '2546730', '2897078', '3556937', '2113006', '929821', '2306533', '2543919', '1639607', '3958214', '2677193', '763189'}


# Everything in this family shares these; an explicit argument still wins.
with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    docs_file = Resource('antique-docs.txt',
        sources=['https://ciir.cs.umass.edu/downloads/Antique/antique-collection.txt'],
        # Multiple hashes on one Resource, demonstrating the general
        # 'algo:hexdigest' mechanism (see nodes.parse_hash) rather than the
        # md5= shorthand alone -- both computed from the same real file.
        hashes=['md5:684f7015aff377062a758e478476aac8',
               'sha256:68b6688f5f2668c93f0e8e43384f66def768c4da46da4e9f7e2629c1c47a0c36'],
        size=93_608_031
    )
    train_queries_file = Resource('antique-train-queries.txt',
        sources=['https://ciir.cs.umass.edu/downloads/Antique/antique-train-queries.txt'],
        md5='7684bd977d2682177b559d8da714f45a',
        size=136_512
    )
    train_qrels_file = Resource('antique-train-qrels.qrel',
        sources=['https://ciir.cs.umass.edu/downloads/Antique/antique-train.qrel'],
        md5='bac76531a3313a2d1debf5f1602d88ab',
        size=625_622
    )
    test_queries_file = Resource('antique-test-queries.txt',
        sources=['https://ciir.cs.umass.edu/downloads/Antique/antique-test-queries.txt'],
        md5='d09c5d9ad14368c23c853f6be81e7f2e',
        size=11_434
    )
    test_qrels_file = Resource('antique-test-qrels.qrel',
        sources=['https://ciir.cs.umass.edu/downloads/Antique/antique-test.qrel'],
        md5='c93ab0f0ce7937c84270c1eef172db4e',
        size=149_838
    )
    disallow_list_file = Resource('antique-test-disallow-list.txt',
        sources=['https://ciir.cs.umass.edu/downloads/Antique/test-queries-blacklist.txt'],
        md5='4ca64485dabf26221b90cf96ae2997f9',
        size=184
    )

    # Datasets
    # -----------------------------------------
    docs = TsvDocs('antique-docs',
        source=docs_file,
    )
    train_queries = TsvQueries('antique-train-queries',
        source=train_queries_file
    )
    train_qrels = TrecQrels('antique-train-qrels',
        source=train_qrels_file,
        defs=QREL_DEFS
    )
    test_queries = TsvQueries('antique-test-queries',
        source=test_queries_file
    )
    test_qrels = TrecQrels('antique-test-qrels',
        source=test_qrels_file,
        defs=QREL_DEFS
    )

    # Benchmarks
    # -----------------------------------------
    train = Benchmark('antique-train',
        docs=docs,
        queries=train_queries,
        qrels=train_qrels,
        citation=CITATION,
        metrics=MEASURES,
        desc='Official train set of the ANTIQUE dataset.'
    )
    test = Benchmark('antique-test',
        docs=docs,
        queries=test_queries,
        qrels=test_qrels,
        citation=CITATION,
        metrics=MEASURES,
        desc='Official test set of the ANTIQUE dataset.'
    )
    split200_train = Benchmark('antique-split200-train',
        derived_from=train,
        filter=Filter(query_ids=VALIDATION_QIDS, mode='exclude'),
        citation=CITATION,
        metrics=MEASURES,
        desc='antique-train without the 200 queries held out in '
             'antique-split200-valid.'
    )
    split200_valid = Benchmark('antique-split200-valid',
        derived_from=train,
        filter=Filter(query_ids=VALIDATION_QIDS, mode='include'),
        citation=CITATION,
        metrics=MEASURES,
        desc='A held-out subset of 200 queries from antique-train. Use with '
             'antique-split200-train.'
    )
    non_offensive = Benchmark('antique-test-non-offensive',
        derived_from=test,
        filter=Filter(query_ids=ids_from_lines(disallow_list_file), mode='exclude'),
        citation=CITATION,
        metrics=MEASURES,
        desc='antique-test without the queries the ANTIQUE authors deemed '
             '"offensive (and noisy)".'
    )


# Registration
# -----------------------------------------
# Constructing a node creates an object and nothing else. Registering the five
# benchmarks pulls in everything they depend on (tables, then files), so only
# the roots are listed. Anything not registered here simply isn't in the graph.
irds.register(train, test, split200_train, split200_valid, non_offensive)
