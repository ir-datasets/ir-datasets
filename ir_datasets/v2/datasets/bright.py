"""BRIGHT — a reasoning-intensive retrieval benchmark.

12 domains (StackExchange topics, coding, theorem-proving, competition math),
each with docs + queries + qrels. Unlike most families in this catalog,
BRIGHT's qrels are embedded in the *same* parquet file as its queries (not a
separate qrels file) -- reflected here by ``QueryTable`` and ``QrelTable`` both
declaring the same queries Resource as their ``source``. Queries also carry
up to 5 auxiliary "reasoning" fields, one per LLM, each sourced from its own
parquet file and joined in by v1's own ``BrightQueries.queries_iter``.

8 of the 12 domains additionally have a "long" variant: same queries (the
QueryTable object is reused verbatim, not re-declared) against long-form
documents, with qrels re-derived from the same queries file's
``gold_ids_long`` field instead of ``gold_ids``.

Reuses v1's ``BrightDocs``/``BrightQueries``/``BrightQrels`` handler classes
directly as v2 ``parser=`` wrappers -- the usual "thin layer over v1
machinery" every family in this catalog uses.

One quirk transcribed faithfully from v1's own ``downloads.json``, not fixed
here: the ``aops`` and ``theoremqa-questions`` doc files share an identical
md5/size. That's a property of the upstream data, not something this
migration should second-guess.
"""
from ir_datasets.datasets.bright import (
    BrightDocs as _V1BrightDocs, BrightQueries as _V1BrightQueries, BrightQrels as _V1BrightQrels,
    REASONING_FIELDS,
)

from ir_datasets.v2 import Benchmark, DocTable, QrelTable, QueryTable, Resource, Suite, irds
from ir_datasets.v2.formats import Parser

CITATION = 'dblp:conf/iclr/SuYXSMWLSST0YA025'
METRICS = ['nDCG@10']
QRELS_DEFS = {1: 'Relevant', -100: 'Excluded from evaluation'}

SHORT_SUBSETS = ['biology', 'earth-science', 'economics', 'psychology', 'robotics',
                 'stackoverflow', 'sustainable-living', 'leetcode', 'pony', 'aops',
                 'theoremqa-theorems', 'theoremqa-questions']
#: subsets that also have a long-document variant (the other 4 -- leetcode,
#: aops, theoremqa-theorems, theoremqa-questions -- don't).
LONG_SUBSETS = ['biology', 'earth-science', 'economics', 'psychology', 'robotics',
                'stackoverflow', 'sustainable-living', 'pony']

#: subset -> (url, md5, size), transcribed from ir_datasets/etc/downloads.json.
DOCS_SOURCES = {
    'biology': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/biology-00000-of-00001.parquet', 'f5634c69200296066adf9816f5e4f729', 11046045),
    'earth-science': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/earth_science-00000-of-00001.parquet', '210a3b5cdc06cfd9ca6728fa97fb8067', 23084671),
    'economics': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/economics-00000-of-00001.parquet', '51f017a94fc4bccb68e44c85640a2feb', 10969621),
    'psychology': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/psychology-00000-of-00001.parquet', 'cb0a14e59845efe80fdc1fa92c379605', 11430533),
    'robotics': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/robotics-00000-of-00001.parquet', '595fb0d3075eaca19298a51bb706a513', 7874186),
    'stackoverflow': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/stackoverflow-00000-of-00001.parquet', 'e1e3b210340e7fb5ec3782b57525a6ee', 39493317),
    'sustainable-living': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/sustainable_living-00000-of-00001.parquet', 'e7f3e97d3314f2afb662c8779b904ce9', 11720059),
    'leetcode': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/leetcode-00000-of-00001.parquet', 'eda94d3cb40aa9b1f55030d1631acf72', 210510658),
    'pony': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/pony-00000-of-00001.parquet', 'f0805cab692e3a4f4b69819b743c0a82', 1125040),
    'aops': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/aops-00000-of-00001.parquet', '43d86de9400e68911a7d419de8105c61', 65301334),
    'theoremqa-theorems': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/theoremqa_theorems-00000-of-00001.parquet', '93fb1207b33e0dffb79a7923326c060d', 7632381),
    'theoremqa-questions': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/documents/theoremqa_questions-00000-of-00001.parquet', '43d86de9400e68911a7d419de8105c61', 65301334),
}

LONG_DOCS_SOURCES = {
    'biology': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/long_documents/biology-00000-of-00001.parquet', 'adf570ca7c9ded642b9dc5dbca80d3ad', 9063521),
    'earth-science': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/long_documents/earth_science-00000-of-00001.parquet', 'fdcfd781a0277a07edff9e86dd7b5391', 16229287),
    'economics': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/long_documents/economics-00000-of-00001.parquet', 'fb84391a7514cb110a82d88614f5b846', 9422303),
    'psychology': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/long_documents/psychology-00000-of-00001.parquet', 'b8c68cb60fcbed3404c13fbb5d006186', 9855475),
    'robotics': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/long_documents/robotics-00000-of-00001.parquet', '94408fb87b6f7ec43622acdd2ce259e0', 6390001),
    'stackoverflow': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/long_documents/stackoverflow-00000-of-00001.parquet', 'f328785df5459f92934f3580e4252bb0', 43080248),
    'sustainable-living': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/long_documents/sustainable_living-00000-of-00001.parquet', '571c7966369496f852a571996574537f', 9724495),
    'pony': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/long_documents/pony-00000-of-00001.parquet', '67ee14f338db467071b4afc1a4417dc1', 813435),
}

QUERIES_SOURCES = {
    'biology': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/biology-00000-of-00001.parquet', '53d20abac1f80d0ebd223da63a5a7d45', 200655),
    'earth-science': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/earth_science-00000-of-00001.parquet', 'd8a46cd50ff605094bf7460ce387ff4c', 184093),
    'economics': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/economics-00000-of-00001.parquet', '618302357566c1d1aa45dfdc13d471ef', 219518),
    'psychology': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/psychology-00000-of-00001.parquet', '16c99781402179d309696a68d0a1ab86', 183889),
    'robotics': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/robotics-00000-of-00001.parquet', '1ac3ec10dda663f083f28b33d901fad1', 178820),
    'stackoverflow': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/stackoverflow-00000-of-00001.parquet', 'e676fb3e8f822237d67287d75a769155', 250458),
    'sustainable-living': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/sustainable_living-00000-of-00001.parquet', 'e8e73cf45e751e447a470cdb27974f42', 218151),
    'leetcode': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/leetcode-00000-of-00001.parquet', '33e413c4f4c5b80b074b979807eaa5e6', 168560),
    'pony': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/pony-00000-of-00001.parquet', '0decd583ecef3e17b861d0697585033f', 27722),
    'aops': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/aops-00000-of-00001.parquet', '26f1458f835aeba9c915c1eda4321ec0', 1338908),
    'theoremqa-theorems': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/theoremqa_theorems-00000-of-00001.parquet', 'e45115bb80e939fc9a063514f2331e54', 124476),
    'theoremqa-questions': ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/examples/theoremqa_questions-00000-of-00001.parquet', 'eac182e1594f199acda6d18fdbb8a4ef', 1494939),
}

#: (subset, v1 reasoning field name) -> (url, md5, size). Keys use the exact
#: field names v1's own REASONING_FIELDS dict is keyed by (e.g.
#: 'Gemini-1.0_reason'), since that's what BrightQueries.queries_iter looks
#: up its reasoning_dlcs dict by.
REASONING_SOURCES = {
    ('biology', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/biology-00000-of-00001.parquet', '6980c85b5f7d8dd9d57f53797fa5a2f5', 331416),
    ('biology', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/biology-00000-of-00001.parquet', '97927f4b8676eac869261d294f3c6165', 319709),
    ('biology', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/biology-00000-of-00001.parquet', '0616a6f7505d7f7676d1b8c6c2362e52', 359062),
    ('biology', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/biology-00000-of-00001.parquet', '4fe614bf0cbbc2915de4ff0424bf4ea4', 267901),
    ('biology', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/biology-00000-of-00001.parquet', '55c2348607e5ed6831ca9a74c8e48c09', 337499),
    ('earth-science', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/earth_science-00000-of-00001.parquet', '136077f52f6dc20d24bbcb9547925f42', 331551),
    ('earth-science', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/earth_science-00000-of-00001.parquet', 'cbd64a821f0015febb993650f901b29d', 326183),
    ('earth-science', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/earth_science-00000-of-00001.parquet', '17f2798c12b94762b659e16730f834b4', 368863),
    ('earth-science', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/earth_science-00000-of-00001.parquet', '2779fadd62d7e76955948dbfed814ad2', 255820),
    ('earth-science', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/earth_science-00000-of-00001.parquet', '5255ef9014e93a384c1c8557576c9de7', 350327),
    ('economics', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/economics-00000-of-00001.parquet', '8bf0d9f5a1506b5040c38c11cb7aa0c1', 352295),
    ('economics', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/economics-00000-of-00001.parquet', 'a81968f82882a180a490e060740865ab', 331419),
    ('economics', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/economics-00000-of-00001.parquet', '620261a22068afa14d72a46d71fdade6', 365889),
    ('economics', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/economics-00000-of-00001.parquet', 'f77230038f49d40bebea84820b028b37', 275745),
    ('economics', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/economics-00000-of-00001.parquet', 'c2ff8d9e1b6a02657176dd9a72128042', 350900),
    ('psychology', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/psychology-00000-of-00001.parquet', '659c29d093935172e413ab38d96d274c', 325242),
    ('psychology', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/psychology-00000-of-00001.parquet', '95b92706bbd42b64e239d928ad946431', 292087),
    ('psychology', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/psychology-00000-of-00001.parquet', '54e7175bbd51f8275cdef31a640d0f85', 324585),
    ('psychology', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/psychology-00000-of-00001.parquet', '53c6abe1e3386dcd23ea6a1533c06435', 230598),
    ('psychology', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/psychology-00000-of-00001.parquet', '5f042ac62879a5a8a4d5d46b10effaa7', 301536),
    ('robotics', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/robotics-00000-of-00001.parquet', '16da9f768f222c5a64e0bd15fe086dd7', 250109),
    ('robotics', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/robotics-00000-of-00001.parquet', 'aaf139689dfe5241d5977fff5c19f874', 237668),
    ('robotics', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/robotics-00000-of-00001.parquet', '2f222205cb7af5f1c4d87a0dec795aec', 290146),
    ('robotics', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/robotics-00000-of-00001.parquet', 'e0818b8023150f2cd8abcaf7066a392d', 180516),
    ('robotics', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/robotics-00000-of-00001.parquet', 'ce7bb3c08888ca50793d94e4782e7720', 230515),
    ('stackoverflow', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/stackoverflow-00000-of-00001.parquet', '6d206ee701c8822311257cdf100af346', 341441),
    ('stackoverflow', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/stackoverflow-00000-of-00001.parquet', '9fe4d84bf0e58b2c53450b386e2ebf41', 325899),
    ('stackoverflow', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/stackoverflow-00000-of-00001.parquet', '13dd944442241864e5f98c01c781a897', 381634),
    ('stackoverflow', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/stackoverflow-00000-of-00001.parquet', 'd236a5f777875c3d84dd6065ce4c0642', 271334),
    ('stackoverflow', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/stackoverflow-00000-of-00001.parquet', '36e08f78c9c4090b9aa6910366ae87d4', 325909),
    ('sustainable-living', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/sustainable_living-00000-of-00001.parquet', '519f6480bd93241207e387e87f25ec08', 367575),
    ('sustainable-living', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/sustainable_living-00000-of-00001.parquet', '7f422ff10c1a7981cfb8ab748cf07fbf', 348982),
    ('sustainable-living', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/sustainable_living-00000-of-00001.parquet', '97002679a283b3d26b4f12b011c84f1c', 393821),
    ('sustainable-living', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/sustainable_living-00000-of-00001.parquet', '259ec3a093e45f602eababad40203d86', 278439),
    ('sustainable-living', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/sustainable_living-00000-of-00001.parquet', '5d86ce8e87deb411c0f9a1da1b221340', 372805),
    ('leetcode', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/leetcode-00000-of-00001.parquet', '0c9713b2e0e57e4c23af91af2ce41544', 248405),
    ('leetcode', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/leetcode-00000-of-00001.parquet', '9a8b26506063a2ad769b547f7e78d46a', 244469),
    ('leetcode', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/leetcode-00000-of-00001.parquet', '4e7af48992131deab2fed9ec232e6e37', 303035),
    ('leetcode', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/leetcode-00000-of-00001.parquet', 'd0f167d9b3d9f8669dfddbfad1b962aa', 199459),
    ('leetcode', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/leetcode-00000-of-00001.parquet', 'd5d1b80fe71026781403496a6dc54d02', 230999),
    ('pony', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/pony-00000-of-00001.parquet', '53ceff9e21ad32137ed9bbeaf8f536fc', 123344),
    ('pony', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/pony-00000-of-00001.parquet', '23fb3002b7a88d5b0f4a255242aef3d6', 130573),
    ('pony', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/pony-00000-of-00001.parquet', 'fc2cdfe6539c9812d4e4d3a726128ad1', 180494),
    ('pony', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/pony-00000-of-00001.parquet', '18e292091279382f41be47ee2c8419c7', 88676),
    ('pony', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/pony-00000-of-00001.parquet', 'ec910733e78c7e24130d1d20aeb55082', 109129),
    ('aops', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/aops-00000-of-00001.parquet', 'fe4ab6b888f85011bc80579ac04748ed', 1428427),
    ('aops', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/aops-00000-of-00001.parquet', 'b6794dbc9074e94a3427787721070241', 1418228),
    ('aops', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/aops-00000-of-00001.parquet', 'b1ad5adf0d6e352eb01303bd6978ec97', 1508370),
    ('aops', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/aops-00000-of-00001.parquet', 'c5a79b264ef5aeebc255e650ed5a65c8', 1409927),
    ('aops', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/aops-00000-of-00001.parquet', '3c2017f01a42c1dc65ca0e3c50a90949', 1428793),
    ('theoremqa-theorems', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/theoremqa_theorems-00000-of-00001.parquet', '542068936715c95dc8f26711188fe13f', 194993),
    ('theoremqa-theorems', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/theoremqa_theorems-00000-of-00001.parquet', 'e08dc27a2e3840fcb18e8524f3f9e793', 180966),
    ('theoremqa-theorems', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/theoremqa_theorems-00000-of-00001.parquet', '7c822aee62f5b6b833b2090b84880c24', 224366),
    ('theoremqa-theorems', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/theoremqa_theorems-00000-of-00001.parquet', '592937c1342712f54e6cdb6c1d7e06af', 163375),
    ('theoremqa-theorems', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/theoremqa_theorems-00000-of-00001.parquet', '667e20730881552c8ef31869feb2047d', 191675),
    ('theoremqa-questions', 'Gemini-1.0_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/Gemini-1.0_reason/theoremqa_questions-00000-of-00001.parquet', '87333615aa3cc53f45e3e384764a48d5', 1651815),
    ('theoremqa-questions', 'claude-3-opus_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/claude-3-opus_reason/theoremqa_questions-00000-of-00001.parquet', 'a1c276178715e2fbaa1cefa4bcecedd6', 1621291),
    ('theoremqa-questions', 'gpt4_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/gpt4_reason/theoremqa_questions-00000-of-00001.parquet', 'a24fa40d750296130dc4be92863bcfb6', 1720991),
    ('theoremqa-questions', 'grit_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/grit_reason/theoremqa_questions-00000-of-00001.parquet', 'f616552c2991c75b5d57930c1b518f82', 1580624),
    ('theoremqa-questions', 'llama3-70b_reason'): ('https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main/llama3-70b_reason/theoremqa_questions-00000-of-00001.parquet', 'a645b908f1344081c8bf7f3d819ef8f1', 1647316),
}


def _slug(field):
    """'Gemini-1.0_reason' -> 'gemini-1-0-reason' (flat v2 node-name style)."""
    return field.lower().replace('.', '-').replace('_', '-')


class _BrightDocsParser(Parser):
    name = 'BrightDocs'

    def __init__(self, subset):
        self.subset = subset

    def build(self, source, node):
        return _V1BrightDocs(self.subset, source)


class _BrightQueriesParser(Parser):
    name = 'BrightQueries'

    def __init__(self, reasoning_sources):
        self.reasoning_sources = reasoning_sources

    def build(self, source, node):
        return _V1BrightQueries(source, self.reasoning_sources)


class _BrightQrelsParser(Parser):
    name = 'BrightQrels'

    def __init__(self, gold_field):
        self.gold_field = gold_field

    def build(self, source, node):
        return _V1BrightQrels(source, gold_field=self.gold_field)


# license verified 2026-09-30: https://huggingface.co/datasets/xlangai/BRIGHT (card metadata: CC-BY-4.0)
with irds.defaults(lang='en', license='CC-BY-4.0'):
    benchmarks = {}          # short-document benchmarks: flat name -> Benchmark
    long_benchmarks = {}     # long-document variants (their own suite, bright-long)
    queries_by_subset = {}   # subset -> (queries_file Resource, QueryTable table)

    for subset in SHORT_SUBSETS:
        docs_url, docs_md5, docs_size = DOCS_SOURCES[subset]
        docs_file = Resource(f'bright-{subset}-docs.parquet', sources=[docs_url], hash=f'md5:{docs_md5}', size=docs_size)
        docs = DocTable(f'bright-{subset}-docs', source=docs_file, parser=_BrightDocsParser(subset))

        queries_url, queries_md5, queries_size = QUERIES_SOURCES[subset]
        queries_file = Resource(f'bright-{subset}-queries.parquet', sources=[queries_url], hash=f'md5:{queries_md5}', size=queries_size)

        reasoning_sources = {}
        for v1_field in REASONING_FIELDS:
            r_url, r_md5, r_size = REASONING_SOURCES[(subset, v1_field)]
            reasoning_sources[v1_field] = Resource(
                f'bright-{subset}-{_slug(v1_field)}.parquet', sources=[r_url], hash=f'md5:{r_md5}', size=r_size)

        queries = QueryTable(f'bright-{subset}-queries', source=queries_file,
                          derived_from=list(reasoning_sources.values()),
                          parser=_BrightQueriesParser(reasoning_sources))
        qrels = QrelTable(f'bright-{subset}-qrels', source=queries_file, defs=QRELS_DEFS,
                      parser=_BrightQrelsParser(gold_field='gold_ids'))

        queries_by_subset[subset] = (queries_file, queries)

        name = f'bright-{subset}'
        benchmarks[name] = Benchmark(name, docs=docs, queries=queries, qrels=qrels,
                                     citation=CITATION, metrics=METRICS,
                                     desc=f'BRIGHT: {subset} (reasoning-intensive retrieval).')

    for subset in LONG_SUBSETS:
        long_url, long_md5, long_size = LONG_DOCS_SOURCES[subset]
        long_docs_file = Resource(f'bright-{subset}-long-docs.parquet', sources=[long_url], hash=f'md5:{long_md5}', size=long_size)
        long_docs = DocTable(f'bright-{subset}-long-docs', source=long_docs_file, parser=_BrightDocsParser(subset))

        queries_file, queries = queries_by_subset[subset]
        # Same QueryTable object as the short variant -- the long variant only
        # differs in docs (long-form) and qrels (ids mapped to those docs).
        long_qrels = QrelTable(f'bright-{subset}-long-qrels', source=queries_file, defs=QRELS_DEFS,
                           parser=_BrightQrelsParser(gold_field='gold_ids_long'))

        name = f'bright-{subset}-long'
        long_benchmarks[name] = Benchmark(name, docs=long_docs, queries=queries, qrels=long_qrels,
                                     citation=CITATION, metrics=METRICS,
                                     desc=f'BRIGHT: {subset}, long-form documents.')


# Registration
# -----------------------------------------
irds.register(*benchmarks.values(), *long_benchmarks.values())

irds.register(Suite('bright', benchmarks=list(benchmarks.values()), citation=CITATION,
                    desc='The BRIGHT evaluation suite: reasoning-intensive retrieval '
                         'across 12 domains (StackExchange topics, coding, theorem-proving, '
                         'competition math).'))

irds.register(Suite('bright-long', benchmarks=list(long_benchmarks.values()), citation=CITATION,
                    desc='BRIGHT with long-form documents: the same queries as bright, '
                         'against long documents, for 8 of its 12 domains.'))
