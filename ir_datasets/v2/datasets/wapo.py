"""WaPo (The Washington Post collection) -- a v2 dataset family (corpus for
TREC Common Core 2018 and TREC News 2018/2019/2020; the benchmarks
themselves live in ``trec_core.py``/``trec_news.py`` -- see those modules'
docstrings for why they're no longer bundled here).

The corpus is LDC/NIST-gated and not automatically downloadable -- like
``nyt.py``, a single-file ``Resource`` (not a ``Directory`` -- each version
is one tar.gz) pointed at by ``Source.local(path, instructions=...)``, at the
same well-known path v1 used.

Reuses v1's ``WapoDocs`` directly as a v2 ``parser=`` wrapper -- it reads one
named member out of the downloaded tar.gz (``file_name``, which differs
between v2 and v4), same "thin layer over v1 machinery" pattern as every
other migrated family. ``WapoDoc``/``WapoDocMedia`` are imported from legacy
``wapo.py`` as record types only.

``v4`` (the newer corpus revision) has no benchmark of its own in v1 -- it's
registered as a bare, queryless doc collection, same as here.
``trec_cast.py`` is the one real external consumer of this corpus beyond
``trec_core.py``/``trec_news.py``, and imports ``docs_v2``/``docs_v4`` from
here by reference.
"""
import ir_datasets
from ir_datasets.datasets.wapo import WapoDoc, WapoDocs as _V1WapoDocs
from ir_datasets.v2 import DocTable, Resource, Source, irds
from ir_datasets.v2.formats import Parser

NAME = 'wapo'
BASE_PATH = ir_datasets.util.home_path() / NAME

DUA = ("The Washington Post collection is distributed by NIST under a data "
       "usage agreement; request access at <https://trec.nist.gov/data/wapost/>.")

V2_INSTRUCTIONS = (
    "The Washington Post collection can be requested here: "
    "<https://trec.nist.gov/data/wapost/>\n"
    "More details about the procedure can be found here: "
    "<https://ir-datasets.com/wapo.html#DataAccess>.\n"
    "Once completed, place/link the source file: {path}"
)
V4_INSTRUCTIONS = V2_INSTRUCTIONS


class _WapoDocsParser(Parser):
    name = 'WapoDocs'

    def __init__(self, file_name):
        self.file_name = file_name

    def build(self, source, node):
        return _V1WapoDocs(source, self.file_name)


with irds.defaults(dua=DUA, lang='en'):
    # Files
    # -----------------------------------------
    docs_v2_file = Resource('wapo-v2.tar.gz',
        sources=[Source.local(BASE_PATH / 'WashingtonPost.v2.tar.gz', instructions=V2_INSTRUCTIONS)],
        md5='ce6e93f6ce9959b72c2de4f8d12089ab',
    )
    docs_v4_file = Resource('wapo-v4.tar.gz',
        sources=[Source.local(BASE_PATH / 'WashingtonPost.v4.tar.gz', instructions=V4_INSTRUCTIONS)],
        md5='b45b8d34393b4df72737c11aa7fb2b3d',
    )

    # Tables
    # -----------------------------------------
    docs_v2 = DocTable('wapo-v2-docs',
        source=docs_v2_file,
        parser=_WapoDocsParser('WashingtonPost.v2/data/TREC_Washington_Post_collection.v2.jl'),
        count_hint=595_037,
    )
    docs_v4 = DocTable('wapo-v4-docs',
        source=docs_v4_file,
        parser=_WapoDocsParser('WashingtonPost.v4/data/TREC_Washington_Post_collection.v4.jl'),
        count_hint=728_626,
    )


# Registration
# -----------------------------------------
irds.register(docs_v2, docs_v4)
