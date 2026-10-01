"""WaPo (The Washington Post collection) -- a v2 dataset family (corpus for
TREC Common Core 2018 and TREC News 2018/2019/2020; the benchmarks
themselves live in ``trec_core.py``/``trec_news.py`` -- see those modules'
docstrings for why they're no longer bundled here).

The corpus is LDC/NIST-gated and not automatically downloadable -- like
``nyt.py``, a single-file ``Resource`` (not a ``Directory`` -- each version
is one tar.gz) pointed at by ``Source.external(default, old_locations=..., instructions=...)``, at
``<home>/external/wapo-v{2,4}.tar.gz`` (v1's old
``<home>/wapo/...`` still read as fallback).

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
from ir_datasets.datasets.wapo import WapoDoc, WapoDocs as _V1WapoDocs
from ir_datasets.v2 import DocTable, Resource, Source, irds
from ir_datasets.v2.formats import Parser

NAME = 'wapo'

DUA = ("The Washington Post collection is distributed by NIST under a data "
       "usage agreement; request access at <https://trec.nist.gov/data/wapost/>.")

V2_INSTRUCTIONS = [
    "Request the Washington Post collection from NIST: <https://trec.nist.gov/data/wapost/>.\nYour organization may already have a copy, in which case you may only need to complete a new \"Individual Agreement\". Otherwise your organization must file the \"Organizational agreement\" with NIST; once processed, you will receive a password-protected download link.",
    "Download the source file once your request is approved.",
]
V3_INSTRUCTIONS = V2_INSTRUCTIONS
V4_INSTRUCTIONS = V2_INSTRUCTIONS


class _WapoDocsParser(Parser):
    name = 'WapoDocs'

    def __init__(self, file_name):
        self.file_name = file_name

    def build(self, source, node):
        return _V1WapoDocs(source, self.file_name)


# license verified 2026-09-30: terms of the NIST TREC Washington Post Organization Application (research-only, no commercial use): https://trec.nist.gov/data/wapost/Organization%20Application.pdf
with irds.defaults(dua=DUA, lang='en', license='https://trec.nist.gov/data/wapost/Organization%20Application.pdf'):
    # Files
    # -----------------------------------------
    docs_v2_file = Resource('wapo-v2.tar.gz',
        sources=[Source.external('wapo-v2.tar.gz', old_locations=[f'{NAME}/WashingtonPost.v2.tar.gz'], instructions=V2_INSTRUCTIONS)],
        hash='md5:ce6e93f6ce9959b72c2de4f8d12089ab',
        size=1_632_868_829,
    )
    docs_v3_file = Resource('wapo-v3.tar.gz',
        sources=[Source.external('wapo-v3.tar.gz', old_locations=[f'{NAME}/WashingtonPost.v3.tar.gz'], instructions=V3_INSTRUCTIONS)],
        hash='md5:eef2db39c95e0dfa1ca8b53d6ff15143',
        size=1_942_041_158,
    )
    docs_v4_file = Resource('wapo-v4.tar.gz',
        sources=[Source.external('wapo-v4.tar.gz', old_locations=[f'{NAME}/WashingtonPost.v4.tar.gz'], instructions=V4_INSTRUCTIONS)],
        hash='md5:b45b8d34393b4df72737c11aa7fb2b3d',
        size=2_538_859_710,
    )

    # Tables
    # -----------------------------------------
    docs_v2 = DocTable('wapo-v2',
        source=docs_v2_file,
        parser=_WapoDocsParser('WashingtonPost.v2/data/TREC_Washington_Post_collection.v2.jl'),
        count_hint=595_037,
    )
    docs_v4 = DocTable('wapo-v4',
        source=docs_v4_file,
        parser=_WapoDocsParser('WashingtonPost.v4/data/TREC_Washington_Post_collection.v4.jl'),
        count_hint=728_626,
    )


# Registration
# -----------------------------------------
irds.register(docs_v2, docs_v3_file, docs_v4)
