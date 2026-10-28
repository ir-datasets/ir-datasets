"""Attestations: records that someone hashed, inspected or evaluated a dataset node, held by a server
and exposed in the graph as the ``attest:`` provider. See README.md.

    from ir_datasets.v2 import attest
    attest.attest_hash('irds:antique-docs')
"""
from .client import (AttestError, Client, attest_evaluation, attest_hash, attest_visual,
                     SAVED, build_record, environment, for_node, hash_record, revoke, server_url)
from .crypto import Identity
