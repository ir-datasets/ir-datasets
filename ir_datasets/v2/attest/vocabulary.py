"""The ``attest:`` graph classes, as protocols (see ``protocols.py``)."""
from typing import Protocol, runtime_checkable

from .. import protocols as proto
from ..protocols import graph_class


@graph_class
@runtime_checkable
class Attestation(proto.Node, Protocol):
    """A claim someone made about a node, held as a JSON record.

    Anybody may submit one anonymously; a signature ties it to an IdentityAttestation.
    Every record has ``attested_at`` and an optional ``environment``."""


@graph_class
@runtime_checkable
class HashAttestation(Attestation, Protocol):
    """Hashes someone computed for a file (of its bytes) or a table (of its parsed records)."""


@graph_class
@runtime_checkable
class VisualAttestation(Attestation, Protocol):
    """A person's report that they inspected a sample of a table's records by eye."""


@graph_class
@runtime_checkable
class EvaluationAttestation(Attestation, Protocol):
    """Scores a system obtained on a benchmark."""


@graph_class
@runtime_checkable
class IdentityAttestation(Attestation, Protocol):
    """A username bound to a public key, vouched for by its own signature."""
