"""The ``clirmatrix:`` provider: CLIRMatrix's own namespace.

Split out from ``irds`` (which every other bundled family shares) because
CLIRMatrix isn't like them: it's generator-only (no node registrations -- see
``datasets/clirmatrix.py``'s own docstring), on the order of 150,000
addressable names, and its own self-contained vocabulary-reuser rather than a
family that needs ``irds``'s shared registry for anything but the node types
and edge kinds it borrows (``irds:Benchmark``, ``irds:Docs``,
``irds:derived_from``, ...) -- reuse that ``check_type``/``add_edge`` already
allow any provider to do, not something requiring shared registration.

Exposed as the ``clirmatrix`` entry point in the ``ir_datasets.providers``
group (see pyproject.toml), and also added to the default graph when
``ir_datasets.v2`` is imported, so an editable checkout works before its
entry points are installed -- same as ``irds``/``hf``.

No ``package=``: there's nothing to bootstrap-import (``ir_datasets/v2/__init__.py``
already imports ``datasets.clirmatrix`` unconditionally at package-import time
to register its generator rules -- see its own comment on why that import
can't be deferred to first lookup).
"""
from .registry import ManifestProvider

clirmatrix = ManifestProvider('clirmatrix')
