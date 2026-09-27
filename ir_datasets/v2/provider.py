"""The ``irds:`` provider: the datasets bundled with ir_datasets.

Dataset modules register into it::

    from ir_datasets.v2 import irds
    irds.register(train, test)

It is exposed as the ``irds`` entry point in the ``ir_datasets.providers``
group (see pyproject.toml), and also added to the default graph when
``ir_datasets.v2`` is imported, so an editable checkout works before its entry
points are installed.

A ``ManifestProvider`` -- the batteries-included implementation of
``protocols.Provider`` -- rather than something more minimal, because this
package's own catalog is exactly the case it's for: static, worth freezing,
and big enough that registration/vocabulary/generators earn their keep.
"""
from pathlib import Path

from .registry import ManifestProvider

irds = ManifestProvider(
    'irds',
    package='ir_datasets.v2.datasets',
    manifest_path=Path(__file__).parent / 'manifest.json',
)
