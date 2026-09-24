"""The ``irds:`` provider: the datasets bundled with ir_datasets.

Dataset modules register into it::

    from ir_datasets.v2 import irds
    irds.register(train, test)

It is exposed as the ``irds`` entry point in the ``ir_datasets.providers``
group (see pyproject.toml), and also added to the default graph when
``ir_datasets.v2`` is imported, so an editable checkout works before its entry
points are installed.
"""
from pathlib import Path

from .registry import Provider

irds = Provider(
    'irds',
    package='ir_datasets.v2.datasets',
    manifest_path=Path(__file__).parent / 'manifest.json',
)
