"""Dataset definitions bundled with core (the empty-prefix provider).

One module per family. `freeze` imports these once to build the manifest; at
runtime only the module defining a requested node is imported.
"""
