"""Ambient defaults for the fields a whole family of nodes shares.

Some fields repeat on every node of a family -- a data-usage agreement on each
file, the language on each table::

    with irds.defaults(dua=DUA, lang='en'):
        docs = TsvDocs('antique-docs', source=docs_file)
        ...

An explicit argument always wins; these only fill in what was left unset.
Blocks nest, innermost first.

``defaults`` is a *provider* method: a provider may only default fields it has
declared (``irds.defaultable('dua', 'lang')``). The whitelist exists because
anything that determines a node's identity or its data -- a name, a source, an
md5 -- inherited from an enclosing block several screens away would be
miserable to debug. Passing an undeclared field is an error, not a silent
no-op, so the fence cannot erode by accident.

Node constructors consult ``default(key, value)``; this module is just the
stack.
"""
import contextlib

_STACK = []


@contextlib.contextmanager
def push(kwargs):
    _STACK.append(kwargs)
    try:
        yield
    finally:
        _STACK.pop()


def default(key, value):
    """Resolve one field: an explicit value wins, else the innermost default."""
    if value is not None and value != () and value != []:
        return value
    for frame in reversed(_STACK):
        if key in frame:
            return frame[key]
    return value


def active():
    """The defaults currently in effect (innermost last). For introspection."""
    merged = {}
    for frame in _STACK:
        merged.update(frame)
    return merged
