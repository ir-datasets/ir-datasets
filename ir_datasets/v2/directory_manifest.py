"""File manifests for ``Directory`` resources.

A manifest lists every file in a directory tree as one JSON object per line
(optionally gzipped, by extension)::

    {"path": "FBIS/FB396001", "size": 123456, "sha256": "..."}

``path`` is relative to the directory root, with ``/`` separators. It always
carries sizes and (unless generated with ``--no-hash``) a hash -- keyed by
algorithm (``sha256`` from this module, ``md5`` where built from a
distributor's own checksum files, as for ClueWeb) -- even though the
automatic first-access check (``Directory``) only compares paths and sizes:
the hashes are there so a full content check can be added later without
regenerating anything. A manifest holds hashes and names, not data, so it can
be shared for corpora whose contents cannot be.

Generate one from a copy you hold::

    python -m ir_datasets.v2.directory_manifest /path/to/corpus --out corpus.jsonl.gz
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def _open(path, mode):
    return gzip.open(path, mode + 't') if str(path).endswith('.gz') else open(path, mode)


def read_manifest(path):
    """Yield ``{"path", "size", ...}`` dicts from a manifest file."""
    with _open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_files(root):
    """Every file under ``root`` (following symlinks, so a tree assembled from
    linked directories is seen as it will be read), as sorted relative
    ``/``-separated paths."""
    root = Path(root)
    out = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            out.append((Path(dirpath) / name).relative_to(root).as_posix())
    return sorted(out)


def _hash_file(path, algo):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root, out, *, algo='sha256', hash=True, progress=None):
    """Write a manifest for the tree at ``root``. Returns the file count."""
    root = Path(root)
    n = 0
    with _open(out, 'w') as f:
        for rel in iter_files(root):
            full = root / rel
            entry = {'path': rel, 'size': os.stat(full).st_size}
            if hash:
                entry[algo] = _hash_file(full, algo)
            f.write(json.dumps(entry, sort_keys=True) + '\n')
            n += 1
            if progress and n % 1000 == 0:
                progress(n)
    return n


def check_files_and_sizes(root, manifest_path, *, max_problems=10):
    """Compare a tree against a manifest by existence and size only (``stat``,
    no reads). Returns ``(n_files, total_bytes, problems)``; ``problems`` is
    capped at ``max_problems`` messages plus a count of the rest."""
    root = Path(root)
    n = total = bad = 0
    problems = []
    for entry in read_manifest(manifest_path):
        n += 1
        total += entry['size']
        try:
            actual = os.stat(root / entry['path']).st_size
        except OSError:
            problem = f"missing: {entry['path']}"
        else:
            problem = None if actual == entry['size'] else (
                f"size {actual} != expected {entry['size']}: {entry['path']}")
        if problem:
            bad += 1
            if len(problems) < max_problems:
                problems.append(problem)
    if bad > len(problems):
        problems.append(f'... and {bad - len(problems)} more')
    return n, total, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description='Write a file manifest for a directory tree.')
    parser.add_argument('root')
    parser.add_argument('--out', required=True, help='manifest path (.gz to compress)')
    parser.add_argument('--algo', default='sha256')
    parser.add_argument('--no-hash', action='store_true', help='record paths and sizes only')
    args = parser.parse_args(argv)
    n = write_manifest(args.root, args.out, algo=args.algo, hash=not args.no_hash,
                       progress=lambda k: print(f'{k} files...', flush=True))
    print(f'wrote {args.out}: {n} files')


if __name__ == '__main__':
    main()
