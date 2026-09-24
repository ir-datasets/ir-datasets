"""Generate the legacy cache map from v1's downloads.json.

v2 never re-downloads data a user already has: it finds an existing v1 cache
file and migrates it in place, once, to its v2 location (see
``sources.migrate_legacy``) -- a symlink at the old v1 path keeps anything
still looking there working. To find it in the first place needs knowing
where v1 put things, which is derivable -- v1 caches each file at
``<home>/<dataset>/<cache_path>`` -- so the map is generated rather than
hand-maintained.

Keyed by md5, so it is independent of v2 node names: any v2 Resource declaring the
same md5 finds the user's existing copy, whatever it is called now.

    python -m ir_datasets.v2.migrate
"""
import argparse
import json
from pathlib import Path

import pkgutil


def build_map():
    data = json.loads(pkgutil.get_data('ir_datasets', 'etc/downloads.json'))
    out = {}
    for dataset, entries in sorted(data.items()):
        if not isinstance(entries, dict):
            continue
        for key, dlc in entries.items():
            if not isinstance(dlc, dict):
                continue
            md5, cache_path = dlc.get('expected_md5'), dlc.get('cache_path')
            if md5 and cache_path:
                # v1: base_path = home/<dataset>, cache relative to it
                out.setdefault(md5, f'{dataset}/{cache_path}')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ir_datasets-v2-migrate')
    parser.add_argument(
        '--out', default=str(Path(__file__).parent / 'legacy_cache.json'))
    args = parser.parse_args(argv)
    mapping = build_map()
    with open(args.out, 'w') as fout:
        json.dump(mapping, fout, indent=1, sort_keys=True)
        fout.write('\n')
    print(f'wrote {args.out}: {len(mapping)} v1 cache paths')


if __name__ == '__main__':
    main()
