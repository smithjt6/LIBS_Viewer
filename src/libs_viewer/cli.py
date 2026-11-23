'''
`libs-viewer` command line.

    libs-viewer analyze DIR    run a whole temperature study and write the tables
    libs-viewer paths          where libs_viewer keeps its files
    libs-viewer nist status    is the NIST database built, and how big is it
    libs-viewer nist build     build / top up the NIST database (user only, uses the network)

`ui` and `mcp` subcommands arrive with roadmap phases 3 and 5.
'''

import argparse
import sys

from .core import paths
from .nist import build, query


def _key_values(items: list[str] | None) -> dict[str, float] | None:
    '''["max_half_window=60", "lam=1e7"] -> {"max_half_window": 60.0, "lam": 1e7}.'''
    if not items:
        return None
    out = {}
    for item in items:
        name, sep, value = item.partition('=')
        try:
            out[name.strip()] = float(value)
        except ValueError:
            sep = ''
        if not sep or not name.strip():
            raise ValueError(f'--baseline-param needs NAME=NUMBER, got {item!r}')
    return out


def _cmd_analyze(args):
    from .libs.study import run_study       # keeps `libs-viewer paths` fast
    from .nist.query import NISTDatabaseMissing
    try:
        result = run_study(args.folder, elements=args.elements, baseline=args.baseline,
                           baseline_params=_key_values(args.baseline_param),
                           pattern=args.pattern, manifest=args.manifest,
                           temperatures=args.temperatures, ref=args.ref, n_lines=args.lines,
                           min_prominence=args.min_prominence, interferents=args.interferents,
                           scatter=not args.no_scatter, db_path=args.db)
    except (ValueError, FileNotFoundError, NISTDatabaseMissing) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    print(result.summary())
    if args.out:
        written = result.save(args.out)
        print(f'\nwrote {len(written)} files to {written[0].parent}')
    return 0


def _cmd_paths(_args):
    print(f'data folder : {paths.data_dir()}   (override with ${paths.ENV_DATA_DIR})')
    print(f'NIST db     : {paths.nist_db_path()}')
    print(f'NIST cache  : {paths.nist_cache_dir()}')


def _cmd_nist_status(args):
    info = query.database_info(args.db)
    print(f"database : {info['path']}")
    if not info['exists']:
        print('status   : not built (run `libs-viewer nist build`)')
        return 1
    print(f"size     : {info['size_bytes'] / 1e6:.1f} MB, {info['elements']} elements")
    for table, n in info['rows'].items():
        print(f'  {table:20s} {n:>8,} rows')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='libs-viewer', description='LIBS Viewer tools.')
    sub = parser.add_subparsers(dest='command', required=True)

    a = sub.add_parser('analyze', help='run a whole temperature study on a folder of .tdms files',
                       description='Example: libs-viewer analyze data/end_to_end --pattern '
                                   '"{element}/{sample}/*{temperature}*" --temperatures 300 350 400 '
                                   '--elements Ce Gd --out output/run1')
    a.add_argument('folder', help='study folder')
    how = a.add_mutually_exclusive_group(required=True)
    how.add_argument('--pattern', help='file-name template, e.g. "{element}/{sample}/*{temperature}*"')
    how.add_argument('--manifest', help='CSV or XLSX with columns file, element, sample, temperature[, use_columns]')
    a.add_argument('--elements', nargs='+', help='elements to analyse (default: all in the study)')
    a.add_argument('--baseline', default='snip', help='baseline method: snip (default) or airpls')
    a.add_argument('--baseline-param', action='append', metavar='NAME=VALUE',
                   help='baseline setting, repeatable, e.g. max_half_window=60 (SNIP) or lam=1e7 (airPLS)')
    a.add_argument('--temperatures', nargs='+', type=float, help='temperatures used, e.g. 300 350 400')
    a.add_argument('--ref', type=float, default=300, help='reference temperature (default 300)')
    a.add_argument('--lines', type=int, default=4, help='lines per element (default 4)')
    a.add_argument('--min-prominence', type=float, default=0.25,
                   help='line detection threshold, x median intensity (default 0.25)')
    a.add_argument('--interferents', nargs='+', help='extra elements whose lines to avoid')
    a.add_argument('--no-scatter', action='store_true', help='skip the location-to-location scatter')
    a.add_argument('--db', help='NIST database (default: the libs_viewer data folder)')
    a.add_argument('--out', help='folder to write the tables, settings.json and summary.txt into')
    a.set_defaults(func=_cmd_analyze)

    sub.add_parser('paths', help='show where libs_viewer keeps its files').set_defaults(func=_cmd_paths)

    nist = sub.add_parser('nist', help='NIST Handbook line database')
    nist_sub = nist.add_subparsers(dest='nist_command', required=True)
    b = nist_sub.add_parser('build', help='build or top up the database (downloads from physics.nist.gov)')
    build.add_arguments(b)
    b.set_defaults(func=build.run)
    s = nist_sub.add_parser('status', help='show whether the database exists and its row counts')
    s.add_argument('--db', help='database file (default: the libs_viewer data folder)')
    s.set_defaults(func=_cmd_nist_status)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
