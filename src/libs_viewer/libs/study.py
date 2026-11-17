'''
One-call temperature study: folder in, tables out.

    result = run_study('data/end_to_end', elements=['Ce', 'Gd'], baseline='snip',
                       pattern='{element}/{sample}/*{temperature}*', temperatures=[300, 350, 400])
    result.r_rel          # line ratios vs the reference temperature
    result.save('output/run1')

`run_study` runs the same steps as examples/end_to_end_temp_study.ipynb:
    1. study_plan          which file is which (pattern or manifest; no guessing)
    2. column choice       grid locations acquired with other settings than the rest of the
                           sample are dropped (and reported in `notes`)
    3. per-shot spectra    + saturated pixels per sample
    4. select_lines        NIST lines per element, other study elements as interferents
    5. analyze_sample      line / total areas with the chosen baseline, ratio tables
    6. scatter             location-to-location CV of every multi-location file

Everything it decided is recorded in `StudyResult.settings` and `StudyResult.notes`, and
`save()` writes both next to the tables. No plotting here (see plotting.plot_study).
'''

import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..nist.query import resolve_db
from .analysis import TOTAL, analyze_sample, line_windows, per_column_areas, ratio_tables
from .lines import select_lines
from .loading import LIBSFile, overview_table, study_plan
from .processing import BASELINE_METHODS, per_shot_spectrum, saturated_mask

# Settings that must match for two spectra of one sample to be comparable.
KEY_SETTINGS = ('Gate Delay (usec)', 'Energy')


@dataclass
class StudyResult:
    '''
    Everything one `run_study` call produced.

    Tables (rows indexed by sample, then line / total):
        areas      raw areas, columns = temperatures
        r_direct   A(T) / A(ref)
        r_rel      [A_line(T)/A_total(T)] / [A_line(ref)/A_total(ref)]; Total rows = R_total
        scatter    location-to-location CV (%) of every file with more than one location
    Also:
        plan       which file is which, with the grid locations actually used
        overview   acquisition settings of every file
        line_tables {element: select_lines table}; lines {sample: {label: nm}}
        notes      what was dropped, adjusted or could not be measured
        settings   the inputs and versions (provenance); saved as settings.json
        wl, spectra {sample: {T: per-shot spectrum}}, masks {sample: saturated pixels}
    '''
    settings: dict
    plan: pd.DataFrame
    overview: pd.DataFrame
    line_tables: dict[str, pd.DataFrame]
    lines: dict[str, dict[str, float]]
    areas: pd.DataFrame
    r_direct: pd.DataFrame
    r_rel: pd.DataFrame
    scatter: pd.DataFrame
    notes: list[str]
    wl: np.ndarray = field(repr=False)
    spectra: dict[str, dict] = field(repr=False)
    masks: dict[str, np.ndarray] = field(repr=False)

    @property
    def samples(self) -> list[str]:
        return list(self.spectra)

    def summary(self) -> str:
        '''Plain-text summary: inputs, picked lines, R_rel table and notes.'''
        s = self.settings
        out = [f"LIBS temperature study: {s['folder']}",
               f"elements {', '.join(s['elements'])} | baseline {s['baseline']} | "
               f"reference {s['ref']} C | {len(self.plan)} files, {len(self.samples)} samples", '',
               'Lines:']
        out += [f"  {sample}: {', '.join(lines) or '(none detected)'}" for sample, lines in self.lines.items()]
        out += ['', 'R_rel (line share of total light, relative to the reference):',
                self.r_rel.round(3).to_string()]
        if self.notes:
            out += ['', 'Notes:'] + [f'  - {n}' for n in self.notes]
        return '\n'.join(out)

    def save(self, out_dir: str | Path) -> list[Path]:
        '''
        Writes the tables as CSV plus settings.json (inputs, notes, picked lines) and
        summary.txt into `out_dir`, overwriting files of the same name. Refuses to write inside
        the data folder, so raw files are never touched.
        '''
        out = Path(out_dir).resolve()
        data = Path(self.settings['folder']).resolve()
        if out == data or data in out.parents:
            raise ValueError(f'refusing to write results inside the data folder {data}')
        out.mkdir(parents=True, exist_ok=True)
        plan = self.plan.assign(use_columns=[';'.join(c) if c else '' for c in self.plan.use_columns])
        tables = {'plan.csv': plan.set_index('file'), 'overview.csv': self.overview,
                  'areas.csv': self.areas, 'R_direct.csv': self.r_direct, 'R_rel.csv': self.r_rel,
                  'scatter.csv': self.scatter}
        tables.update({f'lines_{el}.csv': t.set_index('label') for el, t in self.line_tables.items()})
        written = []
        for name, df in tables.items():
            df.to_csv(out / name)
            written.append(out / name)
        meta = {**self.settings, 'lines': self.lines, 'notes': self.notes}
        (out / 'settings.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
        (out / 'summary.txt').write_text(self.summary() + '\n', encoding='utf-8')
        return written + [out / 'settings.json', out / 'summary.txt']


def _column_settings(f: LIBSFile, col: str) -> tuple:
    cp = f.column_params
    return tuple((k, float(cp.at[k, col])) for k in KEY_SETTINGS if k in cp.index)


def _choose_columns(plan: pd.DataFrame, files: dict[str, LIBSFile], notes: list[str]) -> list:
    '''
    Grid locations to use per file. Per sample, the most common KEY_SETTINGS combination
    (one vote per file) is the sample's setting; locations acquired otherwise are dropped.
    Explicit use_columns from a manifest are kept as given.
    '''
    chosen = {}
    for sample, g in plan.groupby('sample', sort=False):
        votes = Counter(s for k in g.file for s in {_column_settings(files[k], c) for c in files[k].spectra.columns})
        (best, n1), *rest = votes.most_common() + [((), 0)]
        if rest and rest[0][1] == n1 and len(votes) > 1:
            raise ValueError(f'{sample}: no single acquisition setting is the most common '
                             f'({dict(votes)}); set use_columns in a manifest')
        for row in g.itertuples():
            f = files[row.file]
            if row.use_columns is not None:
                unknown = set(row.use_columns) - set(f.spectra.columns)
                if unknown:
                    raise ValueError(f'{row.file}: unknown grid location(s) {sorted(unknown)}')
                chosen[row.Index] = row.use_columns
                continue
            match = [c for c in f.spectra.columns if _column_settings(f, c) == best]
            if len(match) == f.spectra.shape[1]:
                chosen[row.Index] = None
            elif not match:
                chosen[row.Index] = None
                notes.append(f'{row.file}: acquired with {dict(_column_settings(f, f.spectra.columns[0]))}, '
                             f'unlike the rest of {sample} ({dict(best)}); kept, but compare with care')
            else:
                chosen[row.Index] = match
                notes.append(f'{row.file}: used {len(match)} of {f.spectra.shape[1]} grid locations '
                             f'(the others were acquired with different {", ".join(KEY_SETTINGS)})')
    return [chosen[i] for i in plan.index]


def run_study(folder: str | Path, elements: list[str] | None = None, baseline: str = 'snip',
              pattern: str | None = None, manifest=None, temperatures=None, ref: float = 300,
              n_lines: int = 4, min_prominence: float = 0.25, interferents: list[str] | None = None,
              scatter: bool = True, db_path: str | Path | None = None) -> StudyResult:
    '''
    Runs a complete temperature study on a folder of Applied Spectra .tdms files.

    Args:
        folder: the study folder.
        elements: elements to analyse (default: every element in the plan).
        baseline: continuum removal for the total areas, 'snip' or 'airpls'.
        pattern / manifest / temperatures: how files map to element, sample and temperature;
            exactly one of pattern or manifest (see `study_plan`).
        ref: reference temperature; every sample needs a file at it.
        n_lines: lines to pick per element.
        min_prominence: line detection threshold as a multiple of the spectrum's median
            (0.25 finds weak lines on a high continuum; select_lines' stricter default is 2).
        interferents: extra elements whose lines must be avoided (the other study elements
            are always included).
        scatter: also compute the location-to-location scatter (slower with airpls).
        db_path: NIST database (default: the per-user database).

    Returns:
        StudyResult. Raises ValueError / FileNotFoundError for an ambiguous or incomplete study.
    '''
    if baseline not in BASELINE_METHODS:
        raise ValueError(f'baseline must be one of {BASELINE_METHODS}, got {baseline!r}')
    folder = Path(folder)
    db = resolve_db(db_path)                       # fail early if the NIST database is missing
    plan = study_plan(folder, pattern=pattern, manifest=manifest, temperatures=temperatures)

    if elements:
        absent = sorted(set(elements) - set(plan.element))
        if absent:
            raise ValueError(f'no files for element(s) {absent}; found {sorted(set(plan.element))}')
        plan = plan[plan.element.isin(elements)].reset_index(drop=True)
    elements = list(dict.fromkeys(plan.element))
    no_ref = [s for s, g in plan.groupby('sample') if ref not in set(g.temperature)]
    if no_ref:
        raise ValueError(f'sample(s) without a file at the reference temperature {ref}: {no_ref}')

    notes: list[str] = []
    files = {}
    for k in plan.file:
        with warnings.catch_warnings():            # mixed settings are handled below
            warnings.simplefilter('ignore', UserWarning)
            files[k] = LIBSFile.from_tdms(folder / k)
    plan['use_columns'] = _choose_columns(plan, files, notes)

    wl = files[plan.file[0]].wavelengths
    if not all(np.array_equal(f.wavelengths, wl) for f in files.values()):
        raise ValueError('files have different wavelength axes; analyse them separately')

    spectra, masks = {}, {}
    for sample, g in plan.groupby('sample', sort=False):
        spectra[sample] = {t: per_shot_spectrum(files[k], c)
                           for k, t, c in zip(g.file, g.temperature, g.use_columns)}
        masks[sample] = saturated_mask([files[k] for k in g.file], list(g.use_columns))

    line_tables, lines = {}, {}
    for el in elements:
        samples = list(dict.fromkeys(plan.loc[plan.element == el, 'sample']))
        others = [e for e in dict.fromkeys(elements + list(interferents or [])) if e != el]
        table = select_lines(el, wl, {s: spectra[s][ref] for s in samples}, n=n_lines,
                             interferents=others or None,
                             exclude=np.logical_or.reduce([masks[s] for s in samples]),
                             min_prominence=min_prominence, db_path=db)
        line_tables[el] = table
        picked = table[table.status == 'picked']
        for s in samples:
            lines[s] = {r.label: float(r.wavelength_nm) for r in picked.itertuples()
                        if s in r.detected_in.split(', ')}
            if not lines[s]:
                notes.append(f'{s}: no {el} line detected at {ref} C; only totals are reported')

    areas, r_direct, r_rel = {}, {}, {}
    for s in spectra:
        areas[s] = analyze_sample(wl, spectra[s], lines[s], method=baseline, exclude=masks[s], ref=ref)
        r = ratio_tables(areas[s], ref=ref)
        r_direct[s], r_rel[s] = r['R_direct'], r['R_rel']

    cv = {}
    if scatter:
        for row in plan.itertuples():
            f = files[row.file]
            n_loc = f.spectra.shape[1] if row.use_columns is None else len(row.use_columns)
            if n_loc < 2:
                continue
            windows = line_windows(wl, spectra[row.sample][ref], lines[row.sample])
            per_loc = per_column_areas(f, lines[row.sample], windows, columns=row.use_columns,
                                       method=baseline, exclude=masks[row.sample])
            c = 100 * per_loc.std(axis=1) / per_loc.mean(axis=1)
            line_cv = c[list(lines[row.sample])]
            cv[(row.sample, row.temperature)] = {'locations': n_loc,
                                                 'line CV (%), mean': line_cv.mean() if len(line_cv) else np.nan,
                                                 'line CV (%), max': line_cv.max() if len(line_cv) else np.nan,
                                                 'Total CV (%)': c[TOTAL]}
    scatter_df = pd.DataFrame.from_dict(cv, orient='index')
    if len(scatter_df):
        scatter_df.index.names = ['sample', 'temperature']

    settings = {'libs_viewer': __version__, 'created': datetime.now().isoformat(timespec='seconds'),
                'folder': str(folder.resolve()), 'pattern': pattern,
                'manifest': str(manifest) if isinstance(manifest, (str, Path)) else
                            ('DataFrame' if manifest is not None else None),
                'temperatures': list(temperatures) if temperatures else None,
                'elements': elements, 'baseline': baseline, 'ref': ref, 'n_lines': n_lines,
                'min_prominence': min_prominence, 'interferents': interferents,
                'key_settings': list(KEY_SETTINGS), 'nist_db': str(db)}

    def stack(d):
        return pd.concat(d, names=['sample', 'row'])

    files_used = {k: files[k] for k in plan.file}
    return StudyResult(settings=settings, plan=plan, overview=overview_table(files_used),
                       line_tables=line_tables, lines=lines, areas=stack(areas),
                       r_direct=stack(r_direct), r_rel=stack(r_rel), scatter=scatter_df,
                       notes=notes, wl=wl, spectra=spectra, masks=masks)
