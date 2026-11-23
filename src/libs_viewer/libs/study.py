'''
One-call temperature study: folder in, tables out.

    result = run_study('data/end_to_end', elements=['Ce', 'Gd'], baseline='snip',
                       pattern='{element}/{sample}/*{temperature}*', temperatures=[300, 350, 400])
    result.r_rel          # line ratios vs the reference temperature
    result.save('output/run1')

`run_study` runs these stages, each a function below:
    1. study_plan        which file is which (pattern or manifest; no guessing)
    2. _load_files       read the files; per sample, keep only the stored spectra acquired with
                         the sample's gate delay / energy (dropped spectra go in `notes`)
    3. _build_samples    per-shot mean spectrum per temperature + saturated pixels, per sample
    4. _pick_lines       NIST lines per element, other study elements as interferents
    5. _measure          line / total areas with the chosen baseline, ratio tables
    6. _scatter          location-to-location and shot-to-shot CV

Everything it decided is recorded in `StudyResult.settings` and `StudyResult.notes`, and
`save()` writes both next to the tables. No plotting here (see plotting.plot_study).
'''

import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import __version__
from ..nist.query import lines_near, resolve_db
from .analysis import (TOTAL_EXCL, RatioWarning, analyze_sample, areas_table, line_windows,
                       ratio_tables)
from .lines import select_lines
from .loading import LayoutWarning, LIBSFile, SettingsWarning, overview_table
from .plan import study_plan
from .processing import (baseline_settings, location_spectra, per_shot_columns, per_shot_spectrum,
                         saturated_mask)

# Settings that must match for two spectra of one sample to be comparable.
KEY_SETTINGS = ('Gate Delay (usec)', 'Energy')

# progress(done, total, message): called before each step of `run_study`; done counts the
# steps finished so far, and a last call has done == total.
Progress = Callable[[int, int, str], None]


def _no_progress(message: str) -> None:
    pass


def _counter(progress: Progress | None, total: int) -> Callable[[str], None]:
    '''A step(message) function that reports to `progress` and counts the steps.'''
    if progress is None:
        return _no_progress
    done = 0

    def step(message: str) -> None:
        nonlocal done
        progress(min(done, total), total, message)
        done += 1
    return step


@dataclass
class Sample:
    '''One sample of a study: the spectra used at each temperature and what was measured on them.'''
    name: str
    element: str
    files: dict[float, LIBSFile]          # {temperature: file, restricted to the spectra used}
    spectra: dict[float, np.ndarray]      # {temperature: mean per-shot spectrum}
    mask: np.ndarray                      # saturated pixels at any temperature
    lines: dict[str, float] = field(default_factory=dict)   # {label: nm}
    areas: pd.DataFrame | None = None
    windows: dict[str, tuple[int, int]] | None = None   # {label: (lo, hi)}


@dataclass
class StudyResult:
    '''
    Everything one `run_study` call produced.

    Tables (rows indexed by sample, then line / total):
        areas      raw areas, columns = temperatures
        r_direct   A(T) / A(ref)
        r_rel      [A_line(T)/A_total(T)] / [A_line(ref)/A_total(ref)], A_total = settings['total_row']
                   ('Total (excl. sat.)' by default); Total rows = R_total
        scatter    per (sample, temperature): location-to-location CV (needs 2+ locations) and
                   shot-to-shot CV (needs 2+ single-shot spectra at a location)
    Also:
        plan       which file is which, with the stored spectra actually used
        overview   how every file was acquired (spectra used only)
        line_tables {element: select_lines table, or the chosen lines (status 'chosen by user')}
        notes      what was dropped, adjusted or could not be measured
        settings   the inputs and versions (provenance); saved as settings.json
        wl, sample_data {sample: Sample}; shortcuts: lines, spectra, masks
    '''
    settings: dict
    plan: pd.DataFrame
    overview: pd.DataFrame
    line_tables: dict[str, pd.DataFrame]
    areas: pd.DataFrame
    r_direct: pd.DataFrame
    r_rel: pd.DataFrame
    scatter: pd.DataFrame
    notes: list[str]
    wl: np.ndarray = field(repr=False)
    sample_data: dict[str, Sample] = field(repr=False)

    @property
    def samples(self) -> list[str]:
        return list(self.sample_data)

    @property
    def lines(self) -> dict[str, dict[str, float]]:
        return {s: d.lines for s, d in self.sample_data.items()}

    @property
    def spectra(self) -> dict[str, dict[float, np.ndarray]]:
        return {s: d.spectra for s, d in self.sample_data.items()}

    @property
    def masks(self) -> dict[str, np.ndarray]:
        return {s: d.mask for s, d in self.sample_data.items()}

    def summary(self) -> str:
        '''Plain-text summary: inputs, picked lines, R_rel table and notes.'''
        s = self.settings
        out = [f"LIBS temperature study: {s['folder']}",
               f"elements {', '.join(s['elements'])} | baseline {_baseline_text(s)} | "
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
        tables.update({f'lines_{el}.csv': t.assign(detected_in=t.detected_in.str.join('; ')).set_index('label')
                       for el, t in self.line_tables.items()})
        written = []
        for name, df in tables.items():
            df.to_csv(out / name)
            written.append(out / name)
        meta = {**self.settings, 'lines': self.lines, 'notes': self.notes}
        (out / 'settings.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
        (out / 'summary.txt').write_text(self.summary() + '\n', encoding='utf-8')
        return written + [out / 'settings.json', out / 'summary.txt']


def run_study(folder: str | Path, elements: list[str] | None = None, baseline: str = 'snip',
              baseline_params: dict | None = None, lines: dict | None = None,
              pattern: str | None = None, manifest=None, temperatures=None, ref: float = 300,
              n_lines: int = 4, min_prominence: float = 0.25, interferents: list[str] | None = None,
              total_row: str = TOTAL_EXCL, scatter: bool = True,
              db_path: str | Path | None = None, progress: Progress | None = None) -> StudyResult:
    '''
    Runs a complete temperature study on a folder of Applied Spectra .tdms files.

    Args:
        folder: the study folder.
        elements: elements to analyse (default: every element in the plan).
        baseline: continuum removal for the total areas: a name in `processing.BASELINES`,
            'snip' (default) or 'airpls', or one added with `register_baseline`.
        baseline_params: that method's parameters, e.g. {'max_half_window': 60} for SNIP or
            {'lam': 1e7} for airPLS; left-out ones take their defaults. The full set used is
            recorded in `settings['baseline_params']`.
        pattern / manifest / temperatures: how files map to element, sample and temperature;
            exactly one of pattern or manifest (see `study_plan`).
        ref: reference temperature; every sample needs a file at it.
        lines: lines to measure instead of picking them, per element, e.g. from Inspect:
            {'Ce': [418.66, 399.92]} or {'Ce': {'Ce II 418.66': 418.6596}}. Elements left
            out are picked automatically. Chosen lines are measured in every sample of the
            element; a list is labelled from the NIST line within 0.02 nm, if any.
        n_lines: lines to pick per element.
        min_prominence: line detection threshold as a multiple of the spectrum's median
            (0.25 finds weak lines on a high continuum; select_lines' stricter default is 2).
        interferents: extra elements whose lines must be avoided (the other study elements
            are always included).
        total_row: the total used for R_rel, the R_total curve and the Total scatter.
            'Total (excl. sat.)' (default) leaves the saturated pixels out; 'Total' keeps
            them, for comparison only (a note is added when they hold much of the light).
        scatter: also compute the location / shot scatter (slower with airpls).
        db_path: NIST database (default: the per-user database).
        progress: optional progress(done, total, message), called before each step: every
            file read, every element's line selection, every sample measured and every file's
            scatter. A last call has done == total. E.g. to drive a progress bar in a UI:
            `progress=lambda done, total, msg: bar.progress(done / total, text=msg)`.

    Returns:
        StudyResult. Raises ValueError / FileNotFoundError for an ambiguous or incomplete study.
    '''
    baseline_params = baseline_settings(baseline, baseline_params)   # fail early; full set for provenance
    
    folder = Path(folder)
    db = resolve_db(db_path)                       # fail early if the NIST database is missing
    plan = _restrict(study_plan(folder, pattern=pattern, manifest=manifest, temperatures=temperatures),
                     elements, ref)
    elements = list(dict.fromkeys(plan.element))
    notes: list[str] = []
    n_files, n_samples = len(plan), plan['sample'].nunique()
    total = n_files + len(elements) + n_samples + (n_files if scatter else 0)
    step = _counter(progress, total)

    files, plan['use_columns'] = _load_files(folder, plan, notes, step)
    wl = next(iter(files.values())).wavelengths
    if not all(np.array_equal(f.wavelengths, wl) for f in files.values()):
        raise ValueError('files have different wavelength axes; analyse them separately')

    samples = _build_samples(plan, files, notes)
    chosen = _chosen_lines(lines, elements, wl, db)
    line_tables = _pick_lines(samples, wl, ref, n_lines, min_prominence, interferents, db, notes,
                              step, chosen)
    areas, r_direct, r_rel = _measure(samples, wl, baseline, baseline_params, ref, total_row,
                                      notes, step)
    scatter_df = (_scatter(samples, wl, baseline, baseline_params, ref, total_row, step) if scatter
                  else pd.DataFrame())
    if progress is not None:
        progress(total, total, 'Done')

    settings = {'libs_viewer': __version__, 'created': datetime.now().isoformat(timespec='seconds'),
                'folder': str(folder.resolve()), 'pattern': pattern,
                'manifest': str(manifest) if isinstance(manifest, (str, Path)) else
                            ('DataFrame' if manifest is not None else None),
                'temperatures': list(temperatures) if temperatures else None,
                'elements': elements, 'baseline': baseline, 'baseline_params': baseline_params,
                'chosen_lines': chosen or None, 'ref': ref, 'n_lines': n_lines,
                'min_prominence': min_prominence, 'interferents': interferents, 'total_row': total_row,
                'key_settings': list(KEY_SETTINGS), 'nist_db': str(db)}
    
    return StudyResult(settings=settings, plan=plan, overview=overview_table(files),
                       line_tables=line_tables, areas=areas, r_direct=r_direct, r_rel=r_rel,
                       scatter=scatter_df, notes=notes, wl=wl, sample_data=samples)


def _baseline_text(settings: dict) -> str:
    '''e.g. "snip (max_half_window=40)".'''
    params = settings.get('baseline_params') or {}
    return settings['baseline'] + (f" ({', '.join(f'{k}={v:g}' for k, v in params.items())})"
                                   if params else '')


# ------------------------------------------------------------------ 1. plan
def _restrict(plan: pd.DataFrame, elements, ref) -> pd.DataFrame:
    '''Keeps the requested elements and checks every sample has a file at `ref`.'''
    if elements:
        absent = sorted(set(elements) - set(plan.element))
        if absent:
            raise ValueError(f'no files for element(s) {absent}; found {sorted(set(plan.element))}')
        plan = plan[plan.element.isin(elements)].reset_index(drop=True)
    no_ref = [s for s, g in plan.groupby('sample') if ref not in set(g.temperature)]
    if no_ref:
        raise ValueError(f'sample(s) without a file at the reference temperature {ref}: {no_ref}')
    return plan


# ------------------------------------------------------------------ 2. files
def _settings(f: LIBSFile) -> list[tuple]:
    '''KEY_SETTINGS of every stored spectrum, as ((name, value), ...) tuples.'''
    keys = [k for k in KEY_SETTINGS if k in f.acquisitions.columns]
    return [tuple(zip(keys, map(float, row))) for row in f.acquisitions[keys].itertuples(index=False)]


def _fmt(setting: tuple) -> str:
    return ', '.join(f'{k} {v:g}' for k, v in setting)


def _load_files(folder: Path, plan: pd.DataFrame, notes: list[str],
                step=_no_progress) -> tuple[dict[str, LIBSFile], list]:
    '''
    Reads every file of the plan, restricted to the spectra to use: the manifest's
    use_columns if given, otherwise those acquired with the sample's usual setting (the
    KEY_SETTINGS combination found in the most files of that sample).

    Returns:
        ({file: LIBSFile with only the spectra used}, use_columns per plan row, None = all).
    '''
    raw = {}

    for i, k in enumerate(plan.file, 1):
        step(f'Reading file {i} of {len(plan)}: {k}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            raw[k] = LIBSFile.from_tdms(folder / k)
        for w in caught:
            if issubclass(w.category, LayoutWarning):
                notes.append(str(w.message))             # e.g. shots of one location at two positions
            elif not issubclass(w.category, SettingsWarning):   # mixed settings are resolved below
                warnings.warn(w.message, w.category, stacklevel=2)

    files = {}

    for sample, g in plan.groupby('sample', sort=False):
        votes = Counter(s for k in g.file for s in set(_settings(raw[k])))
        (usual, n), *others = votes.most_common()
        if others and others[0][1] == n:
            raise ValueError(f'{sample}: no acquisition setting is clearly the most common '
                             f'({ {_fmt(s): c for s, c in votes.items()} }); set use_columns in a manifest')
        for key, use in zip(g.file, g.use_columns):
            f = raw[key]
            if use is not None:
                files[key] = f.subset(columns=use)
                continue
            keep = np.array([s == usual for s in _settings(f)])
            if keep.all():
                files[key] = f
            elif not keep.any():
                files[key] = f
                notes.append(f'{key}: acquired with {_fmt(_settings(f)[0])}, unlike the rest of '
                             f'{sample} ({_fmt(usual)}); kept, but compare with care')
            else:
                files[key] = f.subset(columns=f.spectra.columns[keep])
                dropped = f.subset(columns=f.spectra.columns[~keep])
                notes.append(f'{key}: used {files[key].describe()}; dropped {dropped.describe()} '
                             f'acquired with {_fmt(_settings(dropped)[0])} instead of {_fmt(usual)}')
                
    use = [None if files[k].spectra.shape[1] == raw[k].spectra.shape[1] else list(files[k].spectra.columns)
           for k in plan.file]
    
    return {k: files[k] for k in plan.file}, use


# ------------------------------------------------------------------ 3. samples
def _build_samples(plan: pd.DataFrame, files: dict[str, LIBSFile], notes: list[str]) -> dict[str, Sample]:
    samples: dict[str, Sample] = {}

    for key, g in plan.groupby('sample', sort=False):
        name = str(key)                     # groupby keys are typed as any scalar; samples are str
        by_t = {t: files[k] for k, t in zip(g.file, g.temperature)}
        how = {t: f.describe() for t, f in by_t.items()}
        
        if len(set(how.values())) > 1:
            notes.append(f'{name}: acquired differently across temperatures '
                         f"({'; '.join(f'{t}: {d}' for t, d in how.items())}); per-shot averages "
                         'still compare, scatter values do not')
            
        samples[name] = Sample(name=name, element=str(g.element.iloc[0]), files=by_t,
                               spectra={t: per_shot_spectrum(f) for t, f in by_t.items()},
                               mask=saturated_mask(list(by_t.values())))
        
    return samples


# ------------------------------------------------------------------ 4. lines
def _chosen_lines(lines: dict | None, elements: list[str], wl, db) -> dict[str, dict[str, float]]:
    '''`run_study(lines=...)` as {element: {label: nm}}, checked against the study.'''
    out = {}
    for el, given in (lines or {}).items():
        if el not in elements:
            raise ValueError(f'lines given for {el}, which is not in the study {elements}')
        if isinstance(given, dict):
            chosen = {str(label): float(nm) for label, nm in given.items()}
        else:
            nms = [float(nm) for nm in given]
            near = lines_near(nms, tol=0.02, elements=[el], db_path=db).drop_duplicates('peak_nm')
            names = dict(zip(near.peak_nm, near.species))
            chosen = {f'{names.get(nm, el)} {nm:.2f}': nm for nm in nms}
        if not chosen:
            raise ValueError(f'no lines given for {el}')
        outside = [nm for nm in chosen.values() if not wl.min() <= nm <= wl.max()]
        if outside:
            raise ValueError(f'{el} line(s) {outside} are outside the measured range '
                             f'{wl.min():.1f}-{wl.max():.1f} nm')
        out[el] = chosen
    return out


def _pick_lines(samples: dict[str, Sample], wl, ref, n_lines, min_prominence, interferents, db,
                notes: list[str], step=_no_progress,
                chosen: dict[str, dict[str, float]] | None = None) -> dict[str, pd.DataFrame]:
    '''
    select_lines per element; each sample keeps the picked lines detected in it. Elements in
    `chosen` skip the selection: every sample gets exactly those lines.
    '''
    elements = list(dict.fromkeys(s.element for s in samples.values()))
    tables = {}
    for el in elements:
        group = [s for s in samples.values() if s.element == el]
        if chosen and el in chosen:
            step(f'Using the chosen {el} lines')
            tables[el] = pd.DataFrame({
                'species': [label.rsplit(' ', 1)[0] for label in chosen[el]],
                'intensity': np.nan, 'wavelength_nm': list(chosen[el].values()),
                'status': 'chosen by user', 'n_detected': np.nan,
                'detected_in': [[] for _ in chosen[el]], 'label': list(chosen[el])})
            for s in group:
                s.lines = dict(chosen[el])
            continue
        step(f'Choosing {el} lines from the NIST database')
        others = [e for e in dict.fromkeys(elements + list(interferents or [])) if e != el]
        table = select_lines(el, wl, {s.name: s.spectra[ref] for s in group}, n=n_lines,
                             interferents=others or None,
                             exclude=np.logical_or.reduce([s.mask for s in group]),
                             min_prominence=min_prominence, db_path=db)
        tables[el] = table
        picked = table[table.status == 'picked']
        # (label, nm, samples it was detected in) for every picked line
        picked_lines = list(zip(picked['label'].astype(str), picked['wavelength_nm'].astype(float),
                                picked['detected_in']))
        for s in group:
            s.lines = {label: nm for label, nm, found in picked_lines if s.name in found}
            if not s.lines:
                notes.append(f'{s.name}: no {el} line detected at {ref} C; only totals are reported')
    return tables


# ------------------------------------------------------------------ 5. areas and ratios
def _measure(samples: dict[str, Sample], wl, baseline, baseline_params, ref, total_row,
             notes: list[str], step=_no_progress) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''analyze_sample + ratio_tables per sample; ratio warnings go into `notes`.'''
    areas: dict[str, pd.DataFrame] = {}
    r_direct: dict[str, pd.DataFrame] = {}
    r_rel: dict[str, pd.DataFrame] = {}

    for s in samples.values():
        step(f'Measuring line and total areas: {s.name}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', RatioWarning)
            a = analyze_sample(wl, s.spectra, s.lines, method=baseline,
                               baseline_params=baseline_params, exclude=s.mask, ref=ref)
            r = ratio_tables(a, ref=ref, total_row=total_row)
        s.areas = a

        for w in caught:
            if issubclass(w.category, RatioWarning):
                notes.append(f'{s.name}: {w.message}')
            else:
                warnings.warn(w.message, w.category, stacklevel=2)

        areas[s.name], r_direct[s.name], r_rel[s.name] = a, r['R_direct'], r['R_rel']
        s.windows = line_windows(wl, s.spectra[ref], s.lines)

    def stack(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(d, names=['sample', 'row'])
    
    return stack(areas), stack(r_direct), stack(r_rel)


# ------------------------------------------------------------------ 6. scatter
def _cv(table: pd.DataFrame) -> pd.Series:
    '''Coefficient of variation (%) of every row across the columns.'''
    return 100 * table.std(axis=1) / table.mean(axis=1)


def _scatter(samples: dict[str, Sample], wl, baseline, baseline_params, ref,
             total_row=TOTAL_EXCL, step=_no_progress) -> pd.DataFrame:
    '''
    Per (sample, temperature):
        location CV   spread of the per-location mean spectra (2+ locations)
        shot CV       spread of single-shot spectra within a location, averaged over the
                      locations with 2+ single shots
    Each for the lines (mean over the sample's lines) and for the Total area ('Total' here
    is `total_row`, the same total R_rel uses).
    '''
    rows = {}

    for s in samples.values():
        lines = list(s.lines)
        windows = s.windows if s.windows is not None else line_windows(wl, s.spectra[ref], s.lines)

        for t, f in s.files.items():
            step(f'Location / shot scatter: {s.name} at {t:g} C')
            row = {}

            locs = location_spectra(f)
            if locs.shape[1] > 1:
                cv = _cv(areas_table(wl, locs, s.lines, windows, method=baseline,
                                     baseline_params=baseline_params, exclude=s.mask))
                row['location CV, lines (%)'] = cv[lines].mean() if lines else np.nan
                row['location CV, Total (%)'] = cv[total_row]

            single = f.acquisitions[f.acquisitions['shots_summed'] == 1]
            per_loc = single.groupby('location').size()
            if (per_loc > 1).any():
                shots = per_shot_columns(f)[single.index]
                a = areas_table(wl, shots, s.lines, windows, method=baseline,
                                baseline_params=baseline_params, exclude=s.mask)
                g = a.T.groupby(single['location'])
                cv = (100 * g.std() / g.mean())[per_loc > 1].mean()
                row['shot CV, lines (%)'] = cv[lines].mean() if lines else np.nan
                row['shot CV, Total (%)'] = cv[total_row]

            if row:
                rows[(s.name, t)] = {'locations': f.n_locations, 'spectra': f.spectra.shape[1], **row}

    cols = ['locations', 'spectra', 'location CV, lines (%)', 'location CV, Total (%)',
            'shot CV, lines (%)', 'shot CV, Total (%)']
    df = pd.DataFrame.from_dict(rows, orient='index').reindex(columns=cols)
    if len(df):
        df.index.names = ['sample', 'temperature']
    return df
