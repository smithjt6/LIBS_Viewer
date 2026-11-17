"""
NIST Handbook of Basic Atomic Spectroscopic Data -> Local SQLite Builder
Crawls Tables 1-7 for elements Z=1..99 while strictly respecting physics.nist.gov/robots.txt.

USER-ONLY (tier 3): this module uses the network and writes the shared database. It is run
from the CLI (`libs-viewer nist build`) or the UI, and is never registered as an LLM tool.

By default the database and the raw-page cache go in the libs_viewer data folder
(see `libs_viewer.core.paths`). With the cache in place, a rebuild needs no network.
"""

import re
import time
import html as html_lib
import sqlite3
import argparse
import urllib.robotparser
from pathlib import Path
from typing import Optional, List, Dict
import requests

from ..core.paths import nist_db_path, nist_cache_dir

BASE_URL = "https://physics.nist.gov/PhysRefData/Handbook/Tables"
ROBOTS_URL = "https://physics.nist.gov/robots.txt"
USER_AGENT = "NIST-Handbook-SQLite-Cacher/1.0 (Academic Spectroscopy Research)"

# All 99 elements included in the NIST Handbook of Basic Atomic Spectroscopic Data
NIST_ELEMENTS: Dict[str, str] = {
    "H": "hydrogen", "He": "helium", "Li": "lithium", "Be": "beryllium", "B": "boron",
    "C": "carbon", "N": "nitrogen", "O": "oxygen", "F": "fluorine", "Ne": "neon",
    "Na": "sodium", "Mg": "magnesium", "Al": "aluminum", "Si": "silicon", "P": "phosphorus",
    "S": "sulfur", "Cl": "chlorine", "Ar": "argon", "K": "potassium", "Ca": "calcium",
    "Sc": "scandium", "Ti": "titanium", "V": "vanadium", "Cr": "chromium", "Mn": "manganese",
    "Fe": "iron", "Co": "cobalt", "Ni": "nickel", "Cu": "copper", "Zn": "zinc",
    "Ga": "gallium", "Ge": "germanium", "As": "arsenic", "Se": "selenium", "Br": "bromine",
    "Kr": "krypton", "Rb": "rubidium", "Sr": "strontium", "Y": "yttrium", "Zr": "zirconium",
    "Nb": "niobium", "Mo": "molybdenum", "Tc": "technetium", "Ru": "ruthenium", "Rh": "rhodium",
    "Pd": "palladium", "Ag": "silver", "Cd": "cadmium", "In": "indium", "Sn": "tin",
    "Sb": "antimony", "Te": "tellurium", "I": "iodine", "Xe": "xenon", "Cs": "cesium",
    "Ba": "barium", "La": "lanthanum", "Ce": "cerium", "Pr": "praseodymium", "Nd": "neodymium",
    "Pm": "promethium", "Sm": "samarium", "Eu": "europium", "Gd": "gadolinium", "Tb": "terbium",
    "Dy": "dysprosium", "Ho": "holmium", "Er": "erbium", "Tm": "thulium", "Yb": "ytterbium",
    "Lu": "lutetium", "Hf": "hafnium", "Ta": "tantalum", "W": "tungsten", "Re": "rhenium",
    "Os": "osmium", "Ir": "iridium", "Pt": "platinum", "Au": "gold", "Hg": "mercury",
    "Tl": "thallium", "Pb": "lead", "Bi": "bismuth", "Po": "polonium", "At": "astatine",
    "Rn": "radon", "Fr": "francium", "Ra": "radium", "Ac": "actinium", "Th": "thorium",
    "Pa": "protactinium", "U": "uranium", "Np": "neptunium", "Pu": "plutonium", "Am": "americium",
    "Cm": "curium", "Bk": "berkelium", "Cf": "californium", "Es": "einsteinium"
}


# =====================================================================
# 1. TEXT & QUANTUM NUMBER CLEANING HELPERS
# =====================================================================

def j_to_g(j_str: str) -> Optional[int]:
    """Converts a J quantum number string ('7/2', '4') into statistical weight g = 2J + 1."""
    j_str = j_str.strip()
    if not j_str or "," in j_str or "?" in j_str:
        return None
    try:
        if "/" in j_str:
            num, den = j_str.split("/", 1)
            j_val = float(num) / float(den)
        else:
            j_val = float(j_str)
        return int(round(2.0 * j_val + 1.0))
    except ValueError:
        return None


def clean_float(val: str) -> Optional[float]:
    """Strips NIST uncertainty brackets/flags ('[', ']', '?', '+x', 'a') and converts to float."""
    if not val:
        return None
    # Remove parenthetical uncertainties at the end like 140.116(1) -> 140.116
    val_no_paren = re.sub(r"\(\d+\)", "", val.strip())
    cleaned = re.sub(r"[^\d.\-]", "", val_no_paren)
    try:
        return float(cleaned) if cleaned and cleaned not in ("-", ".", "-.") else None
    except ValueError:
        return None


def strip_html_tags(text: str) -> str:
    """Strips HTML tags, converts entities (&deg;, &nbsp;), and collapses whitespace."""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    text = text.replace("\xa0", " ").replace("°", "°")
    return re.sub(r"\s+", " ", text).strip()


def extract_pre_lines(html_text: str) -> List[str]:
    """Extracts plain text lines inside the <pre>...</pre> block of a NIST _a.htm ASCII page."""
    m = re.search(r"<pre>(.*?)</pre>", html_text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    text = re.sub(r"<[^>]+>", "", m.group(1))
    text = html_lib.unescape(text).replace("\xa0", " ")
    return text.splitlines()


# =====================================================================
# 2. TABLE PARSERS (TABLES 1 TO 7)
# =====================================================================

def parse_table1_atomic_data(symbol: str, html_text: str) -> List[Dict]:
    """
    Parses {element}table1_a.htm (ASCII) for Atomic Number, Atomic Weight, and the
    ground state / ionization energy of each spectrum (I, II). The page gives these as
    free text, not a table:

        <h3>Atomic Number = 26 </h3>
        <h3>Atomic Weight = 55.847 </h3>
        Fe I  Ground State 1<i>s</i><Sup>2</sup>...4<i>s</i><Sup>2</sup>  <Sup>5</sup>D<sub>4</sub>
              Ionization energy  63737 cm<sup>-1</sup> (7.9024 eV)   Ref. <a ...>SC85</a>

    The term follows the configuration after a run of 2+ spaces, and J is the trailing
    run of <sub> tags, so both are split off before the tags are stripped.
    """
    z_match = re.search(r"Atomic Number\s*=\s*(\d+)", html_text, re.IGNORECASE)
    wt_match = re.search(r"Atomic Weight\s*=\s*([^<\s]+)", html_text, re.IGNORECASE)
    atomic_number = int(z_match.group(1)) if z_match else None
    atomic_weight = None
    if wt_match:
        # '(254)' is the mass number of the longest-lived isotope, not an uncertainty
        mass_number = re.fullmatch(r"[(\[](\d+)[)\]]", wt_match.group(1))
        atomic_weight = float(mass_number.group(1)) if mass_number else clean_float(wt_match.group(1))

    records = []
    lines = html_text.splitlines()
    for n, line in enumerate(lines):
        m = re.match(rf"^\s*({re.escape(symbol)} [IVX]+)\s+Ground State\s+(.*?)\s*$", line)
        if not m:
            continue
        species, state = m.groups()
        parts = re.split(r"\s{2,}", state)
        config_html = " ".join(parts[:-1]) if len(parts) > 1 else parts[0]
        term_html = parts[-1] if len(parts) > 1 else ""
        j_match = re.search(r"((?:<sub>[^<]*</sub>)+)\s*$", term_html, re.IGNORECASE)
        ground_j = strip_html_tags(j_match.group(1)).replace(" ", "") if j_match else ""
        ground_term = strip_html_tags(term_html[:j_match.start()] if j_match else term_html)

        # The ionization energy and its reference are on the following line
        ion_cm1 = ion_ev = None
        ref = ""
        if n + 1 < len(lines):
            nxt = strip_html_tags(lines[n + 1])
            ion_match = re.search(r"Ionization energy\s+(\S+)\s*cm-1\s*\(\s*(\S+)\s*eV\)", nxt)
            ref_match = re.search(r"Ref\.\s*(\S+)", nxt)
            if ion_match:
                ion_cm1, ion_ev = clean_float(ion_match.group(1)), clean_float(ion_match.group(2))
            if ref_match:
                ref = ref_match.group(1)

        records.append({
            "species": species,
            "element": symbol,
            "ion_stage": species.split()[1],
            "atomic_number": atomic_number,
            "atomic_weight": atomic_weight,
            "ground_config": strip_html_tags(config_html),
            "ground_term": ground_term,
            "ground_j": ground_j,
            "ground_g": j_to_g(ground_j),
            "ionization_energy_eV": ion_ev,
            "ionization_energy_cm1": ion_cm1,
            "ref": ref
        })
    return records


def vacuum_to_air(wl_vac_A: float) -> Optional[float]:
    """
    Vacuum -> standard air wavelength (Angstrom), Morton (2000, ApJS 130, 403) / IAU formula.
    Returns None below 1700 A, where the dispersion formula diverges (and air is opaque anyway).
    """
    if wl_vac_A < 1700.0:
        return None
    return wl_vac_A / _air_index(wl_vac_A)


def _air_index(wl_vac_A: float) -> float:
    """Refractive index of standard air at vacuum wavelength wl_vac_A (Morton 2000); valid above ~1700 A."""
    s2 = (1e4 / wl_vac_A) ** 2
    return 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)


def resolve_medium(wl_A: float, section: Optional[str],
                   ei_cm1: Optional[float] = None, ek_cm1: Optional[float] = None) -> str:
    """
    The page's Air/Vacuum section if one was seen, else the Handbook rule, which is applied to
    the *vacuum* wavelength: vacuum below 2000 A, air above. A published value in
    [1999.35, 2000) A can therefore be either (e.g. Cu II 1999.698 A is air = 2000.346 A vacuum);
    there the level energies decide, as Ek - Ei must equal 1 / lambda_vac.
    """
    if section:
        return section
    if wl_A >= 2000.0:
        return "air"
    if ei_cm1 is not None and ek_cm1 is not None and wl_A >= 2000.0 / _air_index(2000.0):
        sigma = abs(ek_cm1 - ei_cm1)
        air_as_vac = wl_A * _air_index(wl_A)  # air -> vacuum, to well under 0.001 A here
        if abs(sigma - 1e8 / air_as_vac) < abs(sigma - 1e8 / wl_A):
            return "air"
    return "vacuum"


def air_wavelength(wl_A: float, medium: str) -> Optional[float]:
    """Air wavelength for a line published in `medium` ('air' or 'vacuum')."""
    return wl_A if medium == "air" else vacuum_to_air(wl_A)


def _medium_marker(line: str) -> Optional[str]:
    """Tables 2-4 split lines into sections headed by a bare 'Air' or 'Vacuum' line."""
    s = line.strip().lower()
    return s if s in ("air", "vacuum") else None


def parse_table2_strong_lines(symbol: str, html_text: str) -> List[Dict]:
    """
    Parses {element}table2_a.htm (ASCII) Strong Lines across neutral and ionized states.
    Wavelengths are in Angstrom, in air or vacuum according to the section they sit in.
    """
    records = []
    medium = None
    for line in extract_pre_lines(html_text):
        medium = _medium_marker(line) or medium
        # Format: [Intensity + Flags] | Wavelength | Spectrum (e.g., Ce I, Ce II) | Ref
        # (a few lines have no intensity at all)
        m = re.match(
            r"^(?P<pre>[^.]*?)\s(?P<wl>\d+\.\d+)\s+(?P<sp>[A-Z][a-z]?(?:\s+[IVX]+)?)(?:\s+(?P<ref>\S+))?\s*$",
            line
        )
        if m:
            intens_num = re.match(r"\s*(\d+)", m.group("pre"))
            clean_flags = re.sub(r"[\d\s]+", "", m.group("pre"))
            wl = float(m.group("wl"))
            line_medium = resolve_medium(wl, medium)
            records.append({
                "element": symbol,
                "species": re.sub(r"\s+", " ", m.group("sp")),
                "wavelength_A": wl,
                "medium": line_medium,
                "wavelength_air_A": air_wavelength(wl, line_medium),
                "intensity": int(intens_num.group(1)) if intens_num else None,
                "flags": clean_flags,
                "is_persistent": 1 if "P" in clean_flags else 0,
                "ref": m.group("ref") or ""
            })
    return records


_NUMBER = re.compile(r"^[\[(]?\d+(?:\.\d*)?[\])]?[?*]?$")
_J_VALUE = re.compile(r"^\d+(?:/\d+)?(?:,\d+(?:/\d+)?)*\??$")


def _level_columns(lines: List[str]) -> Optional[Dict[str, int]]:
    """
    Column starts of the Configuration, Term and J fields (and Aki, on Tables 3-4), read
    from the page's own header. Column widths differ from element to element, but within a
    page the stripped <pre> text is aligned as the browser renders it, so the header
    positions hold to within a character or so.
    """
    for line in lines:
        if "Configuration" in line and "Term" in line:
            term = line.index("Term")
            j = re.search(r"\sJ(\s|$)", line[term:])
            if j:
                aki = re.search(r"\bAki\b", line)
                return {"config": line.index("Configuration"), "term": term, "j": term + j.start() + 1,
                        "aki": aki.start() if aki else -1}
    return None


def _split_level(line: str, cols: Dict[str, int]) -> Dict:
    """
    Splits one level row into its numeric left part (str), config / term / J (str), and the
    whitespace-separated fields after J (list: energy and/or references).

    From the Term column on, each word is placed by where it starts: words left of the J
    column make up the term ('a 5D', '(1/2,1/2)°', or a bare level number like '7'), and
    the first J-like word at or after it is J. The header's J can sit a character off from
    the data, so a J-like word that follows the term after a 2+ space gap also counts as J.
    """
    padded = line.ljust(cols["term"])
    term_parts: List[str] = []
    j = ""
    rest: List[str] = []
    prev_end = 0
    for m in re.finditer(r"\S+", padded[cols["term"]:]):
        word = m.group()
        gap = m.start() - prev_end
        prev_end = m.end()
        in_term_col = cols["term"] + m.start() < cols["j"] - 1
        if j or rest:
            rest.append(word)
        elif in_term_col and not (term_parts and gap >= 2 and _J_VALUE.match(word)):
            term_parts.append(word)
        elif _J_VALUE.match(word):
            j = word
        else:
            rest.append(word)
    return {
        "left": padded[:cols["config"]],
        "config": padded[cols["config"]:cols["term"]].strip(),
        "term": " ".join(term_parts),
        "j": j,
        "rest": rest,
    }


def _numeric_tokens(text: str) -> List[re.Match]:
    """Whitespace tokens with positions; a lone '?' (uncertain value) is merged into the token before it."""
    tokens: List[re.Match] = []
    for m in re.finditer(r"\S+(?: \?(?=\s|$))?", text):
        tokens.append(m)
    return tokens


def parse_table3_4_persistent(symbol: str, ion_stage: str, html_text: str) -> List[Dict]:
    """
    Parses {element}table3_a.htm (I) and table4_a.htm (II) (ASCII).
    Pairs the 2-row Lower Level (i) and Upper Level (k) entries into flat records:

         1000    2483.2708    4.81        0.000      3d64s2             a 5D    4   NJLT94   OWLW91
                                      40257.311      3d6(5D)4s4p(1P)    x 5F°   5

    The numeric fields (intensity, wavelength in Angstrom, Aki which may be blank, Ei / Ek)
    are whitespace-separated tokens; the text fields are cut at the header's column starts.

    A pair whose first row has Aki and Ei but no intensity or wavelength is a further
    transition blended into the line above; it is stored with that line's wavelength and
    no intensity of its own.
    """
    lines = extract_pre_lines(html_text)
    cols = _level_columns(lines)
    if cols is None:
        return []
    records = []
    medium = None
    prev_wl = None
    i = 0
    while i < len(lines):
        medium = _medium_marker(lines[i]) or medium
        lower = _split_level(lines[i], cols)
        tokens = _numeric_tokens(lower["left"])
        # The wavelength is the first decimal number that ends before the Aki column
        # (an Aki value starts at or after it). Intensity, if any, sits to its left.
        wl_idx = next((k for k, t in enumerate(tokens)
                       if re.fullmatch(r"\d+\.\d+", t.group()) and (cols["aki"] < 0 or t.end() <= cols["aki"] + 1)),
                      None)
        if wl_idx is not None:
            # A transition row: [intensity] wavelength [Aki] [Ei]
            numbers = [t.group().replace(" ", "") for t in tokens[wl_idx + 1:]]
            wl = float(tokens[wl_idx].group())
            intens_raw = "".join(t.group() for t in tokens[:wl_idx])
        elif len(tokens) == 2 and lower["config"] and prev_wl is not None:
            # Blended component: Aki Ei only, under the previous line's wavelength
            numbers, wl, intens_raw = [t.group().replace(" ", "") for t in tokens], prev_wl, ""
        else:
            i += 1
            continue
        if len(numbers) > 2 or not all(_NUMBER.match(t) for t in numbers):
            i += 1
            continue
        # Two numbers are Aki and Ei; a single one is Ei (Aki not given)
        aki_raw, ei_raw = {0: ("", ""), 1: ("", numbers[0] if numbers else ""), 2: tuple(numbers)}[len(numbers)]
        prev_wl = wl

        upper = {"left": "", "config": "", "term": "", "j": ""}
        if i + 1 < len(lines) and lines[i + 1].strip():
            cand = _split_level(lines[i + 1], cols)
            cand_left = cand["left"].replace(" ", "")
            if cand_left and _NUMBER.match(cand_left):  # continuation row holds only Ek on the left
                upper = cand
                i += 1

        ei, ek = clean_float(ei_raw), clean_float(upper["left"].replace(" ", ""))
        line_medium = resolve_medium(wl, medium, ei, ek)
        intens_num = re.match(r"\d+", intens_raw)
        records.append({
            "species": f"{symbol} {ion_stage}",
            "element": symbol,
            "ion_stage": ion_stage,
            "wavelength_A": wl,
            "medium": line_medium,
            "wavelength_air_A": air_wavelength(wl, line_medium),
            "intensity": int(intens_num.group(0)) if intens_num else None,
            "intensity_flags": re.sub(r"\d+", "", intens_raw),
            "aki_10_8_s1": clean_float(aki_raw),
            "ei_cm1": ei,
            "ek_cm1": ek,
            "config_i": lower["config"],
            "term_i": lower["term"],
            "j_i": lower["j"],
            "g_i": j_to_g(lower["j"]),
            "config_k": upper["config"],
            "term_k": upper["term"],
            "j_k": upper["j"],
            "g_k": j_to_g(upper["j"]),
            "line_ref": lower["rest"][0] if len(lower["rest"]) > 0 else "",
            "aki_ref": lower["rest"][1] if len(lower["rest"]) > 1 else ""
        })
        i += 1
    return records


def parse_table5_6_energy_levels(symbol: str, ion_stage: str, html_text: str) -> List[Dict]:
    """
    Parses {element}table5_a.htm (I) and table6_a.htm (II) (ASCII) Energy Levels.
    Forward-fills Configuration and Term across fine-structure J rows within each multiplet block:

        3d64s2            a 5D   4       0.000   NJLT94
                                 3     415.933   NJLT94
    """
    lines = extract_pre_lines(html_text)
    cols = _level_columns(lines)
    if cols is None:
        return []
    records = []
    curr_config = ""
    curr_term = ""

    for line in lines:
        if not line.strip():
            curr_config, curr_term = "", ""
            continue
        if "Configuration" in line and "Term" in line:
            continue

        # Ionization limit rows ignore the column layout:  'H            Limit       109678.7717   MK00a'
        limit = re.match(r"^\s*(.*?)\s*\bLimit\b\s*-*\s*(\S+)\s*(.*)$", line)
        if limit and clean_float(limit.group(2)) is not None:
            records.append({
                "species": f"{symbol} {ion_stage}",
                "element": symbol,
                "ion_stage": ion_stage,
                "configuration": limit.group(1),
                "term": "Limit",
                "parity": "",
                "j": "",
                "g": None,
                "energy_cm1": clean_float(limit.group(2)),
                "is_ionization_limit": 1,
                "ref": limit.group(3).strip()
            })
            curr_config, curr_term = "", ""
            continue

        row = _split_level(line, cols)
        if row["config"]:
            curr_config = row["config"]
        if row["term"]:
            curr_term = row["term"]
        if not row["rest"]:
            continue
        energy_val = clean_float(row["rest"][0])
        if energy_val is None:
            continue

        is_limit = 1 if "Limit" in curr_term else 0
        parity = "odd" if "°" in curr_term else "even"

        records.append({
            "species": f"{symbol} {ion_stage}",
            "element": symbol,
            "ion_stage": ion_stage,
            "configuration": curr_config,
            "term": curr_term,
            "parity": parity,
            "j": row["j"],
            "g": j_to_g(row["j"]),
            "energy_cm1": energy_val,
            "is_ionization_limit": is_limit,
            "ref": " ".join(row["rest"][1:])
        })
    return records


def parse_table7_references(symbol: str, html_text: str) -> List[Dict]:
    """
    Parses {element}table7.htm (HTML) to map citation codes (e.g., 'C73', 'SR65')
    to their full bibliographic reference strings.
    """
    records = []
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_text, flags=re.DOTALL | re.IGNORECASE)
    for row in rows:
        anchor_match = re.search(r'<a\s+name=["\']?([^"\'>\s]+)["\']?\s*>', row, flags=re.IGNORECASE)
        cols = [
            strip_html_tags(c)
            for c in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL | re.IGNORECASE)
        ]
        cols = [c for c in cols if c]
        if anchor_match and len(cols) >= 2:
            records.append({
                "element": symbol,
                "ref_code": anchor_match.group(1).strip(),
                "citation": cols[1]
            })
    return records


# =====================================================================
# 3. ROBOTS.TXT COMPLIANT FETCHER & SQLITE BUILDER
# =====================================================================

class PoliteNISTFetcher:
    """
    Fetches Handbook pages with a delay between requests. With `cache_dir`, each page is
    saved there as fetched and read back on later runs, so re-parsing needs no network.
    robots.txt is only requested once the first page actually has to be downloaded.
    """
    def __init__(self, user_agent: str = USER_AGENT, min_delay: float = 0.5,
                 cache_dir: Optional[str | Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self.min_delay = min_delay
        self.delay = min_delay
        self.rp = urllib.robotparser.RobotFileParser()
        self.rp.set_url(ROBOTS_URL)
        self.has_robots = False
        self._robots_checked = False

    def _load_robots(self):
        self._robots_checked = True
        min_delay = self.min_delay
        try:
            resp = self.session.get(ROBOTS_URL, timeout=5)
            if resp.status_code == 200:
                self.rp.parse(resp.text.splitlines())
                self.has_robots = True
                robots_delay = self.rp.crawl_delay("*") or 0.0
                self.delay = max(min_delay, float(robots_delay))
                print(f"[robots.txt] Loaded {ROBOTS_URL} (delay={self.delay:.2f}s)")
            else:
                print(f"[robots.txt] No robots.txt found at {ROBOTS_URL} (HTTP {resp.status_code}). Proceeding with default delay={self.delay:.2f}s")
        except Exception as e:
            print(f"[robots.txt] Note: {ROBOTS_URL} unreachable ({e}). Using delay={min_delay}s")

    def get(self, url: str) -> Optional[str]:
        cached = self.cache_dir / url.rsplit("/", 1)[-1] if self.cache_dir else None
        if cached and cached.exists():
            return cached.read_text(encoding="latin-1")
        text = self._fetch(url)
        if cached and text is not None:
            cached.write_text(text, encoding="latin-1")
        return text

    def _fetch(self, url: str) -> Optional[str]:
        if not self._robots_checked:
            self._load_robots()
        # Only block if robots.txt was successfully found and explicitly forbids it
        if self.has_robots and not self.rp.can_fetch(self.user_agent, url):
            print(f"  [BLOCKED BY ROBOTS.TXT] {url}")
            return None
        try:
            resp = self.session.get(url, timeout=15)
            time.sleep(self.delay)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 404:
                return None
            else:
                print(f"  [HTTP {resp.status_code}] {url}")
                return None
        except requests.RequestException as e:
            print(f"  [NETWORK ERROR] {url}: {e}")
            return None


def init_db(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS crawl_log (
            element TEXT PRIMARY KEY,
            crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS atomic_data (
            species TEXT PRIMARY KEY,
            element TEXT,
            ion_stage TEXT,
            atomic_number INTEGER,
            atomic_weight REAL,
            ground_config TEXT,
            ground_term TEXT,
            ground_j TEXT,
            ground_g INTEGER,
            ionization_energy_eV REAL,
            ionization_energy_cm1 REAL,
            ref TEXT
        );
        CREATE TABLE IF NOT EXISTS strong_lines (
            element TEXT,
            species TEXT,
            wavelength_A REAL,          -- as published: Angstrom, in `medium`
            medium TEXT,                -- 'air' or 'vacuum' (Handbook: vacuum below 2000 A)
            wavelength_air_A REAL,      -- standard air wavelength; NULL below 1700 A
            intensity INTEGER,
            flags TEXT,
            is_persistent INTEGER,
            ref TEXT
        );
        CREATE TABLE IF NOT EXISTS persistent_lines (
            species TEXT,
            element TEXT,
            ion_stage TEXT,
            wavelength_A REAL,
            medium TEXT,
            wavelength_air_A REAL,
            intensity INTEGER,
            intensity_flags TEXT,
            aki_10_8_s1 REAL,
            ei_cm1 REAL,
            ek_cm1 REAL,
            config_i TEXT,
            term_i TEXT,
            j_i TEXT,
            g_i INTEGER,
            config_k TEXT,
            term_k TEXT,
            j_k TEXT,
            g_k INTEGER,
            line_ref TEXT,
            aki_ref TEXT
        );
        CREATE TABLE IF NOT EXISTS energy_levels (
            species TEXT,
            element TEXT,
            ion_stage TEXT,
            configuration TEXT,
            term TEXT,
            parity TEXT,
            j TEXT,
            g INTEGER,
            energy_cm1 REAL,
            is_ionization_limit INTEGER,
            ref TEXT
        );
        CREATE TABLE IF NOT EXISTS bibliographic_refs (
            element TEXT,
            ref_code TEXT,
            citation TEXT,
            PRIMARY KEY (element, ref_code)
        );

        CREATE INDEX IF NOT EXISTS idx_pl_species_wl ON persistent_lines(species, wavelength_A);
        CREATE INDEX IF NOT EXISTS idx_pl_wl ON persistent_lines(wavelength_A);
        CREATE INDEX IF NOT EXISTS idx_sl_wl ON strong_lines(wavelength_A, element);
        CREATE INDEX IF NOT EXISTS idx_el_species_e ON energy_levels(species, energy_cm1);
    """)
    return conn


def insert_rows(cur: sqlite3.Cursor, table: str, rows: List[Dict], replace: bool = False):
    """Inserts parser records by column name, so the dict keys must match the table's columns."""
    if not rows:
        return
    cols = list(rows[0].keys())
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    cur.executemany(
        f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})",
        rows
    )


def build_database(
    db_path: Optional[str | Path] = None,
    elements: Optional[List[str]] = None,
    force_refresh: bool = False,
    cache_dir: Optional[str | Path] = None,
    use_cache: bool = True,
):
    """
    Builds (or tops up) the NIST database. USER-ONLY: uses the network and writes shared state.

    Args:
        db_path: database file (default: the libs_viewer data folder).
        elements: symbols to fetch (default: all 99).
        force_refresh: re-parse elements already in the database.
        cache_dir: folder for raw pages (default: the libs_viewer data folder).
        use_cache: False to neither read nor keep raw pages.
    """
    db_path = Path(db_path) if db_path else nist_db_path()
    if use_cache and cache_dir is None:
        cache_dir = nist_cache_dir()
    print(f"Database: {db_path}")
    print(f"Page cache: {cache_dir if use_cache else '(off)'}")
    conn = init_db(db_path)
    cur = conn.cursor()
    fetcher = PoliteNISTFetcher(cache_dir=cache_dir if use_cache else None)

    target_elements = elements if elements else list(NIST_ELEMENTS.keys())

    for idx, sym in enumerate(target_elements, 1):
        if sym not in NIST_ELEMENTS:
            print(f"Skipping unknown element symbol: {sym}")
            continue

        if not force_refresh:
            cur.execute("SELECT 1 FROM crawl_log WHERE element = ?", (sym,))
            if cur.fetchone():
                print(f"[{idx}/{len(target_elements)}] {sym} already in DB (skipping)")
                continue

        el_name = NIST_ELEMENTS[sym]
        print(f"[{idx}/{len(target_elements)}] Fetching {sym} ({el_name})...")

        # Clear any partial data for this element before inserting fresh rows
        for tbl in ("atomic_data", "strong_lines", "persistent_lines", "energy_levels", "bibliographic_refs"):
            cur.execute(f"DELETE FROM {tbl} WHERE element = ?", (sym,))

        # Table 1: Atomic Data (ASCII)
        html_t1 = fetcher.get(f"{BASE_URL}/{el_name}table1_a.htm")
        if html_t1:
            insert_rows(cur, "atomic_data", parse_table1_atomic_data(sym, html_t1), replace=True)

        # Table 2: Strong Lines (ASCII)
        html_t2 = fetcher.get(f"{BASE_URL}/{el_name}table2_a.htm")
        if html_t2:
            insert_rows(cur, "strong_lines", parse_table2_strong_lines(sym, html_t2))

        # Tables 3 & 4: Persistent Lines I & II (ASCII)
        for stage, t_num in [("I", "3"), ("II", "4")]:
            html_pl = fetcher.get(f"{BASE_URL}/{el_name}table{t_num}_a.htm")
            if html_pl:
                insert_rows(cur, "persistent_lines", parse_table3_4_persistent(sym, stage, html_pl))

        # Tables 5 & 6: Energy Levels I & II (ASCII)
        for stage, t_num in [("I", "5"), ("II", "6")]:
            html_el = fetcher.get(f"{BASE_URL}/{el_name}table{t_num}_a.htm")
            if html_el:
                insert_rows(cur, "energy_levels", parse_table5_6_energy_levels(sym, stage, html_el))

        # Table 7: Bibliographic References (HTML)
        html_t7 = fetcher.get(f"{BASE_URL}/{el_name}table7.htm")
        if html_t7:
            insert_rows(cur, "bibliographic_refs", parse_table7_references(sym, html_t7), replace=True)

        cur.execute("INSERT OR REPLACE INTO crawl_log (element) VALUES (?)", (sym,))
        conn.commit()

    # Print summary counts
    print("\n--- Database Build Complete ---")
    for tbl in ("atomic_data", "strong_lines", "persistent_lines", "energy_levels", "bibliographic_refs"):
        count = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:20s}: {count:,} rows")

    conn.close()


def add_arguments(parser: argparse.ArgumentParser):
    """Build options, shared by `python -m libs_viewer.nist.build` and `libs-viewer nist build`."""
    parser.add_argument("--db", help=f"Output SQLite database (default: {nist_db_path()}).")
    parser.add_argument(
        "--elements", nargs="*",
        help="Optional subset of element symbols to fetch (e.g., --elements Ce U Th Nd Li F Be). Fetches all 99 if omitted."
    )
    parser.add_argument("--force", action="store_true", help="Re-parse elements even if already logged in crawl_log.")
    parser.add_argument(
        "--cache-dir",
        help=f"Folder for the raw pages (default: {nist_cache_dir()}); pages already there are parsed without re-downloading."
    )
    parser.add_argument("--no-cache", action="store_true", help="Don't read or keep raw pages.")


def run(args: argparse.Namespace):
    build_database(db_path=args.db, elements=args.elements, force_refresh=args.force,
                   cache_dir=args.cache_dir, use_cache=not args.no_cache)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Build local SQLite DB from NIST Atomic Spectroscopy Handbook.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()