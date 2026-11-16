#!/usr/bin/env python3
"""
NIST Handbook of Basic Atomic Spectroscopic Data -> Local SQLite Builder
Crawls Tables 1-7 for elements Z=1..99 while strictly respecting physics.nist.gov/robots.txt.
"""

import re
import time
import html as html_lib
import sqlite3
import argparse
import urllib.robotparser
from typing import Optional, List, Dict
import requests

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
    Parses {element}table1.htm (HTML) for Atomic Number, Atomic Weight,
    and Ground State / Ionization Energies across stages I, II, III, IV.
    """
    clean_all = strip_html_tags(html_text)
    z_match = re.search(r"Atomic Number:\s*(\d+)", clean_all, re.IGNORECASE)
    wt_match = re.search(r"Atomic Weight:\s*([0-9.()\[\]]+)", clean_all, re.IGNORECASE)

    atomic_number = int(z_match.group(1)) if z_match else None
    atomic_weight = clean_float(wt_match.group(1)) if wt_match else None

    records = []
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_text, flags=re.DOTALL | re.IGNORECASE)
    for row in rows:
        cols = [
            strip_html_tags(c)
            for c in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL | re.IGNORECASE)
        ]
        if len(cols) >= 6 and re.match(rf"^{symbol}\s+[IVX]+$", cols[0]):
            parts = cols[0].split()
            records.append({
                "species": cols[0],
                "element": symbol,
                "ion_stage": parts[1] if len(parts) > 1 else "",
                "atomic_number": atomic_number,
                "atomic_weight": atomic_weight,
                "ground_config": cols[1],
                "ground_term": cols[2],
                "ground_j": cols[3],
                "ground_g": j_to_g(cols[3]),
                "ionization_energy_eV": clean_float(cols[4]),
                "ionization_energy_cm1": clean_float(cols[5]),
                "ref": cols[6] if len(cols) > 6 else ""
            })
    return records


def parse_table2_strong_lines(symbol: str, html_text: str) -> List[Dict]:
    """Parses {element}table2_a.htm (ASCII) Strong Lines across neutral and ionized states."""
    records = []
    for line in extract_pre_lines(html_text):
        s = line.strip()
        if not s or s.startswith(("Intensity", "Wavelength", "---", "Air", "Vacuum")):
            continue
        # Format: Intensity + Flags | Wavelength | Spectrum (e.g., Ce I, Ce II) | Ref
        m = re.match(
            r"^\s*(\d+)\s*([a-zA-Z,\s\*]*?)\s+(\d+\.\d+)\s+([A-Za-z]+\s+[IVX]+)\s+(\S+)",
            line
        )
        if m:
            intens_str, flags, wl_str, spectrum, ref = m.groups()
            clean_flags = flags.replace(" ", "")
            records.append({
                "element": symbol,
                "species": re.sub(r"\s+", " ", spectrum.strip()),
                "wavelength_A": float(wl_str),
                "intensity": int(intens_str),
                "flags": clean_flags,
                "is_persistent": 1 if "P" in clean_flags else 0,
                "ref": ref.strip()
            })
    return records


def parse_table3_4_persistent(symbol: str, ion_stage: str, html_text: str) -> List[Dict]:
    """
    Parses {element}table3_a.htm (I) and table4_a.htm (II) (ASCII).
    Pairs the 2-row Lower Level (i) and Upper Level (k) entries into flat records.
    """
    lines = extract_pre_lines(html_text)
    records = []
    i = 0
    while i < len(lines):
        line1 = lines[i]
        s = line1.strip()
        if not s or s.startswith(("Intensity", "Wavelength", "---", "Levels", "Air", "Vacuum")):
            i += 1
            continue

        padded1 = line1.ljust(95)
        intens_raw = padded1[0:11].strip()
        wl_raw = padded1[11:24].strip()

        # First row of a transition pair has both Intensity and a decimal Wavelength
        if intens_raw and wl_raw and re.match(r"^\d+\.\d+$", wl_raw):
            aki_raw = padded1[24:31].strip()
            ei_raw = padded1[31:44].strip()
            conf_i = padded1[44:69].strip()
            term_i = padded1[69:79].strip()
            j_i = padded1[79:85].strip()
            refs = padded1[85:].strip().split()

            ek_raw, conf_k, term_k, j_k = "", "", "", ""
            if i + 1 < len(lines) and lines[i + 1].strip():
                padded2 = lines[i + 1].ljust(95)
                # Ensure line2 is actually the upper state continuation (blank intensity/wavelength)
                if not padded2[0:24].strip():
                    ek_raw = padded2[31:44].strip()
                    conf_k = padded2[44:69].strip()
                    term_k = padded2[69:79].strip()
                    j_k = padded2[79:85].strip()
                    i += 1

            intens_num = re.search(r"(\d+)", intens_raw)
            records.append({
                "species": f"{symbol} {ion_stage}",
                "element": symbol,
                "ion_stage": ion_stage,
                "wavelength_A": float(wl_raw),
                "intensity": int(intens_num.group(1)) if intens_num else None,
                "intensity_flags": re.sub(r"[\d\s]+", "", intens_raw),
                "aki_10_8_s1": clean_float(aki_raw),
                "ei_cm1": clean_float(ei_raw),
                "ek_cm1": clean_float(ek_raw),
                "config_i": conf_i,
                "term_i": term_i,
                "j_i": j_i,
                "g_i": j_to_g(j_i),
                "config_k": conf_k,
                "term_k": term_k,
                "j_k": j_k,
                "g_k": j_to_g(j_k),
                "line_ref": refs[0] if len(refs) > 0 else "",
                "aki_ref": refs[1] if len(refs) > 1 else ""
            })
        i += 1
    return records


def parse_table5_6_energy_levels(symbol: str, ion_stage: str, html_text: str) -> List[Dict]:
    """
    Parses {element}table5_a.htm (I) and table6_a.htm (II) (ASCII) Energy Levels.
    Forward-fills Configuration and Term across fine-structure J rows within each multiplet block.
    """
    lines = extract_pre_lines(html_text)
    records = []
    curr_config = ""
    curr_term = ""

    for line in lines:
        if not line.strip():
            curr_config, curr_term = "", ""
            continue
        if line.strip().startswith(("Configuration", "---")):
            continue

        padded = line.ljust(75)
        conf_col = padded[0:28].strip()
        term_col = padded[28:39].strip()
        j_col = padded[39:47].strip()
        lvl_col = padded[47:64].strip()
        ref_col = padded[64:].strip()

        if conf_col:
            curr_config = conf_col
        if term_col:
            curr_term = term_col

        energy_val = clean_float(lvl_col)
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
            "j": j_col,
            "g": j_to_g(j_col),
            "energy_cm1": energy_val,
            "is_ionization_limit": is_limit,
            "ref": ref_col
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
    def __init__(self, user_agent: str = USER_AGENT, min_delay: float = 0.5):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self.delay = min_delay
        self.rp = urllib.robotparser.RobotFileParser()
        self.rp.set_url(ROBOTS_URL)
        
        self.has_robots = False
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


def init_db(db_path: str) -> sqlite3.Connection:
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
            wavelength_A REAL,
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


def build_database(
    db_path: str = "nist_handbook.sqlite",
    elements: Optional[List[str]] = None,
    force_refresh: bool = False
):
    conn = init_db(db_path)
    cur = conn.cursor()
    fetcher = PoliteNISTFetcher()

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

        # Table 1: Atomic Data (HTML)
        html_t1 = fetcher.get(f"{BASE_URL}/{el_name}table1.htm")
        if html_t1:
            t1_rows = parse_table1_atomic_data(sym, html_t1)
            cur.executemany("""
                INSERT OR REPLACE INTO atomic_data VALUES (
                    :species, :element, :ion_stage, :atomic_number, :atomic_weight,
                    :ground_config, :ground_term, :ground_j, :ground_g,
                    :ionization_energy_eV, :ionization_energy_cm1, :ref
                )
            """, t1_rows)

        # Table 2: Strong Lines (ASCII)
        html_t2 = fetcher.get(f"{BASE_URL}/{el_name}table2_a.htm")
        if html_t2:
            t2_rows = parse_table2_strong_lines(sym, html_t2)
            cur.executemany("""
                INSERT INTO strong_lines VALUES (
                    :element, :species, :wavelength_A, :intensity, :flags, :is_persistent, :ref
                )
            """, t2_rows)

        # Tables 3 & 4: Persistent Lines I & II (ASCII)
        for stage, t_num in [("I", "3"), ("II", "4")]:
            html_pl = fetcher.get(f"{BASE_URL}/{el_name}table{t_num}_a.htm")
            if html_pl:
                pl_rows = parse_table3_4_persistent(sym, stage, html_pl)
                cur.executemany("""
                    INSERT INTO persistent_lines VALUES (
                        :species, :element, :ion_stage, :wavelength_A, :intensity, :intensity_flags,
                        :aki_10_8_s1, :ei_cm1, :ek_cm1, :config_i, :term_i, :j_i, :g_i,
                        :config_k, :term_k, :j_k, :g_k, :line_ref, :aki_ref
                    )
                """, pl_rows)

        # Tables 5 & 6: Energy Levels I & II (ASCII)
        for stage, t_num in [("I", "5"), ("II", "6")]:
            html_el = fetcher.get(f"{BASE_URL}/{el_name}table{t_num}_a.htm")
            if html_el:
                el_rows = parse_table5_6_energy_levels(sym, stage, html_el)
                cur.executemany("""
                    INSERT INTO energy_levels VALUES (
                        :species, :element, :ion_stage, :configuration, :term, :parity,
                        :j, :g, :energy_cm1, :is_ionization_limit, :ref
                    )
                """, el_rows)

        # Table 7: Bibliographic References (HTML)
        html_t7 = fetcher.get(f"{BASE_URL}/{el_name}table7.htm")
        if html_t7:
            t7_rows = parse_table7_references(sym, html_t7)
            cur.executemany("""
                INSERT OR REPLACE INTO bibliographic_refs VALUES (:element, :ref_code, :citation)
            """, t7_rows)

        cur.execute("INSERT OR REPLACE INTO crawl_log (element) VALUES (?)", (sym,))
        conn.commit()

    # Print summary counts
    print("\n--- Database Build Complete ---")
    for tbl in ("atomic_data", "strong_lines", "persistent_lines", "energy_levels", "bibliographic_refs"):
        count = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:20s}: {count:,} rows")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build local SQLite DB from NIST Atomic Spectroscopy Handbook.")
    parser.add_argument("--db", default="nist_handbook.sqlite", help="Output SQLite database filename.")
    parser.add_argument(
        "--elements", nargs="*",
        help="Optional subset of element symbols to fetch (e.g., --elements Ce U Th Nd Li F Be). Fetches all 99 if omitted."
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch elements even if already logged in crawl_log.")
    args = parser.parse_args()

    build_database(db_path=args.db, elements=args.elements, force_refresh=args.force)