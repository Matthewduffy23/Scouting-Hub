# streamlit_app.py
# ✅ Club Scouting — CLEAN Pro Tiles (FotMob photos + crests only)
# - Dataset picker (WORLD*.csv or upload)
# - Candidate pool filters (top bar)
# - Team template selector (top)
# - Same Role Fit maths: BaseDist → optional league mismatch penalty → exp-decay → optional league blend
# - Tabs: Strikers / Attackers / Central Mid / Fullbacks / Center Backs
# - Tiles: crest + photo + Match% (Role Fit Score)
# - Optional per-player photo override URL (stored in session)

import io, math, re, time, os, json, base64, unicodedata
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from numpy.linalg import norm
from numpy import exp
import requests
from difflib import SequenceMatcher
from sklearn.preprocessing import StandardScaler

# ========================= PAGE =========================
st.set_page_config(page_title="Club Scouting — Pro Tiles (FotMob)", layout="wide")
st.title("🔎 Advanced Club Scouting — Pro Tiles (FotMob)")
st.caption("Template team → role fit matching → clean tiles. Match% = Role Fit Score (distance + optional league blend).")

# ========================= DATASET PICKER =========================
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def _candidate_dirs() -> List[Path]:
    dirs: List[Path] = [Path.cwd()]
    try:
        dirs.append(Path.cwd().parent)
    except Exception:
        pass
    try:
        here = Path(__file__).resolve().parent
        dirs.extend([here, here.parent])
    except Exception:
        pass
    seen, uniq = set(), []
    for d in dirs:
        rp = d.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq

def _find_world_csvs() -> List[Path]:
    files: List[Path] = []
    for base in _candidate_dirs():
        files.extend(sorted(base.glob("WORLD*.csv")))
    seen, uniq = set(), []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq

def _label_for(p: Path) -> str:
    parent_hint = p.parent.name or str(p.parent)
    return f"{p.name} — {parent_hint}/"

def pick_or_upload_world_csv() -> Tuple[pd.DataFrame, str]:
    st.markdown("### 📁 Data Source")
    found = _find_world_csvs()
    labels_found = [_label_for(p) for p in found]
    labels = ["Upload a CSV…"] + labels_found
    default_index = 1 if found else 0

    sel = st.selectbox("Select a WORLD*.csv file", labels, index=default_index, key="world_csv_picker")

    if sel == "Upload a CSV…":
        up = st.file_uploader("Upload a WORLD*.csv", type=["csv"], key="world_csv_uploader")
        if up is None:
            st.info("Pick a file from the list above or upload one.")
            st.stop()
        df_up = _read_csv_from_bytes(up.getvalue())
        return df_up, up.name

    idx = labels.index(sel) - 1
    chosen_path = found[idx]
    df_disk = _read_csv_from_path(str(chosen_path))
    return df_disk, chosen_path.resolve().name

df, DATASET_NAME = pick_or_upload_world_csv()

# Reset session maps when dataset changes
if st.session_state.get("_active_dataset_name") != DATASET_NAME:
    for k in ["photo_map", "crest_map", "st_tmpl_pick", "att_tmpl_pick", "cm_tmpl_pick", "fb_tmpl_pick", "cb_tmpl_pick"]:
        st.session_state.pop(k, None)
    st.session_state["_active_dataset_name"] = DATASET_NAME

st.session_state.setdefault("photo_map", {})  # per-player overrides
st.session_state.setdefault("crest_map", {})  # per-team overrides

# ========================= LEAGUES & STRENGTHS =========================
INCLUDED_LEAGUES = [
    'England 1.','England 2.','England 3.','England 4.','England 5.','England 6.','England 7.','England 8.','England 9.','England 10.',
    'Albania 1.','Algeria 1.','Andorra 1.','Argentina 1.','Armenia 1.','Australia 1.','Austria 1.','Austria 2.','Azerbaijan 1.','Belgium 1.',
    'Belgium 2.','Bolivia 1.','Bosnia 1.','Brazil 1.','Brazil 2.','Brazil 3.','Bulgaria 1.','Canada 1.','Chile 1.','Colombia 1.',
    'Costa Rica 1.','Croatia 1.','Cyprus 1.','Czech 1.','Czech 2.','Denmark 1.','Denmark 2.','Ecuador 1.','Egypt 1.','Estonia 1.',
    'Finland 1.','France 1.','France 2.','France 3.','Georgia 1.','Germany 1.','Germany 2.','Germany 3.','Germany 4.','Greece 1.',
    'Hungary 1.','Iceland 1.','Israel 1.','Israel 2.','Italy 1.','Italy 2.','Italy 3.','Japan 1.','Japan 2.','Kazakhstan 1.',
    'Korea 1.','Latvia 1.','Lithuania 1.','Malta 1.','Mexico 1.','Moldova 1.','Morocco 1.','Netherlands 1.','Netherlands 2.',
    'North Macedonia 1.','Northern Ireland 1.','Norway 1.','Norway 2.','Paraguay 1.','Peru 1.','Poland 1.','Poland 2.',
    'Portugal 1.','Portugal 2.','Portugal 3.','Qatar 1.','Ireland 1.','Romania 1.','Russia 1.','Saudi 1.','Scotland 1.','Scotland 2.',
    'Scotland 3.','Serbia 1.','Serbia 2.','Slovakia 1.','Slovakia 2.','Slovenia 1.','Slovenia 2.','South Africa 1.','Spain 1.','Spain 2.',
    'Spain 3.','Sweden 1.','Sweden 2.','Switzerland 1.','Switzerland 2.','Tunisia 1.','Turkey 1.','Turkey 2.','Ukraine 1.','UAE 1.',
    'USA 1.','USA 2.','Uruguay 1.','Uzbekistan 1.','Venezuela 1.','Wales 1.'
]

PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.','France 1.','Germany 1.','Italy 1.','Spain 1.'},
    "Top 20 Europe": {'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.','England 2.','Portugal 1.','Belgium 1.','Turkey 1.','Germany 2.','Spain 2.','France 2.','Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'},
    "EFL (England 2–4)": {'England 2.','England 3.','England 4.'}
}

LEAGUE_STRENGTHS = {
'England 1.':100.00,'Spain 1.':87.84,'Germany 1.':87.45,'Italy 1.':85.88,'France 1.':83.14,
'England 2.':75.10,'Belgium 1.':74.51,'Brazil 1.':74.31,'Portugal 1.':72.94,'Argentina 1.':71.37,
'USA 1.':70,'Denmark 1.':70.78,'Poland 1.':69.61,'Turkey 1.':69.02,'Netherlands 1.':69.02,
'Croatia 1.':68.43,'Germany 2.':68.04,'Japan 1.':67.84,'Switzerland 1.':67.45,'Spain 2.':67.06,
'Norway 1.':66.67,'Mexico 1.':66.47,'Sweden 1.':66.27,'Colombia 1.':65.88,'Cyprus 1.':60,
'Czech 1.':65.29,'Ecuador 1.':65.29,'Greece 1.':64.12,'Saudi 1.':64.12,'Italy 2.':63.53,
'Hungary 1.':63.53,'Austria 1.':63.33,'Morocco 1.':63.14,'Korea 1.':62.75,'Paraguay 1.':62.55,
'France 2.':64,'England 3.':61.96,'Romania 1.':61.76,'Scotland 1.':61.76,'Algeria 1.':61.57,
'Uruguay 1.':60.39,'Chile 1.':59.80,'Egypt 1.':59.22,'Israel 1.':58.43,'Brazil 2.':58.04,
'Slovenia 1.':57.45,'Bolivia 1.':57.25,'Slovakia 1.':56.47,'Azerbaijan 1.':56.47,'South Africa 1.':56.27,
'UAE 1.':55.49,'Costa Rica 1.':54.90,'Peru 1.':54.90,'Germany 3.':54.51,'Ukraine 1.':54.31,
'Spain 3.':54.31,'Portugal 2.':53.14,'Bulgaria 1.':53.14,'Australia 1.':52.75,'Serbia 1.':52.16,
'Albania 1.':51.96,'Bosnia 1.':51.76,'Kosovo 1.':51.37,'Japan 2.':50.98,'England 4.':50.78,
'Ireland 1.':50.59,'Russia 1.':62.41,'Kazakhstan 1.':50.39,'Nigeria 1.':50.00,'France 3.':49.61,
'Tunisia 1.':49.22,'Venezuela 1.':48.63,'Belgium 2.':48.43,'Finland 1.':48.43,'Armenia 1.':47.84,
'Georgia 1.':47.65,'Switzerland 2.':46.47,'Qatar 1.':46.27,'Uzbekistan 1.':46.27,'Poland 2.':46.27,
'Iceland 1.':46.08,'Norway 2.':45.88,'Sweden 2.':45.69,'North Macedonia 1.':44.71,'China 1.':44.7, 'Turkey 2.':44.51,
'Korea 2.':43.53,'Czech 2.':43.33,'Brazil 3.':43.14,'Lithuania 1.':42.35,'Netherlands 2.':42.16,
'Malta 1.':41.96,'Italy 3.':45,'Denmark 2.':40.39,'Moldova 1.':40.39,'USA 2.':40.00,
'Latvia 1.':40.00,'Montenegro 1.':39.80,'Scotland 2.':38.63,'Canada 1.':38.24,'Austria 2.':38.24,
'Israel 2.':38.04,'England 7.':37.25,'Germany 4.':35.29,'Portugal 3.':35.29,'England 5.':33.33,
'Estonia 1.':40,'England 9.':31.37,'Northern Ireland 1.':30.98,'Serbia 2.':30.39,'Denmark 3.':29.41,
'Sweden 3.':29.41,'Slovenia 2.':28.82,'Slovakia 2.':28.24,'Greece 2.':27.06,'Wales 1.':26.67,
'USA 3.':22.55,'Scotland 3.':20.00,'England 6.':16.08,'England 8.':15.69,'England 10.':3.92,
'Estonia 2.':3, 'Ireland 2.':10,
}

# ========================= GBE BANDS + REGIONS (PRESETS) =========================
# Put this block AFTER LEAGUE_STRENGTHS = {...} and BEFORE the TOP BAR section.

# ---- GBE league bands (custom: all UK & Ireland leagues in Band 1) ----
GBE_LEAGUE_BANDS = {
    # Band 1 – Top 5 + all England / Scotland / Wales / Ireland / Northern Ireland leagues
    "England 1.": 1, "England 2.": 1, "England 3.": 1, "England 4.": 1,
    "England 5.": 1, "England 6.": 1, "England 7.": 1, "England 8.": 1,
    "England 9.": 1, "England 10.": 1,
    "Scotland 1.": 1, "Scotland 2.": 1, "Scotland 3.": 1,
    "Wales 1.": 1,
    "Ireland 1.": 1,
    "Northern Ireland 1.": 1,

    "Spain 1.": 1, "Germany 1.": 1, "Italy 1.": 1, "France 1.": 1,

    # Band 2
    "Portugal 1.": 2, "Netherlands 1.": 2, "Belgium 1.": 2, "Turkey 1.": 2,

    # Band 3
    "USA 1.": 3, "Brazil 1.": 3, "Argentina 1.": 3, "Mexico 1.": 3,

    # Band 4
    "Czech 1.": 4, "Croatia 1.": 4, "Switzerland 1.": 4,
    "Spain 2.": 4, "Germany 2.": 4,
    "Ukraine 1.": 4, "Greece 1.": 4, "Colombia 1.": 4,
    "Austria 1.": 4, "Denmark 1.": 4, "France 2.": 4, "Russia 1.": 4,

    # Band 5
    "Serbia 1.": 5, "Poland 1.": 5, "Slovenia 1.": 5, "Chile 1.": 5, "Uruguay 1.": 5,
    "Sweden 1.": 5, "Norway 1.": 5, "Italy 2.": 5, "Hungary 1.": 5, "Japan 1.": 5,
    "Korea 1.": 5, "Australia 1.": 5,

    # Everything else defaults to Band 6
}

def gbe_league_band(league_name: str) -> int:
    """Map 'Country N.' league name to custom GBE band 1–6. Unlisted leagues default to Band 6."""
    league_name = str(league_name).strip()
    return int(GBE_LEAGUE_BANDS.get(league_name, 6))

@st.cache_data(show_spinner=False)
def league_strength_band_df(max_band: int = 6) -> pd.DataFrame:
    """
    Return a dataframe with League, League Strength, and GBE Band,
    filtered to bands <= max_band.
    """
    rows = []
    for league, strength in LEAGUE_STRENGTHS.items():
        band = gbe_league_band(league)
        if band <= max_band:
            rows.append({"League": league, "League strength": strength, "GBE band": band})
    df_ls = pd.DataFrame(rows)
    return df_ls.sort_values(["League strength"], ascending=[False])

# --- Country → Region mapping for league-level region filter ---
COUNTRY_TO_REGION = {
    # Europe
    "England": "Europe", "Spain": "Europe", "Germany": "Europe", "Italy": "Europe",
    "France": "Europe", "Belgium": "Europe", "Portugal": "Europe", "Netherlands": "Europe",
    "Croatia": "Europe", "Switzerland": "Europe", "Norway": "Europe", "Sweden": "Europe",
    "Cyprus": "Europe", "Czech": "Europe", "Greece": "Europe", "Austria": "Europe",
    "Hungary": "Europe", "Romania": "Europe", "Scotland": "Europe", "Slovenia": "Europe",
    "Slovakia": "Europe", "Ukraine": "Europe", "Bulgaria": "Europe", "Serbia": "Europe",
    "Albania": "Europe", "Bosnia": "Europe", "Kosovo": "Europe", "Ireland": "Europe",
    "Finland": "Europe", "Armenia": "Europe", "Georgia": "Europe", "Poland": "Europe",
    "Iceland": "Europe", "North Macedonia": "Europe", "Latvia": "Europe",
    "Montenegro": "Europe", "Denmark": "Europe", "Estonia": "Europe",
    "Northern Ireland": "Europe", "Wales": "Europe",

    # South America
    "Brazil": "South America", "Argentina": "South America", "Colombia": "South America",
    "Ecuador": "South America", "Paraguay": "South America", "Uruguay": "South America",
    "Chile": "South America", "Bolivia": "South America", "Peru": "South America",
    "Venezuela": "South America",

    # North America
    "USA": "North America", "Mexico": "North America", "Costa Rica": "North America",
    "Canada": "North America",

    # Africa
    "Morocco": "Africa", "Algeria": "Africa", "Egypt": "Africa", "Nigeria": "Africa",
    "Tunisia": "Africa", "South Africa": "Africa",

    # Asia
    "Japan": "Asia", "Korea": "Asia", "Saudi": "Asia",
    "UAE": "Asia", "Qatar": "Asia", "Uzbekistan": "Asia", "Israel": "Asia",
    "Turkey": "Asia", "Azerbaijan": "Asia",

    # Oceania / other
    "Australia": "Asia",
}

def league_country(league: str) -> str:
    """
    Extract country part from 'Country N.' including multi-word countries.
    Examples:
      'North Macedonia 1.' -> 'North Macedonia'
      'Northern Ireland 1.' -> 'Northern Ireland'
      'South Africa 1.' -> 'South Africa'
      'England 2.' -> 'England'
    """
    s = str(league).strip()
    m = re.match(r"^(.*)\s\d+\.\s*$", s)
    return m.group(1).strip() if m else s

def league_region(league: str) -> str:
    """Map a league to a region using the country part."""
    c = league_country(league)
    return COUNTRY_TO_REGION.get(c, "Other")


# ========================= TOP BAR: FILTERS + SCORING =========================
st.markdown("---")
st.header("⚙️ Adjustments & Candidate Pool")

cA, cB, cC, cD = st.columns([1.2, 1.6, 1.2, 1.2])
with cA:
    st.markdown("**League presets**")
    use_top5 = st.checkbox("Top-5 preset", False)
    use_top20 = st.checkbox("Top-20 preset", False)
    use_efl = st.checkbox("EFL preset", False)

    st.markdown("**Region presets (multi)**")
    region_picks = st.multiselect(
        "Regions",
        ["Europe", "South America", "North America", "Africa", "Asia", "Other"],
        default=[],
        key="preset_region_picks",
    )

    st.markdown("**GBE band presets (multi)**")
    band_picks = st.multiselect(
        "Bands (exact)",
        [1, 2, 3, 4, 5, 6],
        default=[],
        key="preset_band_picks",
    )

    band_max = st.selectbox(
        "Or include all bands ≤",
        ["— None —", 1, 2, 3, 4, 5, 6],
        index=0,
        key="preset_band_max",
    )

# ----- build preset league seed -----
seed = set()

# Existing presets
if use_top5:
    seed |= PRESET_LEAGUES["Top 5 Europe"]
if use_top20:
    seed |= PRESET_LEAGUES["Top 20 Europe"]
if use_efl:
    seed |= PRESET_LEAGUES["EFL (England 2–4)"]

# Region presets (multi)
if region_picks:
    seed |= {lg for lg in INCLUDED_LEAGUES if league_region(lg) in set(region_picks)}

# Band presets (multi)
if band_picks:
    seed |= {lg for lg in INCLUDED_LEAGUES if gbe_league_band(lg) in set(band_picks)}

# Band max preset (≤)
if band_max != "— None —":
    seed |= {lg for lg in INCLUDED_LEAGUES if gbe_league_band(lg) <= int(band_max)}

# ----- leagues list + default selection -----
leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df["League"].dropna().astype(str).unique()))
default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES

with cB:
    leagues_sel = st.multiselect(
        "Leagues in candidate pool",
        leagues_avail,
        default=default_leagues
    )

with cC:
    min_minutes, max_minutes = st.slider("Minutes (pool)", 0, 6000, (750, 6000))
    age_min_data = int(np.nanmin(pd.to_numeric(df["Age"], errors="coerce"))) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(pd.to_numeric(df["Age"], errors="coerce"))) if df["Age"].notna().any() else 50
    min_age, max_age = st.slider("Age (pool)", age_min_data, age_max_data, (16, 50))

with cD:
    mv_col = pd.to_numeric(df["Market value"], errors="coerce")
    mv_max_raw = int(np.nanmax(mv_col)) if mv_col.notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    use_m = st.checkbox("Market value in M€", True)
    if use_m:
        max_m = max(1, mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("MV range (M€)", 0, max_m, (0, min(max_m, 10)))
        pool_min_value, pool_max_value = mv_min_m * 1_000_000, mv_max_m * 1_000_000
    else:
        pool_min_value, pool_max_value = st.slider("MV range (€)", 0, mv_cap, (0, min(mv_cap, 10_000_000)), step=100_000)

cE, cF, cG, cH = st.columns([1.2, 1.2, 1.2, 1.2])
with cE:
    decay_rate = st.slider(
        "Exp. decay (↑=stricter)", 0.5, 10.0, 5.0, 0.5,
        help="Higher = harsher scoring. Lower = more forgiving shortlist."
    )
with cF:
    use_league_weighting = st.checkbox(
        "Blend league strength (β)", value=True,
        help="If on: adds league strength into the final score (controlled by β)."
    )
    beta = st.slider(
        "β (0–1)", 0.0, 1.0, 0.40, 0.05,
        help="β=0 pure role fit. β=1 pure league strength."
    )
with cG:
    use_league_mismatch = st.checkbox(
        "League mismatch penalty (α,p)", value=True,
        help="If on: penalizes players from much weaker/stronger leagues vs your template league."
    )
    alpha = st.slider(
        "α", 0.0, 5.0, 1.20, 0.05,
        help="Penalty size. Higher α makes league differences matter more."
    )
    p_exp = st.slider(
        "p", 1.0, 3.0, 1.50, 0.10,
        help="Penalty curve. p>1 punishes big league gaps much more than small gaps."
    )
with cH:
    penalty_mode = st.selectbox(
        "Penalty combine", ["Additive (stronger)", "Quadrature (gentler)"], index=0,
        help="Additive = BaseDist+Penalty. Quadrature = sqrt(BaseDist²+Penalty²)."
    )
    min_strength, max_strength = st.slider("League strength (pool)", 0, 101, (0, 101))
    top_n = st.number_input("Top N", 5, 200, 20, 5)

if seed:
    st.caption(f"Preset leagues selected: **{len(seed)}**")

DEBUG_PHOTOS = st.checkbox("Debug photos", False)


# ========================= TEAM TEMPLATE (TOP) =========================
st.markdown("---")
st.header("🎯 Team Template")

template_league_list = sorted([str(x) for x in df["League"].dropna().unique()])
template_league = st.selectbox("Template league", template_league_list)

templ_teams_all = sorted(df.loc[df["League"].astype(str) == template_league, "Team"].dropna().astype(str).unique())
team_search = st.text_input("Search team", "")
templ_teams = [t for t in templ_teams_all if team_search.lower() in t.lower()] or templ_teams_all
template_team = st.selectbox("Template team", templ_teams)

min_minutes_template = st.slider("Min minutes for template players", 0, 6000, 1000, 100)
use_single_template_player = st.checkbox("Use single template player (otherwise role avg)", False)
template_strength = float(LEAGUE_STRENGTHS.get(template_league, 0.0))

# ========================= HELPERS =========================
def build_base_pool():
    p = df.copy()
    p = p[p["League"].isin(leagues_sel)]

    for c in ["Minutes played", "Age", "Market value", "Goals"]:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")

    # These are the ONLY pool filters now (top-bar controls)
    p = p[p["Minutes played"].between(min_minutes, max_minutes)]

    p["League Strength"] = p["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
    p = p[(p["League Strength"] >= float(min_strength)) & (p["League Strength"] <= float(max_strength))]
    return p

def _template_rows_for_role(pos_predicate):
    src = df[
        (df["League"].astype(str) == template_league)
        & (df["Team"].astype(str) == template_team)
        & (df["Position"].apply(lambda p: pos_predicate(str(p))))
    ].copy()
    src["Minutes played"] = pd.to_numeric(src["Minutes played"], errors="coerce")
    src = src[src["Minutes played"] >= min_minutes_template]
    return src

def _score_block(df_with_baseDist: pd.DataFrame) -> pd.DataFrame:
    if use_league_mismatch:
        base_min, base_max = float(df_with_baseDist["BaseDist"].min()), float(df_with_baseDist["BaseDist"].max())
        spread = max(1e-9, base_max - base_min)

        def _with_pen(row):
            ls = float(LEAGUE_STRENGTHS.get(str(row["League"]), 0.0))
            delta = abs(ls - template_strength) / 100.0
            pen = alpha * (delta ** p_exp) * spread
            return row["BaseDist"] + pen if penalty_mode.startswith("Additive") else float(np.hypot(row["BaseDist"], pen))

        df_with_baseDist["Role Fit Distance"] = df_with_baseDist.apply(_with_pen, axis=1)
    else:
        df_with_baseDist["Role Fit Distance"] = df_with_baseDist["BaseDist"]

    dmin, dmax = float(df_with_baseDist["Role Fit Distance"].min()), float(df_with_baseDist["Role Fit Distance"].max())
    rng = dmax - dmin
    if rng <= 1e-12:
        base_score = pd.Series(100.0, index=df_with_baseDist.index)
    else:
        base_score = 100.0 * exp(-decay_rate * ((df_with_baseDist["Role Fit Distance"] - dmin) / rng))

    league_part = df_with_baseDist["League"].map(LEAGUE_STRENGTHS).fillna(0.0) if use_league_weighting else 0.0
    df_with_baseDist["Role Fit Score"] = (1.0 - beta) * base_score + beta * league_part
    return df_with_baseDist.sort_values("Role Fit Score", ascending=False).reset_index(drop=True)

def _safe_verticality(forward_per90, passes_per90):
    f = pd.to_numeric(forward_per90, errors="coerce")
    p = pd.to_numeric(passes_per90, errors="coerce").replace(0, np.nan)
    return (f / p).fillna(0.0)

def render_template_players_used(role_name: str, tmpl_src: pd.DataFrame):
    showcols = [c for c in ["Player", "Minutes played", "Position", "League", "Team"] if c in tmpl_src.columns]
    st.subheader(f"🧩 Template players used — {role_name}")
    if tmpl_src.empty or not showcols:
        st.info("No eligible template players for the selected team/filters.")
        return
    st.dataframe(tmpl_src[showcols].sort_values("Minutes played", ascending=False), use_container_width=True)

# ---------- NEW: percentile helpers for ROLE FIT ----------
def _percentile_of_value(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.5
    return float((s <= float(value)).mean())

def _add_role_metric_percentiles_vs_league(
    *,
    pool_df: pd.DataFrame,
    role_pos_predicate,
    feats: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    For Role Fit: compute per-league percentiles for engineered role metrics, and attach
    columns __pct__<metric> (0..100) to pool_df.
    Reference distribution is: df filtered to same league, same role predicate, minutes within (min_minutes,max_minutes),
    and non-missing feats + metric_cols.
    """
    out = pool_df.copy()

    # make sure minutes numeric for reference filtering
    ref = df.copy()
    if "Minutes played" in ref.columns:
        ref["Minutes played"] = pd.to_numeric(ref["Minutes played"], errors="coerce")
    for c in feats:
        if c in ref.columns:
            ref[c] = pd.to_numeric(ref[c], errors="coerce")

    # Build per-league reference ranks for each metric
    leagues_needed = sorted(set(out["League"].dropna().astype(str).unique()).union({str(template_league)}))
    pct_maps = {}

    for lg in leagues_needed:
        ref_lg = ref[ref["League"].astype(str) == str(lg)].copy()
        ref_lg = ref_lg[ref_lg["Position"].apply(lambda p: role_pos_predicate(str(p)))].copy()

        # align with pool minutes filter for stability
        if "Minutes played" in ref_lg.columns:
            ref_lg = ref_lg[ref_lg["Minutes played"].between(min_minutes, max_minutes)]

        # compute metrics must already exist in ref_lg when called (caller ensures by computing on a copy if needed)
        # Here we assume metrics exist in ref_lg; if not, we'll skip (handled by caller by computing on pool only).
        if any(m not in ref_lg.columns for m in metric_cols):
            pct_maps[str(lg)] = None
            continue

        ref_lg = ref_lg.dropna(subset=metric_cols)

        if ref_lg.empty:
            pct_maps[str(lg)] = None
            continue

        # ranks within league
        pct_maps[str(lg)] = {m: ref_lg[m].rank(pct=True).astype(float) for m in metric_cols}

    # Now compute percentiles for pool_df by league using rank(pct=True) inside pool,
    # BUT we want vs full league reference (df), not vs pool slice.
    # We'll do value->percentile lookup using _percentile_of_value per league/metric.
    for m in metric_cols:
        out[f"__pct__{m}"] = 50.0

    for lg in leagues_needed:
        idx = out["League"].astype(str) == str(lg)
        if not idx.any():
            continue

        ref_lg = df[df["League"].astype(str) == str(lg)].copy()
        if "Minutes played" in ref_lg.columns:
            ref_lg["Minutes played"] = pd.to_numeric(ref_lg["Minutes played"], errors="coerce")
            ref_lg = ref_lg[ref_lg["Minutes played"].between(min_minutes, max_minutes)]
        ref_lg = ref_lg[ref_lg["Position"].apply(lambda p: role_pos_predicate(str(p)))].copy()

        # Ensure numeric for metrics
        for m in metric_cols:
            ref_lg[m] = pd.to_numeric(ref_lg.get(m), errors="coerce")

        ref_lg = ref_lg.dropna(subset=metric_cols)
        if ref_lg.empty:
            continue

        for m in metric_cols:
            series = ref_lg[m]
            vals = pd.to_numeric(out.loc[idx, m], errors="coerce").astype(float)
            out.loc[idx, f"__pct__{m}"] = vals.map(lambda v: _percentile_of_value(series, v) * 100.0)

    return out


# ========================= SIMILARITY (PER ROLE) =========================
SIM_WEIGHTS = {
    "CB": {
        'Passes per 90': 2,
        'Accurate passes, %': 2,
        'Long passes per 90': 2,
        'Progressive passes per 90': 2,
        'Defensive duels per 90': 2,
        'Defensive duels won, %': 2,
        'Dribbles per 90': 2,
        'PAdj Interceptions': 1,
        'Progressive runs per 90': 2,
        'Aerial duels per 90': 2,
        'Aerial duels won, %': 3,
    },
    "FB": {
        'Passes per 90': 3,
        'Passes to penalty area per 90': 2,
        'Dribbles per 90': 2,
        'xA per 90': 2,
        'Progressive passes per 90': 3,
        'Defensive duels per 90': 2,
        'Forward passes per 90': 3,
        'PAdj Interceptions': 2,
        'Aerial duels won, %': 2,
        'Touches in box per 90': 2,
        'Crosses per 90': 2,
    },
    "CM": {
        'Passes per 90': 2,
        'Progressive runs per 90': 2,
        'Progressive passes per 90': 2,
        'Dribbles per 90': 2,
        'xA per 90': 2,
        'Touches in box per 90': 2,
        'Accurate passes, %': 2,
        'Aerial duels won, %': 2,
        'Non-penalty goals per 90': 2,
        'Passes to penalty area per 90': 2,
        'Defensive duels per 90': 2,
        'PAdj Interceptions': 2,
        'Defensive duels won, %': 2,
    },
    "ATT": {
        'Passes per 90': 3,
        'Accurate passes, %': 2,
        'Dribbles per 90': 3,
        'Non-penalty goals per 90': 2,
        'Shots per 90': 2,
        'Successful dribbles, %': 2,
        'Aerial duels won, %': 2,
        'xA per 90': 2,
        'xG per 90': 2,
        'Touches in box per 90': 2,
        'Passes to penalty area per 90': 2,
        'Passes to final third per 90': 2,
        'Crosses per 90': 2,
    },
    "ST": {
        'Passes per 90': 3,
        'Dribbles per 90': 3,
        'Non-penalty goals per 90': 2,
        'Aerial duels per 90': 2,
        'Aerial duels won, %': 3,
        'xA per 90': 2,
        'xG per 90': 3,
        'Touches in box per 90': 2,
        'Progressive runs per 90': 2,
        'Shots per 90': 2,
        'Accurate passes, %': 2,
    },
}

def compute_similarity_from_template(
    tmpl_src: pd.DataFrame,
    pool: pd.DataFrame,
    sim_features: List[str],
    weights_dict: dict,
    target_league: str,
    percentile_weight: float = 0.70,
    apply_league_adjust: bool = True,
    league_weight_sim: float = 0.20,
) -> pd.DataFrame:
    """
    Similarity follows top-bar pool filters (build_base_pool) EXCEPT:
      - ignores exp decay
      - ignores league mismatch penalty
      - ignores penalty combine
    """
    if tmpl_src.empty or pool.empty:
        return pd.DataFrame()

    # target vector: avg across template players (or single if already filtered upstream)
    tmpl_numeric = tmpl_src.copy()
    for f in sim_features:
        tmpl_numeric[f] = pd.to_numeric(tmpl_numeric.get(f), errors="coerce")
    tmpl_numeric = tmpl_numeric.dropna(subset=sim_features)
    if tmpl_numeric.empty:
        return pd.DataFrame()

    target_vals = tmpl_numeric[sim_features].mean().astype(float).values

    # candidates: ensure numeric, dropna
    cand = pool.copy()
    for f in sim_features:
        cand[f] = pd.to_numeric(cand.get(f), errors="coerce")
    cand = cand.dropna(subset=sim_features)

    if cand.empty:
        return pd.DataFrame()

    # one row per player (keep most minutes, then stronger league)
    cand["Minutes played"] = pd.to_numeric(cand.get("Minutes played"), errors="coerce")
    cand["League strength"] = cand["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
    cand = (
        cand.sort_values(["Player", "Minutes played", "League strength"], ascending=[True, False, False])
            .drop_duplicates(subset=["Player"], keep="first")
    )

    # weights
    weights_vec = np.array([float(weights_dict.get(f, 1.0)) for f in sim_features], dtype=float)

    # percentile target vs target league distribution
    league_block = df.loc[df["League"].astype(str) == str(target_league), sim_features].copy()
    for f in sim_features:
        league_block[f] = pd.to_numeric(league_block.get(f), errors="coerce")
    target_pct = []
    for i, f in enumerate(sim_features):
        target_pct.append(_percentile_of_value(league_block[f], float(target_vals[i])))
    target_pct = np.asarray(target_pct, dtype=float)

    # candidate percentiles per league
    percl = cand.groupby("League")[sim_features].rank(pct=True).values

    # standardize on candidate pool (actual values)
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(cand[sim_features])
    target_features_standardized = scaler.transform([target_vals])

    # distances + blend
    percentile_distances = np.linalg.norm((percl - target_pct) * weights_vec, axis=1)
    actual_value_distances = np.linalg.norm((standardized_features - target_features_standardized) * weights_vec, axis=1)
    combined = percentile_distances * float(percentile_weight) + actual_value_distances * (1.0 - float(percentile_weight))

    # normalize -> similarity 0..100
    arr = np.asarray(combined, dtype=float).ravel()
    rng = np.ptp(arr)
    normed = (arr - arr.min()) / (rng if rng != 0 else 1.0)
    similarities = ((1.0 - normed) * 100.0)

    out = cand[["Player", "Team", "League", "Position", "Age", "Minutes played", "Market value"]].copy()
    out["League strength"] = out["League"].map(LEAGUE_STRENGTHS).fillna(0.0)

    tgt_ls = float(LEAGUE_STRENGTHS.get(str(target_league), 1.0))
    eps = 1e-6
    cand_ls = np.maximum(out["League strength"].astype(float).values, eps)
    tgt_ls_safe = max(tgt_ls, eps)
    league_ratio = np.minimum(cand_ls / tgt_ls_safe, tgt_ls_safe / cand_ls)

    out["Similarity"] = np.round(similarities, 2)
    out["Adjusted Similarity"] = (
        out["Similarity"] * ((1.0 - float(league_weight_sim)) + float(league_weight_sim) * league_ratio)
    ) if apply_league_adjust else out["Similarity"]

    out = out.sort_values("Adjusted Similarity", ascending=False).reset_index(drop=True)
    return out

# ========================= ROLE CALCULATORS =========================
def compute_strikers():
    feats = ['Touches in box per 90','xG per 90','Dribbles per 90','Progressive runs per 90',
             'Aerial duels per 90','Aerial duels won, %','Passes per 90','Non-penalty goals per 90','Accurate passes, %']

    # role predicate for reference distributions
    def pos_ok(p):
        return str(p).strip().upper().startswith("CF")

    tmpl_src = _template_rows_for_role(pos_ok).dropna(subset=feats)

    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (ST)", ["— Select —"] + players, index=0, key="st_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]

    if tmpl_src.empty:
        st.error("No strikers found for template conditions.")
        st.stop()

    # ---- Template metrics (raw) ----
    f = tmpl_src.copy()
    f["Opportunities"]      = 0.7*f['Touches in box per 90'] + 0.3*f['xG per 90']
    f["Ball Carrying"]      = 0.65*f['Dribbles per 90'] + 0.35*f['Progressive runs per 90']
    f["Aerial Requirement"] = f['Aerial duels per 90'] * f['Aerial duels won, %'] / 100.0
    f["Passing Volume"]     = f['Passes per 90']
    f["Goal Output"]        = f['Non-penalty goals per 90']
    f["Retention"]          = f['Accurate passes, %']

    metric_cols = ["Opportunities","Ball Carrying","Aerial Requirement","Passing Volume","Goal Output","Retention"]

    # ---- Template vector in PERCENTILES vs TEMPLATE LEAGUE ----
    # Use template league distribution for these role metrics (not pool)
    ref_tmpl = df[(df["League"].astype(str) == str(template_league))].copy()
    ref_tmpl["Minutes played"] = pd.to_numeric(ref_tmpl.get("Minutes played"), errors="coerce")
    ref_tmpl = ref_tmpl[ref_tmpl["Minutes played"].between(min_minutes, max_minutes)]
    ref_tmpl = ref_tmpl[ref_tmpl["Position"].apply(lambda p: pos_ok(str(p)))].copy()

    # compute the same metrics on the reference block (so percentiles make sense)
    for c in feats:
        ref_tmpl[c] = pd.to_numeric(ref_tmpl.get(c), errors="coerce")
    ref_tmpl = ref_tmpl.dropna(subset=feats)

    if not ref_tmpl.empty:
        ref_tmpl["Opportunities"]      = 0.7*ref_tmpl['Touches in box per 90'] + 0.3*ref_tmpl['xG per 90']
        ref_tmpl["Ball Carrying"]      = 0.65*ref_tmpl['Dribbles per 90'] + 0.35*ref_tmpl['Progressive runs per 90']
        ref_tmpl["Aerial Requirement"] = ref_tmpl['Aerial duels per 90'] * ref_tmpl['Aerial duels won, %'] / 100.0
        ref_tmpl["Passing Volume"]     = ref_tmpl['Passes per 90']
        ref_tmpl["Goal Output"]        = ref_tmpl['Non-penalty goals per 90']
        ref_tmpl["Retention"]          = ref_tmpl['Accurate passes, %']
        ref_tmpl = ref_tmpl.dropna(subset=metric_cols)

    # turn template metric means into percentiles (0..100)
    tmpl_means = f[metric_cols].mean().astype(float)
    tmpl_vec_pct = {}
    for m in metric_cols:
        if ref_tmpl.empty:
            tmpl_vec_pct[m] = 50.0
        else:
            tmpl_vec_pct[m] = _percentile_of_value(ref_tmpl[m], float(tmpl_means[m])) * 100.0
    tmpl_vec_pct = pd.Series(tmpl_vec_pct)

    # ---- Candidate pool ----
    base_pool = build_base_pool()
    pool = base_pool.copy()
    pool = pool[pool["Position"].str.upper().str.startswith("CF")]
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))].copy()

    for c in feats:
        pool[c] = pd.to_numeric(pool.get(c), errors="coerce")
    pool = pool.dropna(subset=feats)

    # compute metrics (raw)
    pool["Opportunities"]      = 0.7*pool['Touches in box per 90'] + 0.3*pool['xG per 90']
    pool["Ball Carrying"]      = 0.65*pool['Dribbles per 90'] + 0.35*pool['Progressive runs per 90']
    pool["Aerial Requirement"] = pool['Aerial duels per 90'] * pool['Aerial duels won, %'] / 100.0
    pool["Passing Volume"]     = pool['Passes per 90']
    pool["Goal Output"]        = pool['Non-penalty goals per 90']
    pool["Retention"]          = pool['Accurate passes, %']

    # attach per-league percentiles for each role metric (0..100)
    # reference distribution = df (same league, same role, minutes within top-bar minutes range)
    # We do this by computing metrics on df and then percentile lookups per league.
    ref_all = df.copy()
    ref_all["Minutes played"] = pd.to_numeric(ref_all.get("Minutes played"), errors="coerce")
    ref_all = ref_all[ref_all["Minutes played"].between(min_minutes, max_minutes)]
    ref_all = ref_all[ref_all["Position"].apply(lambda p: pos_ok(str(p)))].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all.get(c), errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    if not ref_all.empty:
        ref_all["Opportunities"]      = 0.7*ref_all['Touches in box per 90'] + 0.3*ref_all['xG per 90']
        ref_all["Ball Carrying"]      = 0.65*ref_all['Dribbles per 90'] + 0.35*ref_all['Progressive runs per 90']
        ref_all["Aerial Requirement"] = ref_all['Aerial duels per 90'] * ref_all['Aerial duels won, %'] / 100.0
        ref_all["Passing Volume"]     = ref_all['Passes per 90']
        ref_all["Goal Output"]        = ref_all['Non-penalty goals per 90']
        ref_all["Retention"]          = ref_all['Accurate passes, %']
        ref_all = ref_all.dropna(subset=metric_cols)

    # compute pool metric percentiles vs each player's OWN league distribution
    for m in metric_cols:
        pool[f"{m} %ile"] = 50.0

    for lg in sorted(pool["League"].dropna().astype(str).unique()):
        idx = pool["League"].astype(str) == str(lg)
        ref_lg = ref_all[ref_all["League"].astype(str) == str(lg)].copy()
        if ref_lg.empty:
            continue
        for m in metric_cols:
            series = pd.to_numeric(ref_lg[m], errors="coerce").dropna()
            if series.empty:
                continue
            pool.loc[idx, f"{m} %ile"] = pd.to_numeric(pool.loc[idx, m], errors="coerce").map(
                lambda v: _percentile_of_value(series, v) * 100.0
            )

    # ---- Distance now uses PERCENTILES (0..100), not raw metrics ----
    pct_cols = [f"{m} %ile" for m in metric_cols]
    for m in metric_cols:
        pool[f"__tmpl__{m} %ile"] = float(tmpl_vec_pct[m])

    pool["BaseDist"] = pool.apply(
        lambda r: norm([float(r[f"{m} %ile"]) - float(r[f"__tmpl__{m} %ile"]) for m in metric_cols]),
        axis=1
    )

    ranked = _score_block(pool.copy())
    return ranked, "Strikers (CF)", tmpl_src


def compute_attackers(role_choice: str):
    feats = [
        'Accurate passes, %','xG per 90','Non-penalty goals per 90','Touches in box per 90',
        'xA per 90','Passes to penalty area per 90','Passes per 90',
        'Progressive passes per 90','Passes to final third per 90',
        'Dribbles per 90','Progressive runs per 90'
    ]

    def pos_ok(p):
        s = str(p).upper().strip()
        tokens = [t for t in re.split(r"[,/;]\s*|\s+", s) if t]
        if not tokens:
            return False
        t0 = tokens[0]
        if role_choice == "All":
            return t0 in {"RW","RWF","RAMF","LW","LWF","LAMF","AMF"}
        if role_choice == "Right Wingers":
            return t0 in {"RW","RWF","RAMF"}
        if role_choice == "Left Wingers":
            return t0 in {"LW","LWF","LAMF"}
        if role_choice == "Attacking Midfielders":
            return t0 == "AMF"
        return False

    tmpl_src = _template_rows_for_role(pos_ok).dropna(subset=feats)
    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (Attackers)", ["— Select —"] + players, index=0, key="att_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]

    if tmpl_src.empty:
        st.error("No attackers found for template conditions.")
        st.stop()

    # ---- Template metrics ----
    f = tmpl_src.copy()
    f["Retention Style"]    = f['Accurate passes, %']
    f["Goal Threat"]        = 0.4*f['xG per 90'] + 0.4*f['Non-penalty goals per 90'] + 0.2*f['Touches in box per 90']
    f["Creativity Threat"]  = 0.65*f['xA per 90'] + 0.35*f['Passes to penalty area per 90']
    f["Passing Volume"]     = f['Passes per 90']
    f["Deeper Playmaking"]  = 0.5*f['Progressive passes per 90'] + 0.5*f['Passes to final third per 90']
    f["Ball Carrying"]      = 0.6*f['Dribbles per 90'] + 0.4*f['Progressive runs per 90']
    metric_cols = ["Retention Style","Goal Threat","Creativity Threat","Passing Volume","Deeper Playmaking","Ball Carrying"]

    # template percentiles vs template league
    ref_tmpl = df[(df["League"].astype(str) == str(template_league))].copy()
    ref_tmpl["Minutes played"] = pd.to_numeric(ref_tmpl.get("Minutes played"), errors="coerce")
    ref_tmpl = ref_tmpl[ref_tmpl["Minutes played"].between(min_minutes, max_minutes)]
    ref_tmpl = ref_tmpl[ref_tmpl["Position"].apply(lambda p: pos_ok(str(p)))].copy()
    for c in feats:
        ref_tmpl[c] = pd.to_numeric(ref_tmpl.get(c), errors="coerce")
    ref_tmpl = ref_tmpl.dropna(subset=feats)

    if not ref_tmpl.empty:
        ref_tmpl["Retention Style"]   = ref_tmpl['Accurate passes, %']
        ref_tmpl["Goal Threat"]       = 0.4*ref_tmpl['xG per 90'] + 0.4*ref_tmpl['Non-penalty goals per 90'] + 0.2*ref_tmpl['Touches in box per 90']
        ref_tmpl["Creativity Threat"] = 0.65*ref_tmpl['xA per 90'] + 0.35*ref_tmpl['Passes to penalty area per 90']
        ref_tmpl["Passing Volume"]    = ref_tmpl['Passes per 90']
        ref_tmpl["Deeper Playmaking"] = 0.5*ref_tmpl['Progressive passes per 90'] + 0.5*ref_tmpl['Passes to final third per 90']
        ref_tmpl["Ball Carrying"]     = 0.6*ref_tmpl['Dribbles per 90'] + 0.4*ref_tmpl['Progressive runs per 90']
        ref_tmpl = ref_tmpl.dropna(subset=metric_cols)

    tmpl_means = f[metric_cols].mean().astype(float)
    tmpl_vec_pct = {}
    for m in metric_cols:
        tmpl_vec_pct[m] = 50.0 if ref_tmpl.empty else _percentile_of_value(ref_tmpl[m], float(tmpl_means[m])) * 100.0
    tmpl_vec_pct = pd.Series(tmpl_vec_pct)

    # ---- Candidate pool ----
    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]

    for c in feats:
        pool[c] = pd.to_numeric(pool.get(c), errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Retention Style"]   = pool['Accurate passes, %']
    pool["Goal Threat"]       = 0.4*pool['xG per 90'] + 0.4*pool['Non-penalty goals per 90'] + 0.2*pool['Touches in box per 90']
    pool["Creativity Threat"] = 0.65*pool['xA per 90'] + 0.35*pool['Passes to penalty area per 90']
    pool["Passing Volume"]    = pool['Passes per 90']
    pool["Deeper Playmaking"] = 0.5*pool['Progressive passes per 90'] + 0.5*pool['Passes to final third per 90']
    pool["Ball Carrying"]     = 0.6*pool['Dribbles per 90'] + 0.4*pool['Progressive runs per 90']

    # reference df for percentiles
    ref_all = df.copy()
    ref_all["Minutes played"] = pd.to_numeric(ref_all.get("Minutes played"), errors="coerce")
    ref_all = ref_all[ref_all["Minutes played"].between(min_minutes, max_minutes)]
    ref_all = ref_all[ref_all["Position"].apply(lambda p: pos_ok(str(p)))].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all.get(c), errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    if not ref_all.empty:
        ref_all["Retention Style"]   = ref_all['Accurate passes, %']
        ref_all["Goal Threat"]       = 0.4*ref_all['xG per 90'] + 0.4*ref_all['Non-penalty goals per 90'] + 0.2*ref_all['Touches in box per 90']
        ref_all["Creativity Threat"] = 0.65*ref_all['xA per 90'] + 0.35*ref_all['Passes to penalty area per 90']
        ref_all["Passing Volume"]    = ref_all['Passes per 90']
        ref_all["Deeper Playmaking"] = 0.5*ref_all['Progressive passes per 90'] + 0.5*ref_all['Passes to final third per 90']
        ref_all["Ball Carrying"]     = 0.6*ref_all['Dribbles per 90'] + 0.4*ref_all['Progressive runs per 90']
        ref_all = ref_all.dropna(subset=metric_cols)

    for m in metric_cols:
        pool[f"{m} %ile"] = 50.0
        pool[f"__tmpl__{m} %ile"] = float(tmpl_vec_pct[m])

    for lg in sorted(pool["League"].dropna().astype(str).unique()):
        idx = pool["League"].astype(str) == str(lg)
        ref_lg = ref_all[ref_all["League"].astype(str) == str(lg)].copy()
        if ref_lg.empty:
            continue
        for m in metric_cols:
            series = pd.to_numeric(ref_lg[m], errors="coerce").dropna()
            if series.empty:
                continue
            pool.loc[idx, f"{m} %ile"] = pd.to_numeric(pool.loc[idx, m], errors="coerce").map(
                lambda v: _percentile_of_value(series, v) * 100.0
            )

    pool["BaseDist"] = pool.apply(
        lambda r: norm([float(r[f"{m} %ile"]) - float(r[f"__tmpl__{m} %ile"]) for m in metric_cols]),
        axis=1
    )

    ranked = _score_block(pool.copy())
    return ranked, f"Attackers ({role_choice})", tmpl_src


def compute_central_mid():
    feats = [
        'Passes per 90','Forward passes per 90',
        'Progressive passes per 90','Progressive runs per 90',
        'Defensive duels per 90','PAdj Interceptions',
        'Touches in box per 90','Shots per 90','Accurate passes, %'
    ]

    def pos_ok(p):
        s = str(p).strip().upper()
        return s.startswith(("DMF","CMF","LCMF","RCMF","LDMF","RDMF"))

    tmpl_src = _template_rows_for_role(pos_ok).dropna(subset=feats)
    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (CM)", ["— Select —"] + players, index=0, key="cm_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]

    if tmpl_src.empty:
        st.error("No central midfielders found for template conditions.")
        st.stop()

    f = tmpl_src.copy()
    f["Pass Verticality"]       = _safe_verticality(f['Forward passes per 90'], f['Passes per 90'])
    f["Progression Volume"]     = f['Progressive passes per 90'] + f['Progressive runs per 90']
    f["Attacking Contribution"] = f['Touches in box per 90'] + f['Shots per 90']
    f["Defensive Volume"]       = f['Defensive duels per 90']
    f["Interception Volume"]    = f['PAdj Interceptions']
    f["Retention"]              = f['Accurate passes, %']

    metric_cols = ["Passes per 90","Pass Verticality","Progression Volume","Defensive Volume","Interception Volume","Attacking Contribution","Retention"]

    # template percentiles vs template league
    ref_tmpl = df[(df["League"].astype(str) == str(template_league))].copy()
    ref_tmpl["Minutes played"] = pd.to_numeric(ref_tmpl.get("Minutes played"), errors="coerce")
    ref_tmpl = ref_tmpl[ref_tmpl["Minutes played"].between(min_minutes, max_minutes)]
    ref_tmpl = ref_tmpl[ref_tmpl["Position"].apply(lambda p: pos_ok(str(p)))].copy()
    for c in feats:
        ref_tmpl[c] = pd.to_numeric(ref_tmpl.get(c), errors="coerce")
    ref_tmpl = ref_tmpl.dropna(subset=feats)

    if not ref_tmpl.empty:
        ref_tmpl["Pass Verticality"]       = _safe_verticality(ref_tmpl['Forward passes per 90'], ref_tmpl['Passes per 90'])
        ref_tmpl["Progression Volume"]     = ref_tmpl['Progressive passes per 90'] + ref_tmpl['Progressive runs per 90']
        ref_tmpl["Attacking Contribution"] = ref_tmpl['Touches in box per 90'] + ref_tmpl['Shots per 90']
        ref_tmpl["Defensive Volume"]       = ref_tmpl['Defensive duels per 90']
        ref_tmpl["Interception Volume"]    = ref_tmpl['PAdj Interceptions']
        ref_tmpl["Retention"]              = ref_tmpl['Accurate passes, %']
        ref_tmpl = ref_tmpl.dropna(subset=metric_cols)

    tmpl_means = f[metric_cols].mean().astype(float)
    tmpl_vec_pct = {}
    for m in metric_cols:
        tmpl_vec_pct[m] = 50.0 if ref_tmpl.empty else _percentile_of_value(ref_tmpl[m], float(tmpl_means[m])) * 100.0
    tmpl_vec_pct = pd.Series(tmpl_vec_pct)

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]

    for c in feats:
        pool[c] = pd.to_numeric(pool.get(c), errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Pass Verticality"]       = _safe_verticality(pool['Forward passes per 90'], pool['Passes per 90'])
    pool["Progression Volume"]     = pool['Progressive passes per 90'] + pool['Progressive runs per 90']
    pool["Attacking Contribution"] = pool['Touches in box per 90'] + pool['Shots per 90']
    pool["Defensive Volume"]       = pool['Defensive duels per 90']
    pool["Interception Volume"]    = pool['PAdj Interceptions']
    pool["Retention"]              = pool['Accurate passes, %']

    # reference all leagues for percentiles
    ref_all = df.copy()
    ref_all["Minutes played"] = pd.to_numeric(ref_all.get("Minutes played"), errors="coerce")
    ref_all = ref_all[ref_all["Minutes played"].between(min_minutes, max_minutes)]
    ref_all = ref_all[ref_all["Position"].apply(lambda p: pos_ok(str(p)))].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all.get(c), errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    if not ref_all.empty:
        ref_all["Pass Verticality"]       = _safe_verticality(ref_all['Forward passes per 90'], ref_all['Passes per 90'])
        ref_all["Progression Volume"]     = ref_all['Progressive passes per 90'] + ref_all['Progressive runs per 90']
        ref_all["Attacking Contribution"] = ref_all['Touches in box per 90'] + ref_all['Shots per 90']
        ref_all["Defensive Volume"]       = ref_all['Defensive duels per 90']
        ref_all["Interception Volume"]    = ref_all['PAdj Interceptions']
        ref_all["Retention"]              = ref_all['Accurate passes, %']
        ref_all = ref_all.dropna(subset=metric_cols)

    for m in metric_cols:
        pool[f"{m} %ile"] = 50.0
        pool[f"__tmpl__{m} %ile"] = float(tmpl_vec_pct[m])

    for lg in sorted(pool["League"].dropna().astype(str).unique()):
        idx = pool["League"].astype(str) == str(lg)
        ref_lg = ref_all[ref_all["League"].astype(str) == str(lg)].copy()
        if ref_lg.empty:
            continue
        for m in metric_cols:
            series = pd.to_numeric(ref_lg[m], errors="coerce").dropna()
            if series.empty:
                continue
            pool.loc[idx, f"{m} %ile"] = pd.to_numeric(pool.loc[idx, m], errors="coerce").map(
                lambda v: _percentile_of_value(series, v) * 100.0
            )

    pool["BaseDist"] = pool.apply(
        lambda r: norm([float(r[f"{m} %ile"]) - float(r[f"__tmpl__{m} %ile"]) for m in metric_cols]),
        axis=1
    )

    ranked = _score_block(pool.copy())
    return ranked, "Central Midfield", tmpl_src


def compute_fullbacks(role_choice: str):
    feats = [
        'Passes per 90','Forward passes per 90',
        'Progressive passes per 90','Progressive runs per 90',
        'Defensive duels per 90','PAdj Interceptions','Aerial duels per 90',
        'xA per 90','Crosses per 90','Touches in box per 90',
        'Shots per 90','Passes to penalty area per 90',
        'Accurate passes, %'
    ]

    def pos_ok(p):
        s = str(p).strip().upper()
        if role_choice == "Right Backs":
            prefixes = ("RB","RWB")
        elif role_choice == "Left Backs":
            prefixes = ("LB","LWB")
        else:
            prefixes = ("RB","RWB","LB","LWB")
        return any(s.startswith(px) for px in prefixes)

    tmpl_src = _template_rows_for_role(pos_ok).dropna(subset=feats)
    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (FB)", ["— Select —"] + players, index=0, key="fb_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]

    if tmpl_src.empty:
        st.error("No fullbacks found for template conditions.")
        st.stop()

    f = tmpl_src.copy()
    f["Pass Verticality"]        = _safe_verticality(f['Forward passes per 90'], f['Passes per 90'])
    f["Progression Volume"]      = f['Progressive passes per 90'] + f['Progressive runs per 90']
    f["Attacking Contribution"]  = 0.4*f['xA per 90'] + 0.2*f['Crosses per 90'] + 0.2*f['Touches in box per 90'] + 0.1*f['Shots per 90'] + 0.1*f['Passes to penalty area per 90']
    f["Defensive Volume"]        = 0.5*f['Defensive duels per 90'] + 0.3*f['PAdj Interceptions'] + 0.2*f['Aerial duels per 90']
    f["Retention"]               = f['Accurate passes, %']

    metric_cols = ["Passes per 90","Pass Verticality","Progression Volume","Attacking Contribution","Defensive Volume","Retention"]

    # template percentiles vs template league
    ref_tmpl = df[(df["League"].astype(str) == str(template_league))].copy()
    ref_tmpl["Minutes played"] = pd.to_numeric(ref_tmpl.get("Minutes played"), errors="coerce")
    ref_tmpl = ref_tmpl[ref_tmpl["Minutes played"].between(min_minutes, max_minutes)]
    ref_tmpl = ref_tmpl[ref_tmpl["Position"].apply(lambda p: pos_ok(str(p)))].copy()
    for c in feats:
        ref_tmpl[c] = pd.to_numeric(ref_tmpl.get(c), errors="coerce")
    ref_tmpl = ref_tmpl.dropna(subset=feats)

    if not ref_tmpl.empty:
        ref_tmpl["Pass Verticality"]       = _safe_verticality(ref_tmpl['Forward passes per 90'], ref_tmpl['Passes per 90'])
        ref_tmpl["Progression Volume"]     = ref_tmpl['Progressive passes per 90'] + ref_tmpl['Progressive runs per 90']
        ref_tmpl["Attacking Contribution"] = 0.4*ref_tmpl['xA per 90'] + 0.2*ref_tmpl['Crosses per 90'] + 0.2*ref_tmpl['Touches in box per 90'] + 0.1*ref_tmpl['Shots per 90'] + 0.1*ref_tmpl['Passes to penalty area per 90']
        ref_tmpl["Defensive Volume"]       = 0.5*ref_tmpl['Defensive duels per 90'] + 0.3*ref_tmpl['PAdj Interceptions'] + 0.2*ref_tmpl['Aerial duels per 90']
        ref_tmpl["Retention"]              = ref_tmpl['Accurate passes, %']
        ref_tmpl = ref_tmpl.dropna(subset=metric_cols)

    tmpl_means = f[metric_cols].mean().astype(float)
    tmpl_vec_pct = {}
    for m in metric_cols:
        tmpl_vec_pct[m] = 50.0 if ref_tmpl.empty else _percentile_of_value(ref_tmpl[m], float(tmpl_means[m])) * 100.0
    tmpl_vec_pct = pd.Series(tmpl_vec_pct)

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]

    for c in feats:
        pool[c] = pd.to_numeric(pool.get(c), errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Pass Verticality"]        = _safe_verticality(pool['Forward passes per 90'], pool['Passes per 90'])
    pool["Progression Volume"]      = pool['Progressive passes per 90'] + pool['Progressive runs per 90']
    pool["Attacking Contribution"]  = 0.4*pool['xA per 90'] + 0.2*pool['Crosses per 90'] + 0.2*pool['Touches in box per 90'] + 0.1*pool['Shots per 90'] + 0.1*pool['Passes to penalty area per 90']
    pool["Defensive Volume"]        = 0.5*pool['Defensive duels per 90'] + 0.3*pool['PAdj Interceptions'] + 0.2*pool['Aerial duels per 90']
    pool["Retention"]               = pool['Accurate passes, %']

    ref_all = df.copy()
    ref_all["Minutes played"] = pd.to_numeric(ref_all.get("Minutes played"), errors="coerce")
    ref_all = ref_all[ref_all["Minutes played"].between(min_minutes, max_minutes)]
    ref_all = ref_all[ref_all["Position"].apply(lambda p: pos_ok(str(p)))].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all.get(c), errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    if not ref_all.empty:
        ref_all["Pass Verticality"]       = _safe_verticality(ref_all['Forward passes per 90'], ref_all['Passes per 90'])
        ref_all["Progression Volume"]     = ref_all['Progressive passes per 90'] + ref_all['Progressive runs per 90']
        ref_all["Attacking Contribution"] = 0.4*ref_all['xA per 90'] + 0.2*ref_all['Crosses per 90'] + 0.2*ref_all['Touches in box per 90'] + 0.1*ref_all['Shots per 90'] + 0.1*ref_all['Passes to penalty area per 90']
        ref_all["Defensive Volume"]       = 0.5*ref_all['Defensive duels per 90'] + 0.3*ref_all['PAdj Interceptions'] + 0.2*ref_all['Aerial duels per 90']
        ref_all["Retention"]              = ref_all['Accurate passes, %']
        ref_all = ref_all.dropna(subset=metric_cols)

    for m in metric_cols:
        pool[f"{m} %ile"] = 50.0
        pool[f"__tmpl__{m} %ile"] = float(tmpl_vec_pct[m])

    for lg in sorted(pool["League"].dropna().astype(str).unique()):
        idx = pool["League"].astype(str) == str(lg)
        ref_lg = ref_all[ref_all["League"].astype(str) == str(lg)].copy()
        if ref_lg.empty:
            continue
        for m in metric_cols:
            series = pd.to_numeric(ref_lg[m], errors="coerce").dropna()
            if series.empty:
                continue
            pool.loc[idx, f"{m} %ile"] = pd.to_numeric(pool.loc[idx, m], errors="coerce").map(
                lambda v: _percentile_of_value(series, v) * 100.0
            )

    pool["BaseDist"] = pool.apply(
        lambda r: norm([float(r[f"{m} %ile"]) - float(r[f"__tmpl__{m} %ile"]) for m in metric_cols]),
        axis=1
    )

    ranked = _score_block(pool.copy())
    return ranked, f"Fullbacks ({role_choice})", tmpl_src


def compute_center_backs():
    feats = [
        'Aerial duels per 90','Defensive duels per 90',
        'Passes per 90','Forward passes per 90',
        'Progressive passes per 90','Progressive runs per 90',
        'PAdj Interceptions','Shots blocked per 90'
    ]

    def pos_ok(p):
        s = str(p).strip().upper()
        return s.startswith(("CB","RCB","LCB"))

    tmpl_src = _template_rows_for_role(pos_ok).dropna(subset=feats)
    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (CB)", ["— Select —"] + players, index=0, key="cb_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]

    if tmpl_src.empty:
        st.error("No centre-backs found for template conditions.")
        st.stop()

    f = tmpl_src.copy()
    f["Passing Verticality"] = _safe_verticality(f['Forward passes per 90'], f['Passes per 90'])
    f["Passing Volume"]      = f['Passes per 90']
    f["Positional Demand"]   = f['PAdj Interceptions'] + f['Shots blocked per 90']
    f["Progression Volume"]  = f['Progressive passes per 90'] + f['Progressive runs per 90']

    metric_cols = ["Aerial duels per 90","Defensive duels per 90","Positional Demand","Passing Volume","Passing Verticality","Progression Volume"]

    # template percentiles vs template league
    ref_tmpl = df[(df["League"].astype(str) == str(template_league))].copy()
    ref_tmpl["Minutes played"] = pd.to_numeric(ref_tmpl.get("Minutes played"), errors="coerce")
    ref_tmpl = ref_tmpl[ref_tmpl["Minutes played"].between(min_minutes, max_minutes)]
    ref_tmpl = ref_tmpl[ref_tmpl["Position"].apply(lambda p: pos_ok(str(p)))].copy()
    for c in feats:
        ref_tmpl[c] = pd.to_numeric(ref_tmpl.get(c), errors="coerce")
    ref_tmpl = ref_tmpl.dropna(subset=feats)

    if not ref_tmpl.empty:
        ref_tmpl["Passing Verticality"] = _safe_verticality(ref_tmpl['Forward passes per 90'], ref_tmpl['Passes per 90'])
        ref_tmpl["Passing Volume"]      = ref_tmpl['Passes per 90']
        ref_tmpl["Positional Demand"]   = ref_tmpl['PAdj Interceptions'] + ref_tmpl['Shots blocked per 90']
        ref_tmpl["Progression Volume"]  = ref_tmpl['Progressive passes per 90'] + ref_tmpl['Progressive runs per 90']
        ref_tmpl = ref_tmpl.dropna(subset=metric_cols)

    tmpl_means = f[metric_cols].mean().astype(float)
    tmpl_vec_pct = {}
    for m in metric_cols:
        tmpl_vec_pct[m] = 50.0 if ref_tmpl.empty else _percentile_of_value(ref_tmpl[m], float(tmpl_means[m])) * 100.0
    tmpl_vec_pct = pd.Series(tmpl_vec_pct)

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]

    for c in feats:
        pool[c] = pd.to_numeric(pool.get(c), errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Passing Verticality"] = _safe_verticality(pool['Forward passes per 90'], pool['Passes per 90'])
    pool["Passing Volume"]      = pool['Passes per 90']
    pool["Positional Demand"]   = pool['PAdj Interceptions'] + pool['Shots blocked per 90']
    pool["Progression Volume"]  = pool['Progressive passes per 90'] + pool['Progressive runs per 90']

    ref_all = df.copy()
    ref_all["Minutes played"] = pd.to_numeric(ref_all.get("Minutes played"), errors="coerce")
    ref_all = ref_all[ref_all["Minutes played"].between(min_minutes, max_minutes)]
    ref_all = ref_all[ref_all["Position"].apply(lambda p: pos_ok(str(p)))].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all.get(c), errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    if not ref_all.empty:
        ref_all["Passing Verticality"] = _safe_verticality(ref_all['Forward passes per 90'], ref_all['Passes per 90'])
        ref_all["Passing Volume"]      = ref_all['Passes per 90']
        ref_all["Positional Demand"]   = ref_all['PAdj Interceptions'] + ref_all['Shots blocked per 90']
        ref_all["Progression Volume"]  = ref_all['Progressive passes per 90'] + ref_all['Progressive runs per 90']
        ref_all = ref_all.dropna(subset=metric_cols)

    for m in metric_cols:
        pool[f"{m} %ile"] = 50.0
        pool[f"__tmpl__{m} %ile"] = float(tmpl_vec_pct[m])

    for lg in sorted(pool["League"].dropna().astype(str).unique()):
        idx = pool["League"].astype(str) == str(lg)
        ref_lg = ref_all[ref_all["League"].astype(str) == str(lg)].copy()
        if ref_lg.empty:
            continue
        for m in metric_cols:
            series = pd.to_numeric(ref_lg[m], errors="coerce").dropna()
            if series.empty:
                continue
            pool.loc[idx, f"{m} %ile"] = pd.to_numeric(pool.loc[idx, m], errors="coerce").map(
                lambda v: _percentile_of_value(series, v) * 100.0
            )

    pool["BaseDist"] = pool.apply(
        lambda r: norm([float(r[f"{m} %ile"]) - float(r[f"__tmpl__{m} %ile"]) for m in metric_cols]),
        axis=1
    )

    ranked = _score_block(pool.copy())
    return ranked, "Center Backs", tmpl_src


# ========================= FOTMOB PHOTO + CREST =========================
# Provide a mapping file team_fotmob_urls.py with:
# FOTMOB_TEAM_URLS = {"Arsenal":"https://www.fotmob.com/teams/9825/overview/arsenal", ...}

def _fotmob_team_id_from_url(team_url: str) -> str:
    m = re.search(r"/teams/(\d+)/", str(team_url or ""))
    return m.group(1) if m else ""

def _fotmob_crest_url(team_url: str) -> str:
    tid = _fotmob_team_id_from_url(team_url)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png" if tid else ""

def _fotmob_team_squad(team_id: str) -> List[dict]:
    cache = st.session_state.setdefault("_fotmob_team_squad_cache", {})
    if team_id in cache:
        return cache[team_id] or []

    squad: List[dict] = []
    try:
        url = f"https://www.fotmob.com/api/teams?id={team_id}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            data = r.json() or {}
            raw = data.get("squad", None)

            if isinstance(raw, list):
                for sec in raw:
                    members = sec.get("members") or sec.get("players") or []
                    if isinstance(members, list):
                        squad.extend([m for m in members if isinstance(m, dict)])

            elif isinstance(raw, dict):
                for k in ("members", "players"):
                    members = raw.get(k)
                    if isinstance(members, list):
                        squad.extend([m for m in members if isinstance(m, dict)])
                nested = raw.get("squad")
                if isinstance(nested, list):
                    for sec in nested:
                        members = sec.get("members") or sec.get("players") or []
                        if isinstance(members, list):
                            squad.extend([m for m in members if isinstance(m, dict)])
    except Exception:
        squad = []

    cache[team_id] = squad
    return squad

def _slug_name(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    repl = {
        "ø": "o", "œ": "oe", "æ": "ae", "å": "a", "ä": "a", "ö": "o", "ü": "u",
        "ß": "ss", "ł": "l", "đ": "d", "ð": "d", "þ": "th", "ç": "c", "ş": "s",
        "ğ": "g", "ı": "i",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _player_surname(player: str) -> str:
    p = (player or "").strip()
    if not p:
        return ""
    if "," in p:
        return p.split(",", 1)[0].strip()
    parts = p.split()
    return parts[-1].strip() if parts else ""

PLACEHOLDER_IMG = "https://i.redd.it/43axcjdu59nd1.jpeg"

def resolve_player_photo(player: str, team: str, league: str) -> str:
    """
    Priority:
      1) session override (photo_map)
      2) try fotmob squad match -> playerimages/{id}.png
      3) placeholder
    """
    key_id = f"{player}|||{team}|||{league}"
    override = st.session_state.get("photo_map", {}).get(key_id, "")
    if override:
        return override

    team_url = ""
    try:
        from team_fotmob_urls import FOTMOB_TEAM_URLS
        team_url = (FOTMOB_TEAM_URLS.get(team) or "").strip()
    except Exception:
        team_url = ""

    tid = _fotmob_team_id_from_url(team_url)
    if not tid:
        return PLACEHOLDER_IMG

    squad = _fotmob_team_squad(tid)
    target_surname = _slug_name(_player_surname(player))
    target_full = _slug_name(player)

    best_id = ""

    # exact surname match first
    if target_surname:
        for m in squad:
            name = m.get("name") or m.get("playerName") or ""
            pid = m.get("id") or m.get("playerId") or m.get("primaryId") or ""
            if not pid:
                continue
            if _slug_name(_player_surname(name)) == target_surname:
                best_id = str(pid)
                if target_full and target_full in _slug_name(name):
                    break

    # exact full match fallback
    if not best_id and target_full:
        for m in squad:
            name = m.get("name") or m.get("playerName") or ""
            pid = m.get("id") or m.get("playerId") or m.get("primaryId") or ""
            if not pid:
                continue
            if target_full in _slug_name(name):
                best_id = str(pid)
                break

    # fuzzy surname fallback
    if not best_id and target_surname:
        best_score, best_pid = 0.0, ""
        for m in squad:
            name = m.get("name") or m.get("playerName") or ""
            pid = m.get("id") or m.get("playerId") or m.get("primaryId") or ""
            if not pid:
                continue
            sn = _slug_name(_player_surname(name))
            sc = _similar(sn, target_surname)
            if sc > best_score:
                best_score, best_pid = sc, str(pid)
        if best_score >= 0.86:
            best_id = best_pid

    if best_id and str(best_id).isdigit():
        return f"https://images.fotmob.com/image_resources/playerimages/{best_id}.png"

    return PLACEHOLDER_IMG

def resolve_team_crest(team: str, league: str) -> str:
    """
    Priority:
      1) session override (crest_map)
      2) fotmob teamlogo/{team_id}.png
      3) ""
    """
    crest_key = f"{team}|||{league}"
    override = st.session_state.get("crest_map", {}).get(crest_key, "")
    if override:
        return override

    team_url = ""
    try:
        from team_fotmob_urls import FOTMOB_TEAM_URLS
        team_url = (FOTMOB_TEAM_URLS.get(team) or "").strip()
    except Exception:
        team_url = ""

    return _fotmob_crest_url(team_url) if team_url else ""


# ========================= UI: TILE LAYOUT =========================

# Small helper: safe numeric
def _to_int(x, default=0):
    try:
        v = int(float(x))
        return v
    except Exception:
        return default

def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _fmt_mv(v):
    v = _to_float(v, np.nan)
    if not np.isfinite(v):
        return "—"
    if v >= 1_000_000:
        return f"€{v/1_000_000:.1f}m"
    if v >= 1_000:
        return f"€{v/1_000:.0f}k"
    return f"€{v:.0f}"

def _fmt_age(v):
    a = _to_int(v, 0)
    return "—" if a <= 0 else str(a)

def _tile_css():
    st.markdown(
        """
        <style>
        .tile{
            border-radius:14px;
            border:1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            padding:10px 10px 8px 10px;
            margin-bottom:12px;
        }
        .tile-top{
            display:flex; align-items:center; justify-content:space-between;
            margin-bottom:8px;
        }
        .crest{
            width:26px; height:26px; border-radius:6px;
            object-fit:contain; background:rgba(255,255,255,0.06);
            padding:2px;
        }
        .match{
            font-weight:800;
            font-size:14px;
            padding:4px 10px;
            border-radius:999px;
            border:1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.06);
            white-space:nowrap;
        }
        .photo{
            width:100%;
            aspect-ratio: 4/5;
            border-radius:14px;
            object-fit:cover;
            background: rgba(255,255,255,0.06);
        }
        .name{
            font-weight:800; font-size:15px; line-height:1.1;
            margin-top:8px;
        }
        .meta{
            opacity:0.85; font-size:12px; line-height:1.25;
            margin-top:4px;
        }
        .mini{
            opacity:0.8; font-size:11px; margin-top:6px;
            display:flex; gap:10px; flex-wrap:wrap;
        }
        .mini span{
            padding:2px 8px; border-radius:999px;
            background: rgba(255,255,255,0.05);
            border:1px solid rgba(255,255,255,0.10);
        }
        .override-box{
            margin-top:8px;
            border-top:1px solid rgba(255,255,255,0.06);
            padding-top:8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_tile_css()

def _apply_pool_extra_filters(p: pd.DataFrame) -> pd.DataFrame:
    """Age + market value filters (kept separate from build_base_pool for clarity)."""
    out = p.copy()
    if "Age" in out.columns:
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
        out = out[out["Age"].between(min_age, max_age)]
    if "Market value" in out.columns:
        out["Market value"] = pd.to_numeric(out["Market value"], errors="coerce")
        out = out[out["Market value"].between(pool_min_value, pool_max_value)]
    return out

def _render_photo_debug(player, team, league, url, crest_url):
    if not DEBUG_PHOTOS:
        return
    st.caption(f"DEBUG — {player} | {team} | {league}")
    st.code(url)
    if crest_url:
        st.code(crest_url)

def render_tiles(
    ranked: pd.DataFrame,
    role_title: str,
    tmpl_src: pd.DataFrame,
    *,
    show_template_table: bool = True,
    cols_per_row: int = 5,
):
    if ranked is None or ranked.empty:
        st.info("No candidates found for current filters.")
        return

    # Apply remaining pool filters (Age + MV), and take top_n
    ranked2 = ranked.copy()
    ranked2 = _apply_pool_extra_filters(ranked2)
    if ranked2.empty:
        st.info("No candidates after Age / Market Value filters.")
        return

    ranked2 = ranked2.head(int(top_n)).reset_index(drop=True)

    # Header line + template players used
    left, mid, right = st.columns([1.6, 1.0, 1.0])
    with left:
        st.subheader(role_title)
        st.caption(f"Showing **Top {min(int(top_n), len(ranked2))}** by Role Fit Score (Match%).")
    with mid:
        if "Role Fit Score" in ranked2.columns:
            st.metric("Best match", f"{ranked2.loc[0,'Role Fit Score']:.1f}%")
        else:
            st.metric("Best match", "—")
    with right:
        st.metric("Candidates", str(len(ranked2)))

    if show_template_table:
        with st.expander("Show template players used", expanded=False):
            render_template_players_used(role_title, tmpl_src)

    # Build rows of tiles
    ncols = max(2, min(6, int(cols_per_row)))
    rows = int(math.ceil(len(ranked2) / ncols))

    for r in range(rows):
        row = st.columns(ncols)
        for c in range(ncols):
            i = r * ncols + c
            if i >= len(ranked2):
                row[c].empty()
                continue

            player = str(ranked2.loc[i, "Player"]) if "Player" in ranked2.columns else "—"
            team = str(ranked2.loc[i, "Team"]) if "Team" in ranked2.columns else "—"
            league = str(ranked2.loc[i, "League"]) if "League" in ranked2.columns else "—"
            pos = str(ranked2.loc[i, "Position"]) if "Position" in ranked2.columns else "—"
            age = ranked2.loc[i, "Age"] if "Age" in ranked2.columns else np.nan
            mins = ranked2.loc[i, "Minutes played"] if "Minutes played" in ranked2.columns else np.nan
            mv = ranked2.loc[i, "Market value"] if "Market value" in ranked2.columns else np.nan

            match = ranked2.loc[i, "Role Fit Score"] if "Role Fit Score" in ranked2.columns else np.nan
            match_txt = "—" if not np.isfinite(_to_float(match, np.nan)) else f"{_to_float(match):.0f}%"

            photo_url = resolve_player_photo(player, team, league)
            crest_url = resolve_team_crest(team, league)

            _render_photo_debug(player, team, league, photo_url, crest_url)

            key_id = f"{player}|||{team}|||{league}"
            crest_key = f"{team}|||{league}"

            with row[c]:
                # tile top strip (crest + match)
                crest_html = f'<img class="crest" src="{crest_url}" />' if crest_url else '<div style="width:26px;height:26px;"></div>'
                st.markdown(
                    f"""
                    <div class="tile">
                      <div class="tile-top">
                        {crest_html}
                        <div class="match">{match_txt}</div>
                      </div>
                      <img class="photo" src="{photo_url}" />
                      <div class="name">{player}</div>
                      <div class="meta">{team} • {league}</div>
                      <div class="mini">
                        <span>{pos}</span>
                        <span>Age { _fmt_age(age) }</span>
                        <span>{_to_int(mins)} mins</span>
                        <span>{_fmt_mv(mv)}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("Overrides", expanded=False):
                    st.markdown('<div class="override-box">', unsafe_allow_html=True)

                    # Photo override
                    current_photo_override = st.session_state.get("photo_map", {}).get(key_id, "")
                    new_photo = st.text_input(
                        "Player photo URL (override)",
                        value=current_photo_override,
                        key=f"photo_override__{role_title}__{i}",
                        placeholder="https://… (leave blank to use FotMob/placeholder)",
                    )
                    b1, b2 = st.columns([1, 1])
                    with b1:
                        if st.button("Save photo override", key=f"save_photo__{role_title}__{i}"):
                            if new_photo.strip():
                                st.session_state["photo_map"][key_id] = new_photo.strip()
                            else:
                                st.session_state["photo_map"].pop(key_id, None)
                            st.success("Saved.")
                    with b2:
                        if st.button("Clear photo override", key=f"clear_photo__{role_title}__{i}"):
                            st.session_state["photo_map"].pop(key_id, None)
                            st.success("Cleared.")

                    st.markdown("---")

                    # Crest override
                    current_crest_override = st.session_state.get("crest_map", {}).get(crest_key, "")
                    new_crest = st.text_input(
                        "Team crest URL (override)",
                        value=current_crest_override,
                        key=f"crest_override__{role_title}__{i}",
                        placeholder="https://… (leave blank to use FotMob crest)",
                    )
                    b3, b4 = st.columns([1, 1])
                    with b3:
                        if st.button("Save crest override", key=f"save_crest__{role_title}__{i}"):
                            if new_crest.strip():
                                st.session_state["crest_map"][crest_key] = new_crest.strip()
                            else:
                                st.session_state["crest_map"].pop(crest_key, None)
                            st.success("Saved.")
                    with b4:
                        if st.button("Clear crest override", key=f"clear_crest__{role_title}__{i}"):
                            st.session_state["crest_map"].pop(crest_key, None)
                            st.success("Cleared.")

                    st.markdown('</div>', unsafe_allow_html=True)


# ========================= MAIN UI: TABS =========================
st.markdown("---")
st.header("📌 Shortlist")

# Candidate pool quick info
with st.expander("Show candidate pool diagnostics", expanded=False):
    pool0 = build_base_pool()
    pool0 = _apply_pool_extra_filters(pool0)
    st.write(f"Pool rows after filters: **{len(pool0)}**")
    show_cols = [c for c in ["Player","Team","League","Position","Age","Minutes played","Market value"] if c in pool0.columns]
    if show_cols:
        st.dataframe(pool0[show_cols].head(200), use_container_width=True)

tabs = st.tabs(["Strikers", "Attackers", "Central Mid", "Fullbacks", "Center Backs"])

# ---- Strikers ----
with tabs[0]:
    ranked, title, tmpl_src = compute_strikers()
    render_tiles(ranked, title, tmpl_src, cols_per_row=5)

# ---- Attackers ----
with tabs[1]:
    role_choice = st.selectbox(
        "Attacker subtype",
        ["All", "Right Wingers", "Left Wingers", "Attacking Midfielders"],
        index=0,
        key="att_role_choice",
    )
    ranked, title, tmpl_src = compute_attackers(role_choice)
    render_tiles(ranked, title, tmpl_src, cols_per_row=5)

# ---- Central Mid ----
with tabs[2]:
    ranked, title, tmpl_src = compute_central_mid()
    render_tiles(ranked, title, tmpl_src, cols_per_row=5)

# ---- Fullbacks ----
with tabs[3]:
    role_choice = st.selectbox(
        "Fullback subtype",
        ["All", "Right Backs", "Left Backs"],
        index=0,
        key="fb_role_choice",
    )
    ranked, title, tmpl_src = compute_fullbacks(role_choice)
    render_tiles(ranked, title, tmpl_src, cols_per_row=5)

# ---- Center Backs ----
with tabs[4]:
    ranked, title, tmpl_src = compute_center_backs()
    render_tiles(ranked, title, tmpl_src, cols_per_row=5)

# ========================= FOOTER =========================
st.markdown("---")
st.caption(
    "Notes: Photos/crests come from FotMob where possible via team squad lookup + teamlogo. "
    "If a team is missing in team_fotmob_urls.py, you'll see placeholders until you add the URL or override."



)# ============================ CLUB TOOL — ONE-PAGER (FULL) + STYLES / STRENGTHS / WEAKNESSES ============================
# Paste this WHOLE block into your Club Tool page where you want the one-pager section.
#
# ✅ Percentiles are computed vs: SAME LEAGUE + SAME POSITION-GROUP pool (unless "{metric} Percentile" exists)
# ✅ Target Man CF is DISPLAYED in roles, but NOT used for the big badge score beside the name
# ✅ Header layout: Player PHOTO then NAME then BIG BADGE (beside name). Crest on far-right
# ✅ Reduced top gap: meta + strengths/weaknesses/styles + roles sit tighter, panels start higher
# ✅ Removed the badge number from the info/meta line (no leading "99" etc)
# ✅ Adds: Strengths (>=HI), Weaknesses (<=LO), Styles (>=STYLE_T) from STYLE_MAP per position group
# ✅ Adds: (NEW) Best-role label AFTER crest (first 2 words + first position token)
# ✅ Adds: (NEW) Extra roles (display/label-eligible) that do NOT affect badge score
#
# Assumes you already have:
# - df (DataFrame)
# - streamlit as st, numpy as np, pandas as pd, re, requests available (if not, this block imports what it needs)
# - Optional: resolve_player_photo(player, team, league) and resolve_team_crest(team, league)
# - Optional: LEAGUE_STRENGTHS dict

from io import BytesIO
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests

st.markdown("---")
st.header("🧾 One-pager (Club Tool)")

# -------------------- THRESHOLDS --------------------
HI, LO, STYLE_T = 70, 30, 65

# -------------------- STYLE MAPS (BY POSITION GROUP KEY) --------------------
# Keys in STYLE_MAP correspond to raw metric names in your dataset.
STYLE_MAPS = {
    "CB": {
        "Defensive duels per 90": {"style": "Front Footed", "sw": "Defensive Duel Attempts"},
        "Aerial duels won, %": {"style": "Aerially Dominant", "sw": "Aerial Duels"},
        "Defensive duels won, %": {"style": None, "sw": "Tackling %"},
        "Long Passes per 90": {"style": "Long Passer", "sw": None},
        "PAdj Interceptions": {"style": "Cuts out opposition attacks", "sw": "Interceptions"},
        "Accurate forward passes, %": {"style": None, "sw": "Forward Passing Accuracy"},
        "Dribbles per 90": {"style": "Carries out from the back", "sw": "Dribble Volume"},
        "Successful dribbles, %": {"style": None, "sw": "Dribbling Efficiency"},
        "Progressive runs per 90": {"style": "Gets team up the pitch via carries", "sw": "Progressive Runs"},
        "Passes per 90": {"style": "Ball player", "sw": "Passing Involvement"},
        "Accurate passes, %": {"style": "Technical", "sw": "Passing Retention"},
        "Progressive passes per 90": {"style": "Progressive Passer", "sw": "Ball progression via passes"},
        "Shots blocked per 90": {"style": "Stopper", "sw": None},
    },

    "FB": {
        "Defensive duels per 90": {"style": "Ball Winner", "sw": "Defensive Duel Attempts"},
        "Aerial duels won, %": {"style": None, "sw": "Aerial Duels"},
        "Defensive duels won, %": {"style": None, "sw": "Tackling %"},
        "Long passes per 90": {"style": "Long Passer", "sw": None},
        "xG per 90": {"style": None, "sw": "Goal Threat"},
        "Shots per 90": {"style": "Takes many shots", "sw": None},
        "PAdj Interceptions": {"style": "Cuts out opposition attacks", "sw": "Defensive positioning"},
        "Accurate forward passes, %": {"style": None, "sw": "Forward Passing Accuracy"},
        "Dribbles per 90": {"style": "Dribbler", "sw": "Dribble Volume"},
        "Successful dribbles, %": {"style": None, "sw": "Dribbling Efficiency"},
        "Touches in box per 90": {"style": "Busy in the penalty box", "sw": "Penalty-box Coverage"},
        "Progressive runs per 90": {"style": "Gets team up the pitch via carries", "sw": "Progressive Runs"},
        "Passes per 90": {"style": "Involved in build-up", "sw": "Passing Involvement"},
        "Accurate passes, %": {"style": "Secure Passer", "sw": "Passing Retention"},
        "xA per 90": {"style": "Creates goal scoring chances", "sw": "Creativity"},
        "Passes to penalty area per 90": {"style": "Creates openings", "sw": "Passes to Penalty Area"},
        "Deep completions per 90": {"style": "Gets ball into the box", "sw": None},
        "Progressive passes per 90": {"style": "Build up Passer", "sw": "Ball progression via passes"},
        "Smart passes per 90": {"style": "Attempts through balls", "sw": None},
    },

    "CM": {
        "Defensive duels per 90": {"style": "Ball Winner", "sw": "Defensive Duel Attempts"},
        "Aerial duels won, %": {"style": None, "sw": "Aerial Duels"},
        "Defensive duels won, %": {"style": None, "sw": "Tackling %"},
        "Long Passes per 90": {"style": "Long Passer", "sw": None},
        "Non-penalty goals per 90": {"style": None, "sw": "Scoring Goals"},
        "xG per 90": {"style": "Box-Crasher", "sw": "Goal Threat"},
        "Shots per 90": {"style": "Takes many shots", "sw": None},
        "PAdj Interceptions": {"style": "Cuts out opposition attacks", "sw": "Defensive positioning"},
        "Accurate forward passes, %": {"style": None, "sw": "Forward Passing Accuracy"},
        "Dribbles per 90": {"style": "Dribbler", "sw": "Dribble Volume"},
        "Successful dribbles, %": {"style": None, "sw": "Dribbling Efficiency"},
        "Touches in box per 90": {"style": "Busy in the penalty box", "sw": "Penalty-box Coverage"},
        "Progressive runs per 90": {"style": "Gets team up the pitch via carries", "sw": "Progressive Runs"},
        "Passes per 90": {"style": "Involved in build-up", "sw": "Passing Involvement"},
        "Accurate passes, %": {"style": "Controller", "sw": "Passing Retention"},
        "xA per 90": {"style": "Creates goal scoring chances", "sw": "Creativity"},
        "Passes to penalty area per 90": {"style": "Advanced Playmaker", "sw": "Passes to Penalty Area"},
        "Deep completions per 90": {"style": "Gets ball into the box", "sw": None},
        "Progressive passes per 90": {"style": "Deep Playmaker", "sw": "Ball progression via passes"},
        "Smart passes per 90": {"style": "Attempts through balls", "sw": None},
    },

    "ATT": {
        "Defensive duels per 90": {"style": "High work rate", "sw": "Defensive Duels"},
        "Aerial duels won, %": {"style": None, "sw": "Aerial Duels"},
        "Aerial duels per 90": {"style": "Long Reference Point", "sw": None},
        "Non-penalty goals per 90": {"style": None, "sw": "Scoring Goals"},
        "xG per 90": {"style": "Gets into good goal scoring positions", "sw": "Attacking Positioning"},
        "Shots per 90": {"style": "Takes many shots", "sw": "Shot Volume"},
        "Goal conversion, %": {"style": None, "sw": "Finishing"},
        "Crosses per 90": {"style": "Wide Creator", "sw": "Crossing"},
        "Dribbles per 90": {"style": "Dribbler", "sw": "Dribble Volume"},
        "Successful dribbles, %": {"style": None, "sw": "Dribbling Efficiency"},
        "Touches in box per 90": {"style": "Busy in the penalty box", "sw": "Penalty-box Coverage"},
        "Progressive runs per 90": {"style": "Gets team up the pitch via carries", "sw": "Progressive Runs"},
        "Passes per 90": {"style": "Involved in build-up", "sw": "Involvement"},
        "Accurate passes, %": {"style": None, "sw": "Retention"},
        "xA per 90": {"style": "Creates goal scoring chances", "sw": "Creativity"},
        "Passes to penalty area per 90": {"style": None, "sw": "Passes to Penalty Area"},
        "Deep completions per 90": {"style": "Gets ball into the box", "sw": None},
        "Progressive passes per 90": {"style": "Drops deep to build play", "sw": None},
        "Smart passes per 90": {"style": "Attempts through balls", "sw": None},
    },

    "CF": {
        "Defensive duels per 90": {"style": "High Work Rate", "sw": "Defensive Duel Attempts"},
        "Aerial duels won, %": {"style": None, "sw": "Aerial Duels"},
        "xG per 90": {"style": "Gets into good goalscoring positions", "sw": "Goal Threat"},
        "Shots per 90": {"style": "Takes many shots", "sw": "Shot Volume"},
        "Crosses per 90": {"style": "Moves into wide areas to create", "sw": None},
        "Dribbles per 90": {"style": "Dribbler", "sw": "Dribble Volume"},
        "Successful dribbles, %": {"style": None, "sw": "Dribbling Efficiency"},
        "Touches in box per 90": {"style": "Busy in the penalty box", "sw": "Penalty-box Coverage"},
        "Progressive runs per 90": {"style": "Gets team up the pitch via carries", "sw": "Progressive Runs"},
        "Passes per 90": {"style": "Involved in build-up", "sw": "Involvement"},
        "Accurate passes, %": {"style": None, "sw": "Passing Retention"},
        "xA per 90": {"style": "Creates goal scoring chances", "sw": "Creativity"},
        "Passes to penalty area per 90": {"style": "Creates openings", "sw": "Passes to Penalty Area"},
        "Deep completions per 90": {"style": "Gets ball into the box", "sw": None},
        "Goal conversion, %": {"style": None, "sw": "Finishing"},
        "Smart passes per 90": {"style": "Attempts through balls", "sw": None},
    },
}


# -------------------- ROLE BUCKETS (ALL ROLES BY POSITION) --------------------
ROLE_BUCKETS = {
    "CM": {
        "Deep Playmaker CM": {"metrics": {"Passes per 90": 1, "Accurate passes, %": 1, "Forward passes per 90": 2,
                                         "Accurate forward passes, %": 1.5, "Progressive passes per 90": 3,
                                         "Passes to final third per 90": 2.5, "Accurate long passes, %": 1}},
        "Advanced Playmaker CM": {"metrics": {"Deep completions per 90": 1.5, "Smart passes per 90": 2,
                                             "xA per 90": 4, "Passes to penalty area per 90": 2}},
        "Defensive Midfielder DM": {"metrics": {"Defensive duels per 90": 4, "Defensive duels won, %": 4,
                                               "PAdj Interceptions": 3, "Aerial duels per 90": 0.5, "Aerial duels won, %": 1}},
        "Goal Threat CM": {"metrics": {"Non-penalty goals per 90": 3, "xG per 90": 3, "Shots per 90": 1.5, "Touches in box per 90": 2}},
        "Ball Carrying CM": {"metrics": {"Dribbles per 90": 4, "Successful dribbles, %": 2, "Progressive runs per 90": 3, "Accelerations per 90": 3}},

        # ===== NEW ROLE (display/label-eligible; excluded from badge) =====
        "Box-to-Box CM": {"metrics": {
            "Touches in box per 90": 3,
            "Defensive duels per 90": 3,
            "Non-penalty goals per 90": 2,
        }},
    },
    "CB": {
        "Ball Playing CB": {"metrics": {"Passes per 90": 2, "Accurate passes, %": 2, "Forward passes per 90": 2,
                                        "Accurate forward passes, %": 2, "Progressive passes per 90": 2,
                                        "Progressive runs per 90": 1.5, "Dribbles per 90": 1.5,
                                        "Accurate long passes, %": 1, "Passes to final third per 90": 1.5}},
        "Wide CB": {"metrics": {"Defensive duels per 90": 1.5, "Defensive duels won, %": 2, "Dribbles per 90": 2,
                                "Forward passes per 90": 1, "Progressive passes per 90": 1, "Progressive runs per 90": 2}},
        "Box Defender": {"metrics": {"Aerial duels per 90": 1, "Aerial duels won, %": 3, "PAdj Interceptions": 2,
                                     "Shots blocked per 90": 1, "Defensive duels won, %": 4}},
    },
    "FB": {
        "Build Up FB": {"metrics": {"Passes per 90": 2, "Accurate passes, %": 1.5, "Forward passes per 90": 2,
                                    "Accurate forward passes, %": 2, "Progressive passes per 90": 2.5, "Progressive runs per 90": 2,
                                    "Dribbles per 90": 2, "Passes to final third per 90": 2, "xA per 90": 1}},
        "Attacking FB": {"metrics": {"Crosses per 90": 2, "Dribbles per 90": 3.5, "Accelerations per 90": 1,
                                     "Successful dribbles, %": 1, "Touches in box per 90": 2, "Progressive runs per 90": 3,
                                     "Passes to penalty area per 90": 2, "xA per 90": 3}},
        "Defensive FB": {"metrics": {"Aerial duels per 90": 1, "Aerial duels won, %": 1.5, "Defensive duels per 90": 2,
                                     "PAdj Interceptions": 3, "Shots blocked per 90": 1, "Defensive duels won, %": 3.5}},

        # ===== NEW ROLES (display/label-eligible; excluded from badge) =====
        "Wide Creator FB": {"metrics": {
            "xA per 90": 3,
            "Crosses per 90": 3,
            "Accurate crosses, %": 1,
        }},
        "Wide Carrier FB": {"metrics": {
            "Dribbles per 90": 3,
            "Successful dribbles, %": 1,
            "Progressive runs per 90": 3,
            "Accelerations per 90": 1,
        }},
    },
    "CF": {
        "Target Man CF": {"metrics": {"Aerial duels per 90": 3, "Aerial duels won, %": 5}},
        "Goal Threat CF": {"metrics": {"Non-penalty goals per 90": 3, "Shots per 90": 1.5, "xG per 90": 3,
                                       "Touches in box per 90": 1, "Shots on target, %": 0.5}},
        "Link Up CF": {"metrics": {"Passes per 90": 2, "Passes to penalty area per 90": 1.5, "Deep completions per 90": 1,
                                   "Smart passes per 90": 1.5, "Accurate passes, %": 1.5, "Key passes per 90": 1,
                                   "Dribbles per 90": 2, "Successful dribbles, %": 1, "Progressive runs per 90": 2, "xA per 90": 3}},

        # ===== NEW ROLES (display/label-eligible; excluded from badge) =====
        "False-9 Runner CF": {"metrics": {
            "Progressive runs per 90": 3,
            "Dribbles per 90": 3,
            "Successful dribbles, %": 2,
        }},
        "False-9 Passer CF": {"metrics": {
            "Passes per 90": 3,
            "Accurate passes, %": 2,
            "Smart passes per 90": 2,
            "Deep completions per 90": 2,
            "Passes to penalty area per 90": 3,
            "xA per 90": 3,
        }},
    },
    "ATT": {
        "Playmaker": {"metrics": {"Passes per 90": 2, "xA per 90": 3, "Key passes per 90": 1,
                                  "Deep completions per 90": 1.5, "Smart passes per 90": 1.5, "Passes to penalty area per 90": 2}},
        "Goal Threat": {"metrics": {"xG per 90": 3, "Non-penalty goals per 90": 3, "Shots per 90": 2, "Touches in box per 90": 2}},
        "Ball Carrier": {"metrics": {"Dribbles per 90": 4, "Successful dribbles, %": 2, "Progressive runs per 90": 3, "Accelerations per 90": 3}},
    },
}

# -------------------- POSITION GROUPING (UI) --------------------
POS_GROUPS = {
    "Strikers (CF)": ["CF"],
    "Center Backs (CB)": ["CB", "LCB", "RCB"],
    "Fullbacks (RB/LB/WB)": ["RB", "LB", "RWB", "LWB"],
    "Central Mid (DM/CM)": ["DMF", "CMF", "LCMF", "RCMF", "LDMF", "RDMF"],
    "Attackers (W/AM)": ["RW", "RWF", "LW", "LWF", "AMF", "RAMF", "LAMF"],
}

def _pos_token(p: str) -> str:
    s = str(p or "").strip().upper()
    toks = [t for t in re.split(r"[,/;]\s*|\s+", s) if t]
    return toks[0] if toks else ""

def _role_key_from_pos(tok: str) -> str:
    tok = str(tok or "").upper().strip()
    if tok.startswith("CF"):
        return "CF"
    if tok.startswith(("CB","LCB","RCB")):
        return "CB"
    if tok.startswith(("RB","LB","RWB","LWB")):
        return "FB"
    if tok.startswith(("DMF","CMF","LCMF","RCMF","LDMF","RDMF")):
        return "CM"
    if tok in {"RW","RWF","LW","LWF","AMF","RAMF","LAMF"}:
        return "ATT"
    return ""

# -------------------- Percentiles: compare vs same league + same position-group pool --------------------
@st.cache_data(show_spinner=False)
def _rank_percentiles_for_metric(df_ref: pd.DataFrame, metric: str) -> pd.Series:
    s = pd.to_numeric(df_ref.get(metric, pd.Series(index=df_ref.index, dtype=float)), errors="coerce")
    return s.rank(pct=True) * 100.0

def pct_of_row(ply: pd.Series, metric: str, df_all: pd.DataFrame, ref_df: pd.DataFrame) -> float:
    col = f"{metric} Percentile"
    if col in df_all.columns and pd.notna(ply.get(col, np.nan)):
        return float(ply[col])
    if metric not in df_all.columns:
        return np.nan
    if ref_df is None or ref_df.empty:
        ref_df = df_all
    pcts = _rank_percentiles_for_metric(ref_df, metric)
    return float(pcts.loc[ply.name]) if ply.name in pcts.index else np.nan

def val_str(ply: pd.Series, metric: str) -> str:
    if metric not in ply.index or pd.isna(ply[metric]):
        return "—"
    v = float(ply[metric])
    m = metric.lower()
    if "%" in metric or "percent" in m:
        return f"{int(round(v))}%"
    if "per 90" in m or "xg" in m or "xa" in m:
        return f"{v:.2f}"
    return f"{v:.2f}"

def div_color_tuple(v: float):
    if pd.isna(v): return (0.6, 0.63, 0.66)
    v = float(v)
    if v <= 50:
        t = v / 50.0
        c1, c2 = np.array([239, 68, 68]), np.array([234, 179, 8])
    else:
        t = (v - 50) / 50.0
        c1, c2 = np.array([234, 179, 8]), np.array([34, 197, 94])
    return tuple(((c1 + (c2 - c1) * t) / 255.0).astype(float))

def compute_role_scores(ply: pd.Series, df_all: pd.DataFrame, role_key: str, ref_df: pd.DataFrame) -> dict:
    buckets = ROLE_BUCKETS.get(role_key, {}) if role_key else {}
    out = {}
    for role_name, spec in buckets.items():
        met_w = (spec or {}).get("metrics", {}) or {}
        vals, wts = [], []
        for met, w in met_w.items():
            p = pct_of_row(ply, met, df_all, ref_df)
            if pd.isna(p):
                continue
            vals.append(float(p))
            wts.append(float(w))
        if vals and sum(wts) > 0:
            out[role_name] = float(np.average(vals, weights=wts))
    return out

def compute_strengths_weaknesses_styles(ply: pd.Series, df_all: pd.DataFrame, role_key: str, ref_df: pd.DataFrame):
    """
    Uses STYLE_MAP for role_key.
      - Strengths: sw label where metric pct >= HI
      - Weaknesses: sw label where metric pct <= LO
      - Styles: style label where metric pct >= STYLE_T
    Returns: (strengths_list, weaknesses_list, styles_list, pct_extra_dict)
    """
    style_map = STYLE_MAPS.get(role_key, {}) if role_key else {}
    strengths, weaknesses, styles = [], [], []
    pct_extra = {}

    for metric, meta in style_map.items():
        p = pct_of_row(ply, metric, df_all, ref_df)
        if pd.isna(p):
            continue
        pct_extra[metric] = float(p)

        sw = (meta or {}).get("sw")
        stl = (meta or {}).get("style")

        if sw:
            if p >= HI:
                strengths.append(str(sw))
            elif p <= LO:
                weaknesses.append(str(sw))

        if stl and p >= STYLE_T:
            styles.append(str(stl))

    # de-dupe while preserving order
    def _dedupe(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    strengths = _dedupe(strengths)[:10]
    weaknesses = _dedupe(weaknesses)[:10]
    styles = _dedupe(styles)[:10]
    return strengths, weaknesses, styles, pct_extra

# -------------------- Metric groups (requested) --------------------
ATTACKING_METRICS = [
    ("Crosses", "Crosses per 90"),
    ("Crossing %", "Accurate crosses, %"),
    ("Goals: Non-Penalty", "Non-penalty goals per 90"),
    ("xG", "xG per 90"),
    ("Expected Assists", "xA per 90"),
    ("Offensive Duels", "Offensive duels per 90"),
    ("Offensive Duel %", "Offensive duels won, %"),
    ("Progressive Runs", "Progressive runs per 90"),
    ("Shots", "Shots per 90"),
    ("Touches in box", "Touches in box per 90"),
]
DEFENSIVE_METRICS = [
    ("Aerial Duels", "Aerial duels per 90"),
    ("Aerial Win %", "Aerial duels won, %"),
    ("Defensive Duels", "Defensive duels per 90"),
    ("Defensive Duel %", "Defensive duels won, %"),
    ("PAdj Interceptions", "PAdj Interceptions"),
]
POSSESSION_METRICS = [
    ("Accelerations", "Accelerations per 90"),
    ("Deep completions", "Deep completions per 90"),
    ("Dribbles", "Dribbles per 90"),
    ("Dribbling %", "Successful dribbles, %"),
    ("Forward Passes", "Forward passes per 90"),
    ("Forward Pass %", "Accurate forward passes, %"),
    ("Key passes", "Key passes per 90"),
    ("Long Passes", "Long passes per 90"),
    ("Long Pass %", "Accurate long passes, %"),
    ("Passes", "Passes per 90"),
    ("Passing %", "Accurate passes, %"),
    ("Passes to Final 3rd", "Passes to final third per 90"),
    ("Passes to Penalty Area", "Passes to penalty area per 90"),
    ("Passes to Penalty Area %", "Accurate passes to penalty area, %"),
    ("Progessive Passes", "Progressive passes per 90"),
    ("Progressive Pass %", "Accurate progressive passes, %"),
    ("Smart Passes", "Smart passes per 90"),
]

def build_triples(ply: pd.Series, df_all: pd.DataFrame, ref_df: pd.DataFrame, pairs: list, pct_extra: dict):
    triples = []
    for lab, met in pairs:
        if met not in df_all.columns and f"{met} Percentile" not in df_all.columns:
            continue
        # use pct_extra if we computed it (styles/strength/weakness map)
        p = pct_extra.get(met, None)
        if p is None:
            p = pct_of_row(ply, met, df_all, ref_df)
        triples.append((lab, p, val_str(ply, met)))
    return triples

# -------------------- UI: select position group -> player --------------------
group = st.selectbox("Position group", list(POS_GROUPS.keys()), index=0)
pos_prefixes = {p.upper() for p in POS_GROUPS[group]}

# ✅ NEW: minutes slider for one-pager (affects dropdown eligibility + ref pool calculations)
min_mins_onepager = st.slider("Min minutes (one-pager)", 0, 6000, 600, 50)

df_view = df.copy()
if "Position" in df_view.columns:
    df_view["_pos_tok"] = df_view["Position"].apply(_pos_token)
    df_view = df_view[df_view["_pos_tok"].isin(pos_prefixes)].copy()

# ✅ NEW: apply minutes filter to the dropdown pool
df_view["Minutes played"] = pd.to_numeric(df_view.get("Minutes played", np.nan), errors="coerce")
df_view = df_view[df_view["Minutes played"] >= min_mins_onepager].copy()

if df_view.empty:
    st.info("No players for that position group + minutes filter in this dataset.")
    st.stop()

def _player_label(row: pd.Series) -> str:
    nm = str(row.get("Player", "—"))
    tm = str(row.get("Team", "")).strip()
    lg = str(row.get("League", "")).strip()
    return f"{nm} — {tm} ({lg})" if tm else f"{nm} ({lg})"

df_view["_label"] = df_view.apply(_player_label, axis=1)
picked = st.selectbox("Player", df_view["_label"].astype(str).tolist(), index=0)

player_row = df_view[df_view["_label"].astype(str) == str(picked)].head(1).copy()
if player_row.empty:
    st.info("Pick a player above.")
    st.stop()

ply = player_row.iloc[0]
player_name = str(ply.get("Player", "—"))
team = str(ply.get("Team", "?"))
league = str(ply.get("League", "?"))
pos = str(ply.get("Position", "?"))
pos_tok = _pos_token(pos)

# -------------------- Normalise position token for TOP label --------------------
POS_LABEL_MAP = {
    "LCB": "CB",
    "CB": "CB",
    "RCB": "CB",

    "RDMF": "DM",
    "LDMF": "DM",
    "DMF": "DM",

    "LCMF": "CM",
    "RCMF": "CM",

    "RAMF": "RW",
    "RW": "RW",
    "RWF": "RW",

    "LAMF": "LW",
    "LW": "LW",
    "LWF": "LW",
}

pos_tok_label = POS_LABEL_MAP.get(pos_tok, pos_tok)


# --- build the reference pool: SAME LEAGUE + SAME POSITION-GROUP ---
role_key = _role_key_from_pos(pos_tok)

ref_df = df.copy()
if "League" in ref_df.columns:
    ref_df = ref_df[ref_df["League"].astype(str) == str(league)].copy()

if "Position" in ref_df.columns:
    ref_df["_pos_tok"] = ref_df["Position"].apply(_pos_token)
    if role_key == "CF":
        allowed = {"CF"}
    elif role_key == "CB":
        allowed = {"CB", "LCB", "RCB"}
    elif role_key == "FB":
        allowed = {"RB", "LB", "RWB", "LWB"}
    elif role_key == "CM":
        allowed = {"DMF", "CMF", "LCMF", "RCMF", "LDMF", "RDMF"}
    elif role_key == "ATT":
        allowed = {"RW", "RWF", "LW", "LWF", "AMF", "RAMF", "LAMF"}
    else:
        allowed = set()
    if allowed:
        ref_df = ref_df[ref_df["_pos_tok"].isin(allowed)].copy()

# ✅ NEW: apply minutes filter to the reference pool used for percentiles/role scores/styles
ref_df["Minutes played"] = pd.to_numeric(ref_df.get("Minutes played", np.nan), errors="coerce")
ref_df = ref_df[ref_df["Minutes played"] >= min_mins_onepager].copy()

# -------------------- Badge options --------------------
use_league_weighting = st.toggle(
    "League-weighted badge score",
    value=True,
    help="If on, blends the player's role score with a league strength adjustment."
)

# (Optional) let user tune the blend
BETA_BADGE = st.slider(
    "League weighting strength",
    min_value=0.0, max_value=1.0, value=0.40, step=0.05,
    help="0 = no league influence, 1 = badge equals league strength only."
) if use_league_weighting else 0.0


# -------------------- Role scores + strengths/weaknesses/styles --------------------
role_scores = compute_role_scores(ply, df, role_key, ref_df)
strengths, weaknesses, styles, pct_extra = compute_strengths_weaknesses_styles(ply, df, role_key, ref_df)

# -------------------- Badge role filtering --------------------
# Exclude from badge score ONLY (still allowed for label + tabs)
BADGE_EXCLUDE_ROLES = {
    "target man cf",
}

# Roles that should be displayed / label-eligible but NOT counted in badge
LABEL_ONLY_ROLES = {
    "box-to-box cm",
    "wide creator fb",
    "wide carrier fb",
    "false-9 runner cf",
    "false-9 passer cf",
}

# filtered roles used ONLY for badge score
filtered_roles_for_badge = [
    (k, v) for k, v in role_scores.items()
    if str(k).strip().lower() not in BADGE_EXCLUDE_ROLES
    and str(k).strip().lower() not in LABEL_ONLY_ROLES
]

top3_roles_for_badge = sorted(filtered_roles_for_badge, key=lambda kv: kv[1], reverse=True)[:3]
best_val_raw = float(top3_roles_for_badge[0][1]) if top3_roles_for_badge else (max(role_scores.values()) if role_scores else np.nan)

_ls_map = globals().get("LEAGUE_STRENGTHS", {})
league_strength = float(_ls_map.get(str(league), 50.0)) if isinstance(_ls_map, dict) else 50.0

if use_league_weighting and pd.notna(best_val_raw):
    best_val_adj = (1.0 - BETA_BADGE) * float(best_val_raw) + BETA_BADGE * league_strength
else:
    best_val_adj = float(best_val_raw) if pd.notna(best_val_raw) else league_strength

# -------------------- Best role label text (AFTER crest) --------------------
# Label can use ANY role (including Target Man + the new roles)
best_role_name_for_label = (
    max(role_scores.items(), key=lambda kv: kv[1])[0]
    if role_scores else ""
)

role_prefix = " ".join(str(best_role_name_for_label).split()[:2]).strip()
best_role_pos_label = f"{role_prefix} {pos_tok_label}".strip() if role_prefix and pos_tok_label else ""



# -------------------- Metric triples --------------------
ATTACKING = build_triples(ply, df, ref_df, ATTACKING_METRICS, pct_extra)
DEFENSIVE = build_triples(ply, df, ref_df, DEFENSIVE_METRICS, pct_extra)
POSSESSION = build_triples(ply, df, ref_df, POSSESSION_METRICS, pct_extra)

# -------------------- Photos & crest --------------------
from io import BytesIO

PLACEHOLDER_IMG = "https://i.redd.it/43axcjdu59nd1.jpeg"

def _try_load_img(url: str):
    """
    Returns an image array for a valid URL, else None.
    Robust: uses PIL fallback for JPEG/odd formats.
    """
    if not url or not (str(url).startswith("http://") or str(url).startswith("https://")):
        return None
    try:
        r = requests.get(str(url), timeout=7, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.content:
            return None

        # First try matplotlib (works great for PNG)
        try:
            return plt.imread(BytesIO(r.content))
        except Exception:
            pass

        # Fallback: PIL (handles JPEG/WebP/etc)
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(BytesIO(r.content)).convert("RGB")
            return np.array(im)
        except Exception:
            return None

    except Exception:
        return None


# --- Player photo (FotMob -> placeholder) ---
photo_url = PLACEHOLDER_IMG
if "resolve_player_photo" in globals():
    try:
        resolved = resolve_player_photo(player_name, team, league)
        photo_url = (resolved or "").strip() or PLACEHOLDER_IMG
    except Exception:
        photo_url = PLACEHOLDER_IMG

# Try loading resolved photo; if it fails, force placeholder
photo_img = _try_load_img(photo_url)
if photo_img is None:
    photo_url = PLACEHOLDER_IMG
    photo_img = _try_load_img(photo_url)

# --- Club crest (optional; empty if missing) ---
crest_url = ""
if "resolve_team_crest" in globals():
    try:
        crest_url = (resolve_team_crest(team, league) or "").strip()
    except Exception:
        crest_url = ""

crest_img = _try_load_img(crest_url) if crest_url else None


# -------------------- One-pager styling --------------------
PAGE_BG   = "#0a0f1c"
PANEL_BG  = "#11161C"
TRACK_BG  = "#222c3d"
TEXT      = "#E5E7EB"
ROLE_GREY = "#a3a3a3"

CHIP_G_BG = "#22C55E"
CHIP_R_BG = "#EF4444"
CHIP_B_BG = "#60A5FA"

BAR_PX = 24
GAP_PX = 6
SEP_PX = 2
STEP_PX = BAR_PX + GAP_PX

LABEL_FS = 10.6
VALUE_FS = 8.5
TITLE_FS = 20

def _text_width_frac(fig, s, *, fontsize=8, weight="normal"):
    t = fig.text(0, 0, s, fontsize=fontsize, fontweight=weight, transform=fig.transFigure, alpha=0)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w_px = t.get_window_extent(renderer=r).width
    t.remove()
    return w_px / fig.bbox.width

def _text_height_frac(fig, s, *, fontsize=8, weight="normal"):
    t = fig.text(0, 0, s, fontsize=fontsize, fontweight=weight, transform=fig.transFigure, alpha=0)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    h_px = t.get_window_extent(renderer=r).height
    t.remove()
    return h_px / fig.bbox.height

def chip_row_exact(fig, items, y, bg, *, fs=10.1, weight="900", max_rows=1, gap_x=0.006, max_per_row=6):
    if not items:
        return y
    x0 = x = 0.055
    row_gap = 0.026
    pad_x = 0.004
    pad_y = 0.002
    h = _text_height_frac(fig, "Hg", fontsize=fs, weight=weight) + pad_y * 2
    per_row = 0
    for s in items[:60]:
        w = _text_width_frac(fig, s, fontsize=fs, weight=weight) + pad_x * 2
        need_wrap = (x + w > 0.965) or (max_per_row and per_row >= max_per_row)
        if need_wrap:
            max_rows -= 1
            if max_rows <= 0:
                break
            x = x0
            y -= row_gap
            per_row = 0
        fig.patches.append(
            mpatches.FancyBboxPatch(
                (x, y - h * 0.74), w, h,
                boxstyle=f"round,pad=0.001,rounding_size={h * 0.45}",
                transform=fig.transFigure, facecolor=bg, edgecolor="none"
            )
        )
        fig.text(x + pad_x, y - h * 0.33, s, fontsize=fs, color="#FFFFFF",
                 va="center", ha="left", fontweight=weight)
        x += w + gap_x
        per_row += 1
    return y - row_gap

def roles_row_tight(fig, rs: dict, y, *, fs=10.6, max_items=12):
    if not isinstance(rs, dict) or not rs:
        return y
    x0 = x = 0.055
    row_gap = 0.041
    gap = 0.003
    pad_x = 0.006
    pad_y = 0.003
    items = sorted(rs.items(), key=lambda kv: -kv[1])[:max_items]
    for rname, v in items:
        text_w = _text_width_frac(fig, rname, fontsize=fs, weight="800")
        text_h = _text_height_frac(fig, "Hg", fontsize=fs, weight="800")
        role_w = text_w + pad_x * 2
        role_h = text_h + pad_y * 2

        num_text = f"{int(round(v))}"
        num_wt = _text_width_frac(fig, num_text, fontsize=fs-0.6, weight="900")
        num_ht = _text_height_frac(fig, "Hg", fontsize=fs-0.6, weight="900")
        num_w = num_wt + pad_x * 2 * 0.9
        num_h = num_ht + pad_y * 2 * 0.9

        total = role_w + gap + num_w
        if x + total > 0.965:
            x = x0
            y -= row_gap

        fig.patches.append(mpatches.FancyBboxPatch(
            (x, y - role_h * 0.78), role_w, role_h,
            boxstyle=f"round,pad=0.001,rounding_size={role_h * 0.25}",
            transform=fig.transFigure, facecolor=ROLE_GREY, edgecolor="none"
        ))
        fig.text(x + pad_x, y - role_h * 0.33, rname,
                 fontsize=fs, color="#FFFFFF", va="center", ha="left", fontweight="800")

        R, G, B = [int(255 * c) for c in div_color_tuple(v)]
        bx = x + role_w + gap
        fig.patches.append(mpatches.FancyBboxPatch(
            (bx, y - num_h * 0.78), num_w, num_h,
            boxstyle=f"round,pad=0.001,rounding_size={num_h * 0.25}",
            transform=fig.transFigure, facecolor=f"#{R:02x}{G:02x}{B:02x}", edgecolor="none"
        ))
        fig.text(bx + num_w / 2, y - num_h * 0.33, num_text,
                 fontsize=fs - 0.6, color="#FFFFFF", va="center", ha="center", fontweight="900")

        x = bx + num_w + 0.010
    return y - row_gap

def bar_panel(fig, left, top, width, triples, title):
    n_rows = len(triples)
    fig.canvas.draw()
    fig_px_h = fig.bbox.height

    ax_h_frac = (max(1, n_rows) * STEP_PX) / fig_px_h
    bottom = top - ax_h_frac

    labels = [t[0] for t in triples]
    max_label_w_frac = max(_text_width_frac(fig, s, fontsize=LABEL_FS, weight="bold") for s in labels) if labels else 0
    gutter_w = max_label_w_frac + 0.006

    ax_panel = fig.add_axes([left, bottom, width, ax_h_frac])
    ax_panel.set_facecolor(PANEL_BG)
    ax_panel.set_xticks([]); ax_panel.set_yticks([])
    for sp in ax_panel.spines.values():
        sp.set_visible(False)

    bar_left = left + gutter_w
    bar_width = max(0.001, width - gutter_w - 0.004)
    ax = fig.add_axes([bar_left, bottom, bar_width, ax_h_frac])
    ax.set_facecolor(PANEL_BG)

    pcts = [float(np.nan_to_num(t[1], nan=0.0)) for t in triples]
    texts = [t[2] for t in triples]
    n = len(labels)

    bar_du = BAR_PX / STEP_PX
    gap_du = GAP_PX / STEP_PX
    sep_du = SEP_PX / STEP_PX

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, max(1, n) - 0.5)
    y_idx = np.arange(max(1, n))[::-1]

    track_h = bar_du + gap_du - sep_du
    for yi in y_idx[:n]:
        ax.add_patch(mpatches.Rectangle((0, yi - track_h/2), 100, track_h, facecolor=TRACK_BG, edgecolor="none"))

    for yi, v, t in zip(y_idx[:n], pcts, texts):
        ax.add_patch(mpatches.Rectangle((0, yi - bar_du/2), v, bar_du, facecolor=div_color_tuple(v), edgecolor="none"))
        ax.text(1.0, yi, t, va="center", ha="left", color="#0B0B0B", fontsize=VALUE_FS + 0.5, weight="700")

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=0)
    ax.grid(False)

    # 50% midline (stronger)
    ax.axvline(
        50,
        color="#E5E7EB",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        zorder=5,
    )

    # label under the axis (optional)
    y0, y1 = ax.get_ylim()
    ax.text(
        50,
        y0 - 0.35,
        "League avg",
        color="#CBD5E1",
        fontsize=8,
        ha="center",
        va="top",
    )

    for yi, lab in zip(y_idx[:n], labels):
        y_fig = bottom + ax_h_frac * ((yi + 0.5) / max(1, n))
        fig.text(left + 0.006/2, y_fig, lab, color=TEXT, fontsize=LABEL_FS, fontweight="bold",
                 va="center", ha="left")

    title_y = bottom + ax_h_frac + 0.008
    fig.text(left + 0.006/2, title_y, title, color=TEXT, fontsize=TITLE_FS, fontweight="900",
             ha="left", va="bottom")
    ax.plot([0, 1], [1, 1], transform=ax.transAxes, color="#94A3B8", linewidth=0.8, alpha=0.35)

    return bottom

# -------------------- Build the figure --------------------
W, H = 1500, 1080
fig = plt.figure(figsize=(W/100, H/100), dpi=100)
fig.patch.set_facecolor(PAGE_BG)

# ---- header layout knobs ----
PHOTO_W = 0.10
PHOTO_H = 0.10
PHOTO_X = 0.050
PHOTO_Y = 0.915

NAME_X = PHOTO_X + PHOTO_W + 0.005
NAME_Y = 0.97

BADGE_SCALE = 1.28

# Photo beside name (left)
if photo_img is not None:
    axp = fig.add_axes([PHOTO_X, PHOTO_Y, PHOTO_W, PHOTO_H])
    axp.imshow(photo_img)
    axp.axis("off")
    axp.set_facecolor(PAGE_BG)

# Name
name_fs = 28
name_text = fig.text(NAME_X, NAME_Y, f"{player_name}", color="#FFFFFF",
                     fontsize=name_fs, fontweight="900", va="top", ha="left")

fig.canvas.draw()
r = fig.canvas.get_renderer()
name_bbox = name_text.get_window_extent(renderer=r)
name_w_frac = name_bbox.width / fig.bbox.width
name_h_frac = name_bbox.height / fig.bbox.height
NAME_YC = NAME_Y - (name_h_frac / 2)   # vertical center of the name line


# Bigger badge right beside name
badge_x = NAME_X + name_w_frac + 0.010

bh = name_h_frac * BADGE_SCALE
bw = bh

# TRUE vertical centering
by = NAME_Y - (name_h_frac / 2) - (bh / 2)

R, G, B = [int(255*c) for c in div_color_tuple(best_val_adj)]
fig.patches.append(mpatches.FancyBboxPatch(
    (badge_x, by), bw, bh,
    boxstyle="round,pad=0.001,rounding_size=0.012",
    transform=fig.transFigure,
    facecolor=f"#{R:02x}{G:02x}{B:02x}",
    edgecolor="none",
))
fig.text(badge_x + bw/2, by + bh/2 - 0.0005, f"{int(round(best_val_adj))}",
         fontsize=18.6 * BADGE_SCALE, color="#FFFFFF",
         va="center", ha="center", fontweight="900")

# Crest anchored to the score pill (fixed gap)
CREST_W = 0.050
CREST_H = 0.050
CREST_GAP = 0.018          # distance between pill and crest
CREST_RIGHT_LIMIT = 0.985  # keep inside figure

crest_drawn = False
crest_x = badge_x + bw + CREST_GAP
crest_x = min(crest_x, CREST_RIGHT_LIMIT - CREST_W)

# align vertically to the pill (centered)
crest_y = NAME_YC - (CREST_H / 2)

if crest_img is not None:
    axc = fig.add_axes([crest_x, crest_y, CREST_W, CREST_H])
    axc.imshow(crest_img)
    axc.axis("off")
    axc.set_facecolor(PAGE_BG)
    crest_drawn = True

# -------------------- Best role *position* label AFTER CREST --------------------
ROLE_LABEL_FS = name_fs * 0.60   # slightly smaller than name
ROLE_LABEL_COL = "#9CA3AF"       # lighter grey
ROLE_LABEL_GAP = 0.012           # gap after crest (or after badge if no crest)

label_x = None
if best_role_pos_label:
    label_x = (crest_x + CREST_W + ROLE_LABEL_GAP) if crest_drawn else (badge_x + bw + ROLE_LABEL_GAP)
    label_x = min(label_x, 0.975)

if best_role_pos_label and label_x is not None:
    fig.text(
        label_x, NAME_YC, best_role_pos_label,
        color=ROLE_LABEL_COL,
        fontsize=ROLE_LABEL_FS,
        fontweight="700",
        va="center", ha="left"
    )

# -------------------- META line (NO leading score; tighter) --------------------
age = int(ply["Age"]) if pd.notna(ply.get("Age")) else None
mins = int(ply.get("Minutes played", np.nan)) if pd.notna(ply.get("Minutes played")) else None
matches = int(ply.get("Matches played", np.nan)) if pd.notna(ply.get("Matches played")) else None
goals = int(ply.get("Goals", np.nan)) if pd.notna(ply.get("Goals")) else 0
assists = int(ply.get("Assists", np.nan)) if pd.notna(ply.get("Assists")) else 0

if "xG" in ply.index and pd.notna(ply.get("xG")):
    xg_total = float(ply["xG"])
else:
    xg_per90 = float(ply.get("xG per 90", np.nan)) if pd.notna(ply.get("xG per 90")) else np.nan
    xg_total = float(xg_per90) * (float(mins) / 90.0) if (pd.notna(xg_per90) and mins) else np.nan
xg_total_str = f"{xg_total:.2f}" if pd.notna(xg_total) else "—"

meta_y = 0.9
x_meta = 0.055
runs = [
    (f"{pos} — ", "normal"),
    (team, "bold"),
    (" — ", "normal"),
    (league, "bold"),
    (f" — Age {age if age is not None else '—'} — Minutes {mins if mins is not None else '—'} — "
     f"Matches {matches if matches is not None else '—'} — Goals {goals} — xG {xg_total_str} — Assists {assists}", "normal")
]
for txt, weight in runs:
    fs = 13
    fig.text(x_meta, meta_y, txt, color="#FFFFFF", fontsize=fs,
             fontweight=("900" if weight == "bold" else "normal"),
             ha="left", va="center")
    x_meta += _text_width_frac(fig, txt, fontsize=fs, weight=("900" if weight == "bold" else "normal")) + (0.004 if txt.strip() else 0)

# -------------------- Strengths / Weaknesses / Styles (chips) --------------------
y_chips = 0.88
y_chips = chip_row_exact(fig, strengths,  y_chips, CHIP_G_BG, fs=10.1, max_rows=1, max_per_row=6)
y_chips = chip_row_exact(fig, weaknesses, y_chips, CHIP_R_BG, fs=10.1, max_rows=1, max_per_row=6)
y_chips = chip_row_exact(fig, styles,     y_chips, CHIP_B_BG, fs=10.1, max_rows=1, max_per_row=6)
y_chips -= 0.009

# -------------------- Roles row (DISPLAY ALL roles incl Target Man + NEW roles) --------------------
roles_for_row = dict(sorted(role_scores.items(), key=lambda kv: -kv[1])[:10])
y_roles = roles_row_tight(fig, roles_for_row, y_chips, fs=10.6, max_items=10)

# -------------------- Layout (reduced top gap; panels start higher) --------------------
LEFT = 0.050
WIDTH_L = 0.41
MID_GAP = 0.040
RIGHT = LEFT + WIDTH_L + MID_GAP
WIDTH_R = 0.41

TOP = y_roles - 0.02   # ✅ panels start right under roles (smaller = tighter)
V_GAP_FRAC = 0.050

att_bottom = bar_panel(fig, LEFT, TOP, WIDTH_L, ATTACKING, "Attacking")
_ = bar_panel(fig, LEFT, att_bottom - V_GAP_FRAC, WIDTH_L, DEFENSIVE, "Defensive")
_ = bar_panel(fig, RIGHT, TOP, WIDTH_R, POSSESSION, "Possession")

# -------------------- render + download --------------------
st.pyplot(fig, use_container_width=True)

buf = BytesIO()
fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
st.download_button(
    "⬇️ Download one-pager (PNG)",
    data=buf.getvalue(),
    file_name=f"{str(player_name).replace(' ', '_')}_onepager.png",
    mime="image/png",
)

# ============================ END ONE-PAGER ============================

# ============================ SHORTLIST (T20) — ELITE TABLE (v3.3++) ============================
# EXACT SAME AS YOUR v3.3, plus:
# ✅ Display-only filters: Contract expires (year), Foot, Birth country, Market value (0–150m, 0.5m)
# ✅ Best-role DISPLAY uses ALL roles (including label-only + Target Man); scoring excludes them UNLESS selected
# ✅ Best-role display formatting: first two words + position label (POS_LABEL_MAP)
# ✅ POS_LABEL_MAP applied to the player's first position token
# ✅ If crest/badge cannot be found -> uses FALLBACK_BADGE url
# ✅ Strips accidental HTML tags from Team/Player strings (prevents <span> showing in table)
# ✅ Minutes is still the ONLY filter that changes the scoring pool (percentile reference)
# ✅ (NEW) Output table: hides Region column
# ✅ (NEW) Output table: adds small Birth-country flag beside player name (IMAGE flags)
#
# NEW (your latest asks):
# ✅ Outlier Search toggle: uses Z-scores on RAW metrics (per-league+pos ref), mapped to 0–100 via Normal CDF
# ✅ Display-only raw metric filters (>= / <=) for the given list
# ✅ Youth leagues excluded by default; toggle to include back
# ---------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
import re
import math

FALLBACK_BADGE = "https://clipart-library.com/images/5cRX8Kx9i.png"

# --- Youth leagues: excluded by default, optional toggle to include ---
YOUTH_LEAGUES = {
    "Brazil 3.",
    "England 7.",
    "England 8.",
    "England 9.",
    "England 10.",
    "Portugal 3.",
    "Denmark 3.",
    "Germany 4.",
    "USA 2.",
    "Ireland 2.",
    "Estonia 2.",
}

st.markdown("---")
st.header("📋 Shortlist (T20)")

# ---------- role exclusions (same logic as one-pager badge) ----------
BADGE_EXCLUDE_ROLES = {"target man cf"}  # excluded from badge/base score by default
LABEL_ONLY_ROLES = {
    "box-to-box cm",
    "wide creator fb",
    "wide carrier fb",
    "false-9 runner cf",
    "false-9 passer cf",
}

def _norm_role_name(x: str) -> str:
    return str(x or "").strip().lower()

_BADGE_EXCLUDE_N = {_norm_role_name(r) for r in BADGE_EXCLUDE_ROLES}
_LABEL_ONLY_N = {_norm_role_name(r) for r in LABEL_ONLY_ROLES}

def _is_badge_excluded(role_name: str) -> bool:
    return _norm_role_name(role_name) in _BADGE_EXCLUDE_N

def _is_label_only(role_name: str) -> bool:
    return _norm_role_name(role_name) in _LABEL_ONLY_N

# ---------- POS label map (user spec) ----------
POS_LABEL_MAP = {
    "LCB": "CB",
    "CB": "CB",
    "RCB": "CB",

    "RDMF": "DM",
    "LDMF": "DM",
    "DMF": "DM",

    "LCMF": "CM",
    "RCMF": "CM",

    "RAMF": "RW",
    "RW": "RW",
    "RWF": "RW",

    "LAMF": "LW",
    "LW": "LW",
    "LWF": "LW",
}

# ---------- group → allowed pos tokens ----------
POS_GROUPS_LOCAL = {
    "All positions": None,  # special mode
    "Strikers (CF)": {"CF"},
    "Center Backs (CB)": {"CB", "LCB", "RCB"},
    "Fullbacks (RB/LB/WB)": {"RB", "LB", "RWB", "LWB"},
    "Central Mid (DM/CM)": {"DMF", "CMF", "LCMF", "RCMF", "LDMF", "RDMF"},
    "Attackers (W/AM)": {"RW", "RWF", "LW", "LWF", "AMF", "RAMF", "LAMF"},
}

def _role_key_from_pos_tok(tok: str) -> str:
    t = str(tok or "").upper().strip()
    if t == "CF":
        return "CF"
    if t in {"CB", "LCB", "RCB"}:
        return "CB"
    if t in {"RB", "LB", "RWB", "LWB"}:
        return "FB"
    if t in {"DMF", "CMF", "LCMF", "RCMF", "LDMF", "RDMF"}:
        return "CM"
    if t in {"RW", "RWF", "LW", "LWF", "AMF", "RAMF", "LAMF"}:
        return "ATT"
    return ""

def _role_key_from_group(group: str) -> str:
    if group.startswith("Strikers"):
        return "CF"
    if group.startswith("Center Backs"):
        return "CB"
    if group.startswith("Fullbacks"):
        return "FB"
    if group.startswith("Central Mid"):
        return "CM"
    if group.startswith("Attackers"):
        return "ATT"
    return ""

# ---------- helpers ----------
def _strip_tags(x) -> str:
    s = "" if x is None else str(x)
    return re.sub(r"<[^>]+>", "", s).strip()

def _esc(x) -> str:
    return str(x if x is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _is_http_url(u: str) -> bool:
    u = (u or "").strip()
    return u.startswith("http://") or u.startswith("https://")

def _pos_token(position_str: str) -> str:
    s = _strip_tags(position_str)
    if not s:
        return ""
    return s.split(",")[0].strip().upper()

# ---- z-score -> 0..100 via Normal CDF ----
def _z_to_0_100(z: float) -> float:
    # Phi(z) = 0.5*(1+erf(z/sqrt(2)))
    try:
        p = 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))
        return float(np.clip(p * 100.0, 0.0, 100.0))
    except Exception:
        return 50.0

@st.cache_data(show_spinner=False)
def _country_to_flag_url(country_name: str) -> str:
    """
    Returns a small flag IMAGE url (or "" if unknown).
    Uses explicit mapping for ALL Birth country values you provided.
    """
    c = _strip_tags(country_name).strip()
    if not c:
        return ""

    # Exact dataset values -> FlagCDN code (mostly ISO alpha-2; UK nations use gb-*)
    NAME_TO_CODE = {
        "Netherlands": "nl",
        "Spain": "es",
        "Serbia": "rs",
        "Germany": "de",
        "England": "gb-eng",
        "Republic of Ireland": "ie",
        "France": "fr",
        "Slovakia": "sk",
        "Italy": "it",
        "Switzerland": "ch",
        "Wales": "gb-wls",
        "Belgium": "be",
        "Hungary": "hu",
        "Côte d'Ivoire": "ci",
        "Portugal": "pt",
        "Brazil": "br",
        "Argentina": "ar",
        "Denmark": "dk",
        "Northern Ireland": "gb-nir",
        "Sierra Leone": "sl",
        "Norway": "no",
        "Sweden": "se",
        "United States": "us",
        "Gambia": "gm",
        "Ukraine": "ua",
        "Ghana": "gh",
        "Scotland": "gb-sct",
        "Paraguay": "py",
        "Senegal": "sn",
        "Uruguay": "uy",
        "Czech Republic": "cz",
        "Guernsey": "gg",
        "Croatia": "hr",
        "Nigeria": "ng",
        "Ecuador": "ec",
        "Colombia": "co",
        "Mexico": "mx",
        "Egypt": "eg",
        "Angola": "ao",
        "French Guiana": "gf",
        "Japan": "jp",
        "Burkina Faso": "bf",
        "Mozambique": "mz",
        "Greece": "gr",
        "Slovenia": "si",
        "Guinea-Bissau": "gw",
        "Zimbabwe": "zw",
        "Cameroon": "cm",
        "South Africa": "za",
        "Korea Republic": "kr",
        "Chad": "td",
        "Congo DR": "cd",
        "Austria": "at",
        "Bulgaria": "bg",
        "Türkiye": "tr",
        "New Zealand": "nz",
        "Georgia": "ge",
        "Uzbekistan": "uz",
        "Morocco": "ma",
        "Bosnia and Herzegovina": "ba",
        "Poland": "pl",
        "Australia": "au",
        "Saudi Arabia": "sa",
        "Chile": "cl",
        "Mali": "ml",
        "Tanzania": "tz",
        "Canada": "ca",
        "Montenegro": "me",
        "Zambia": "zm",
        "Panama": "pa",
        "Jersey": "je",
        "Iceland": "is",
        "Algeria": "dz",
        "Curaçao": "cw",
        "Finland": "fi",
        "Bermuda": "bm",
        "Barbados": "bb",
        "Congo": "cg",
        "Grenada": "gd",
        "Montserrat": "ms",
        "Liberia": "lr",
        "Jamaica": "jm",
        "Lithuania": "lt",
        "Afghanistan": "af",
        "Malawi": "mw",
        "Belize": "bz",
        "Guadeloupe": "gp",
        "Albania": "al",
        "Somalia": "so",
        "Guyana": "gy",
        "British Virgin Islands": "vg",
        "Suriname": "sr",
        "Gibraltar": "gi",
        "Honduras": "hn",
        "Mauritius": "mu",
        "Great Britain": "gb",
        "Russia": "ru",
        "Cyprus": "cy",
        "Fiji": "fj",
        "Thailand": "th",
        "Hong Kong": "hk",
        "Latvia": "lv",
        "Trinidad and Tobago": "tt",
        "Eritrea": "er",
        "North Macedonia": "mk",
        "Kosovo": "xk",
        "Azerbaijan": "az",
        "Luxembourg": "lu",
        "Venezuela": "ve",
        "Peru": "pe",
        "Israel": "il",
        "Moldova": "md",
        "Estonia": "ee",
        "Costa Rica": "cr",
        "Armenia": "am",
        "Guinea": "gn",
        "Comoros": "km",
        "Kenya": "ke",
        "Vanuatu": "vu",
        "Malta": "mt",
        "Iraq": "iq",
        "Dominica": "dm",
        "Réunion": "re",
        "Cape Verde Islands": "cv",
        "Romania": "ro",
        "Liechtenstein": "li",
        "Kazakhstan": "kz",
        "Belarus": "by",
        "Benin": "bj",
        "Rwanda": "rw",
        "Dominican Republic": "do",
        "Iran": "ir",
        "Niger": "ne",
        "Singapore": "sg",
        "Burundi": "bi",
        "Madagascar": "mg",
        "Togo": "tg",
        "Central African Republic": "cf",
        "Bolivia": "bo",
        "Tajikistan": "tj",
        "Martinique": "mq",
        "Cuba": "cu",
        "China PR": "cn",
        "Equatorial Guinea": "gq",
        "Gabon": "ga",
        "Chinese Taipei": "tw",
        "Guatemala": "gt",
        "Tunisia": "tn",
        "Lebanon": "lb",
        "Bahrain": "bh",
        "Uganda": "ug",
        "Oman": "om",
        "Faroe Islands": "fo",
        "Jordan": "jo",
        "Haiti": "ht",
        "Syria": "sy",
        "St. Lucia": "lc",
        "Indonesia": "id",
        "Ethiopia": "et",
        "Philippines": "ph",
        "Mauritania": "mr",
        "Palestine": "ps",
        "Libya": "ly",
        "Malaysia": "my",
        "Korea DPR": "kp",
        "Nicaragua": "ni",
        "South Sudan": "ss",
        "Bonaire": "bq",
        "São Tomé e Príncipe": "st",
        "St. Kitts and Nevis": "kn",
        "El Salvador": "sv",
        "New Caledonia": "nc",
        "Kyrgyzstan": "kg",
        "Isle of Man": "im",
        "Lesotho": "ls",
        "United Arab Emirates": "ae",
        "Andorra": "ad",
        "Mongolia": "mn",
        "Namibia": "na",
        "Eswatini": "sz",
        "Pakistan": "pk",
        "Djibouti": "dj",
        "Antigua and Barbuda": "ag",
        "Puerto Rico": "pr",
        "Cayman Islands": "ky",
        "St. Vincent and the Grenadines": "vc",

        # Non-country placeholders (no flag)
        "Africa": "",
        "Other": "",
    }

    code = NAME_TO_CODE.get(c, "")
    if code:
        return f"https://flagcdn.com/w20/{code}.png"

    # Fallback: try pycountry if installed
    try:
        import pycountry  # type: ignore
        obj = pycountry.countries.lookup(c)
        a2 = getattr(obj, "alpha_2", "")
        if a2 and len(a2) == 2:
            return f"https://flagcdn.com/w20/{a2.lower()}.png"
    except Exception:
        pass

    return ""

def _first_pos_label(position_str: str) -> str:
    # takes first token from "CF, LWF, LW" etc; maps via POS_LABEL_MAP if present
    p = _strip_tags(position_str)
    tok = (p.split(",")[0].strip().upper() if p else "")
    return POS_LABEL_MAP.get(tok, tok)

def _format_best_role_display(role_name: str, position_str: str) -> str:
    # "ball-carrying cm" -> "Ball-Carrying CM" (no double CM)
    r = _strip_tags(role_name).strip()
    if not r:
        return ""

    parts = r.split()
    prefix_parts = parts[:2] if len(parts) >= 2 else parts[:1]
    prefix = " ".join(prefix_parts).title()

    pos_lab = _first_pos_label(position_str).upper().strip()
    if not pos_lab:
        return prefix

    # If the role already contains the pos in its first two words, don't append again
    last_word = prefix_parts[-1].upper().strip() if prefix_parts else ""
    if last_word == pos_lab:
        return prefix_parts[0].replace("-", "-").title() + f" {pos_lab}" if len(prefix_parts) == 2 else prefix

    # Also guard against common role suffix tokens (cm/cb/fb/cf/att/dm/rw/lw)
    common_pos_words = {"CM","CB","FB","CF","ATT","DM","RW","LW"}
    if last_word in common_pos_words:
        return " ".join([w.title() for w in prefix_parts[:-1]]) + f" {pos_lab}"

    return f"{prefix} {pos_lab}"

def _contract_year(x):
    s = _strip_tags(x)
    if not s:
        return np.nan
    m = re.search(r"(\d{4})", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return np.nan
    return np.nan

def _norm_foot(x) -> str:
    s = _strip_tags(x).lower()
    if s in {"r", "right"}:
        return "right"
    if s in {"l", "left"}:
        return "left"
    if s in {"both", "b"}:
        return "both"
    if s in {"unknown", "unk", "?"}:
        return "unknown"
    if s == "":
        return ""
    return s

def _parse_market_value_to_m(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    s = str(x).lower().replace(",", "").replace("€", "").replace("$", "").strip()

    if not s:
        return np.nan

    # Explicit suffix handling
    if s.endswith("k"):
        return float(s.replace("k","")) / 1000

    if s.endswith("m"):
        return float(s.replace("m",""))

    if s.endswith("bn") or s.endswith("b"):
        return float(s.replace("bn","").replace("b","")) * 1000

    # Raw numbers (e.g. 500000) → assume euros → convert to millions
    try:
        v = float(s)
        if v > 1000:      # clearly not already millions
            return v / 1_000_000
        return v
    except:
        return np.nan

# ---------- percentiles OR outlier z-score mode (raw metrics) ----------
@st.cache_data(show_spinner=False)
def _scores_for_ref(df_ref: pd.DataFrame, metrics: list, outlier_mode: bool) -> pd.DataFrame:
    """
    If outlier_mode=False: rank percentiles (0..100) like before.
    If outlier_mode=True: compute Z-score on RAW metric values per ref, map to 0..100 via Normal CDF.
    """
    out = pd.DataFrame(index=df_ref.index)
    for m in metrics:
        if m in df_ref.columns:
            s = pd.to_numeric(df_ref[m], errors="coerce")
            if not outlier_mode:
                out[m] = s.rank(pct=True) * 100.0
            else:
                mu = float(np.nanmean(s.values)) if np.isfinite(np.nanmean(s.values)) else 0.0
                sd = float(np.nanstd(s.values)) if np.isfinite(np.nanstd(s.values)) else 0.0
                if sd <= 1e-12:
                    z = pd.Series(0.0, index=s.index)
                else:
                    z = (s - mu) / sd
                out[m] = z.apply(_z_to_0_100)
    return out

def _div_color_hex(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "#9ca3af"
    v = max(0.0, min(100.0, v))
    if v <= 50:
        t = v / 50.0
        c1 = np.array([239, 68, 68])   # red
        c2 = np.array([234, 179, 8])   # amber
    else:
        t = (v - 50) / 50.0
        c1 = np.array([234, 179, 8])   # amber
        c2 = np.array([34, 197, 94])   # green
    rgb = (c1 + (c2 - c1) * t).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

# ============================== Complete Score weights (your formulas) ==============================
COMPLETE_WEIGHTS = {
    "CB": {
        "Aerial duels won, %": 0.15,
        "Defensive duels won, %": 0.15,
        "Accurate passes, %": 0.10,
        "Accurate forward passes, %": 0.10,
        "Dribbles per 90": 0.05,
        "Progressive runs per 90": 0.15,
        "Progressive passes per 90": 0.15,
        "PAdj Interceptions": 0.15,
    },
    "FB": {
        "PAdj Interceptions": 0.10,
        "Defensive duels won, %": 0.10,
        "Accurate passes, %": 0.10,
        "Defensive duels per 90": 0.05,
        "Dribbles per 90": 0.10,
        "Progressive runs per 90": 0.10,
        "Progressive passes per 90": 0.10,
        "Passes to final third per 90": 0.10,
        "xA per 90": 0.10,
        "Passes to penalty area per 90": 0.10,
        "Smart passes per 90": 0.05,
    },
    "CM": {
        "PAdj Interceptions": 0.10,
        "Defensive duels won, %": 0.10,
        "Accurate passes, %": 0.10,
        "Defensive duels per 90": 0.05,
        "Dribbles per 90": 0.10,
        "Progressive runs per 90": 0.10,
        "Progressive passes per 90": 0.10,
        "Passes to final third per 90": 0.05,
        "xA per 90": 0.10,
        "Passes to penalty area per 90": 0.10,
        "Non-penalty goals per 90": 0.05,
        "xG per 90": 0.05,
    },
    "ATT": {
        "Accurate passes, %": 0.10,
        "Dribbles per 90": 0.15,
        "Progressive runs per 90": 0.10,
        "Passes to final third per 90": 0.05,
        "xA per 90": 0.20,
        "Passes to penalty area per 90": 0.05,
        "Non-penalty goals per 90": 0.15,
        "xG per 90": 0.20,
    },
    "CF": {
        "Accurate passes, %": 0.10,
        "Dribbles per 90": 0.15,
        "Progressive runs per 90": 0.10,
        "xA per 90": 0.15,
        "Passes to penalty area per 90": 0.05,
        "Non-penalty goals per 90": 0.20,
        "xG per 90": 0.25,
    },
}

def _complete_score_from_pcts(pcts: pd.DataFrame, rk: str, idx: pd.Index) -> pd.Series:
    wmap = COMPLETE_WEIGHTS.get(rk, {}) or {}
    if not wmap:
        return pd.Series(np.nan, index=idx)

    cols, wts = [], []
    for met, w in wmap.items():
        if met in pcts.columns:
            cols.append(met)
            wts.append(float(w))

    if not cols or sum(wts) <= 0:
        return pd.Series(np.nan, index=idx)

    M = pcts.loc[idx, cols].astype(float)
    w = np.array(wts, dtype=float)
    return (M.mul(w, axis=1).sum(axis=1) / w.sum())  # renormalized

# ============================== UI (controls) ==============================
r1, r2, r3, r4, r5, r6 = st.columns([1.5, 1.2, 1.3, 1.2, 1.2, 1.2])
with r1:
    group_filter = st.selectbox(
        "Position group",
        list(POS_GROUPS_LOCAL.keys()),
        index=1,  # default Strikers
        key="t20_group_v33",
    )
with r2:
    # Only thing that affects scoring pool
    min_minutes_pool = st.slider("Min minutes (pool)", 0, 6000, 500, 50, key="t20_min_minutes_pool_v33")
with r3:
    outlier_search = st.toggle(
        "🔍 Outlier Search",
        value=False,
        key="t20_outlier_search_v33",
        help="Uses Z-scores on RAW metrics (per league+pos reference), mapped to 0–100 via Normal CDF."
    )
with r4:
    default_thr = st.slider("Default role threshold", 0, 100, 60, 1, key="t20_default_thr_v33")
with r5:
    use_league_weighting = st.toggle("League-adjusted score", value=True, key="t20_league_weighting_v33")
with r6:
    include_youth_leagues = st.toggle("Include youth leagues", value=False, key="t20_include_youth_v33")

# Base score toggle
b1, b2, b3, b4, b5 = st.columns([1.4, 1.2, 1.2, 1.2, 1.2])
with b1:
    use_complete_base = st.toggle("Use Complete Score as base", value=False, key="t20_use_complete_v33")
with b2:
    top_n = st.number_input("Top N", 5, 200, 20, 5, key="t20_top_n_v33")
with b3:
    player_q = st.text_input("Player search", "", key="t20_player_q_v33")
with b4:
    team_q = st.text_input("Team search", "", key="t20_team_q_v33")
with b5:
    BETA_BADGE = st.slider(
        "β",
        0.0, 1.0, 0.40, 0.05,
        key="t20_beta_badge_v33",
        help="0 = pure base score, 1 = pure league strength."
    ) if use_league_weighting else 0.0

# Display-only filters (original)
d1, d2 = st.columns([1.2, 1.2])
with d1:
    age_band = st.slider("Age band (display)", 14, 45, (16, 35), 1, key="t20_age_band_v33")
with d2:
    league_q = st.slider("League quality (display)", 0, 100, (20, 100), 1, key="t20_league_q_v33")

# ✅ NEW display-only filters (requested)
st.markdown("#### 🧾 Player filters (display only)")
p1, p2, p3, p4 = st.columns([1.2, 1.2, 1.6, 1.2])

contract_col = "Contract expires"
foot_col = "Foot"
birth_col = "Birth country"
mv_col = "Market value"

with p1:
    contract_year_range = st.slider("Contract year", 2000, 2035, (2000, 2035), 1, key="t20_contract_year_v33")
with p2:
    foot_opts = ["", "right", "left", "both", "unknown"]
    foot_sel = st.multiselect("Foot", foot_opts, default=[], key="t20_foot_v33")
with p3:
    BIRTH_COUNTRIES_ALLOWED = [
        "", "Netherlands","Spain","Serbia","Germany","England","Republic of Ireland","France","Slovakia","Italy",
        "Switzerland","Wales","Belgium","Hungary","Côte d'Ivoire","Portugal","Brazil","Argentina","Denmark",
        "Northern Ireland","Sierra Leone","Norway","Sweden","United States","Gambia","Ukraine","Ghana","Scotland",
        "Paraguay","Senegal","Uruguay","Czech Republic","Guernsey","Croatia","Nigeria","Ecuador","Colombia","Mexico",
        "Egypt","Angola","French Guiana","Japan","Burkina Faso","Mozambique","Greece","Slovenia","Guinea-Bissau",
        "Zimbabwe","Cameroon","South Africa","Korea Republic","Chad","Congo DR","Austria","Bulgaria","Türkiye",
        "New Zealand","Georgia","Uzbekistan","Morocco","Bosnia and Herzegovina","Poland","Australia","Saudi Arabia",
        "Chile","Mali","Tanzania","Canada","Montenegro","Zambia","Panama","Jersey","Iceland","Algeria","Curaçao",
        "Finland","Bermuda","Barbados","Congo","Grenada","Montserrat","Liberia","Jamaica","Lithuania","Afghanistan",
        "Malawi","Belize","Guadeloupe","Albania","Somalia","Guyana","British Virgin Islands","Suriname","Gibraltar",
        "Honduras","Mauritius","Great Britain","Russia","Cyprus","Fiji","Thailand","Hong Kong","Latvia",
        "Trinidad and Tobago","Eritrea","North Macedonia","Kosovo","Azerbaijan","Luxembourg","Venezuela","Peru",
        "Israel","Moldova","Estonia","Costa Rica","Armenia","Guinea","Comoros","Kenya","Vanuatu","Malta","Iraq",
        "Dominica","Réunion","Cape Verde Islands","Romania","Liechtenstein","Kazakhstan","Belarus","Benin","Rwanda",
        "Dominican Republic","Iran","Niger","Singapore","Burundi","Madagascar","Togo","Central African Republic",
        "Bolivia","Tajikistan","Martinique","Cuba","China PR","Equatorial Guinea","Gabon","Chinese Taipei",
        "Guatemala","Tunisia","Lebanon","Bahrain","Uganda","Oman","Faroe Islands","Jordan","Haiti","Africa","Syria",
        "St. Lucia","Indonesia","Ethiopia","Philippines","Mauritania","Palestine","Libya","Malaysia","Korea DPR",
        "Nicaragua","South Sudan","Bonaire","São Tomé e Príncipe","St. Kitts and Nevis","El Salvador",
        "New Caledonia","Kyrgyzstan","Isle of Man","Lesotho","United Arab Emirates","Andorra","Mongolia","Namibia",
        "Eswatini","Pakistan","Djibouti","Antigua and Barbuda","Puerto Rico","Cayman Islands",
        "St. Vincent and the Grenadines"
    ]
    birth_sel = st.multiselect("Birth country", sorted(set(BIRTH_COUNTRIES_ALLOWED)), default=[], key="t20_birth_v33")
with p4:
    mv_range = st.slider("Market value (€m)", 0.0, 150.0, (0.0, 150.0), 0.5, key="t20_mv_v33")

# ---------- League filters (display only) ----------
st.markdown("#### 🌍 League filters (display only)")
l1, l2, l3, l4 = st.columns([1.8, 1.2, 1.2, 1.2])

leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df["League"].dropna().astype(str).unique()))
with l1:
    leagues_sel = st.multiselect("League (multi)", leagues_avail, default=[], key="t20_leagues_sel_v33")
with l2:
    regions = ["Europe", "South America", "North America", "Africa", "Asia", "Other"]
    region_sel = st.multiselect("Region (multi)", regions, default=[], key="t20_region_sel_v33")
with l3:
    band_sel = st.multiselect("Band (multi)", [1, 2, 3, 4, 5, 6], default=[], key="t20_band_sel_v33")
with l4:
    band_max = st.selectbox("Or bands ≤", ["— None —", 1, 2, 3, 4, 5, 6], index=0, key="t20_band_max_v33")

# ============================== NEW: RAW METRIC FILTERS (DISPLAY ONLY) ==============================
st.markdown("#### 📏 Raw metric filters (display only)")

RAW_METRIC_OPTIONS = [
    "Defensive duels per 90",
    "Defensive duels won, %",
    "Aerial duels per 90",
    "Aerial duels won, %",
    "PAdj Interceptions",
    "Non-penalty goals per 90",
    "xG per 90",
    "Shots per 90",
    "Crosses per 90",
    "Accurate crosses, %",
    "Dribbles per 90",
    "Successful dribbles, %",
    "Touches in box per 90",
    "Progressive runs per 90",
    "Accelerations per 90",
    "Passes per 90",
    "Accurate passes, %",
    "Forward passes per 90",
    "Accurate forward passes, %",
    "Long passes per 90",
    "Accurate long passes, %",
    "xA per 90",
    "Smart passes per 90",
    "Passes to final third per 90",
    "Passes to penalty area per 90",
    "Deep completions per 90",
    "Progressive passes per 90",
]

# Only show metrics that exist in the dataset
raw_metric_avail = [m for m in RAW_METRIC_OPTIONS if m in df.columns]

rm1, rm2 = st.columns([2.0, 3.0])
with rm1:
    raw_metric_picks = st.multiselect(
        "Add raw metric filters",
        raw_metric_avail,
        default=[],
        key="t20_raw_metric_picks_v33",
    )

raw_metric_rules = []
if raw_metric_picks:
    with rm2:
        st.caption("Each selected metric becomes an AND rule.")
    with st.expander("Configure raw metric rules", expanded=True):
        for m in raw_metric_picks:
            cA, cB, cC = st.columns([2.8, 1.2, 1.2])
            with cA:
                st.markdown(f"**{m}**")
            with cB:
                op = st.selectbox(
                    "Op",
                    [">=", "<="],
                    index=0,
                    key=f"t20_rm_op_{m}_v33",
                    label_visibility="collapsed",
                )
            with cC:
                thr = st.number_input(
                    "Value",
                    value=0.0,
                    step=0.05,
                    format="%.3f",
                    key=f"t20_rm_thr_{m}_v33",
                    label_visibility="collapsed",
                )
            raw_metric_rules.append((m, op, float(thr)))

# ============================== Role threshold UI ==============================
st.markdown("#### 🎯 Role thresholds")

mode_cols = st.columns([1.2, 2.0])
with mode_cols[0]:
    role_match_mode = st.radio("Match mode", ["ANY (recommended)", "ALL"], index=0, horizontal=False, key="t20_role_match_mode_v33")
with mode_cols[1]:
    st.caption("Pick one or more roles, then set a threshold for each. "
               "ANY = player passes if they meet at least one selected role threshold. "
               "ALL = must meet every selected role threshold.")

role_key_for_group = None if group_filter == "All positions" else _role_key_from_group(group_filter)
mode_any = role_match_mode.startswith("ANY")

if group_filter == "All positions":
    st.info("Role thresholds are only configurable when viewing a specific position group.")
    role_picks = []
    role_thresholds = {}
else:
    roles_for_group = sorted(list((ROLE_BUCKETS.get(role_key_for_group, {}) or {}).keys()))
    role_picks = st.multiselect("Roles (this position group)", roles_for_group, default=[], key="t20_role_picks_v33")

    role_thresholds = {}
    if role_picks:
        with st.expander("Set thresholds per selected role", expanded=True):
            for rn in role_picks:
                cA, cB = st.columns([2.5, 1.5])
                with cA:
                    st.markdown(f"**{rn}**")
                with cB:
                    role_thresholds[rn] = st.number_input(
                        "Threshold",
                        0, 100, int(default_thr),
                        key=f"t20_thr_{role_key_for_group}_{rn}_v33",
                        label_visibility="collapsed",
                    )

# ============================== Order-by UI ==============================
st.markdown("#### ↕ Ordering")
order_role = None
order_dir_desc = True

if group_filter != "All positions":
    order_choices = ["Score", "Best role (A→Z)", "Complete Score"]
    order_choices += [f"Role: {r}" for r in (sorted(list((ROLE_BUCKETS.get(role_key_for_group, {}) or {}).keys())))]
    oc1, oc2 = st.columns([2.0, 1.0])
    with oc1:
        order_by = st.selectbox("Order by", order_choices, index=0, key="t20_order_by_v33")
    with oc2:
        order_dir_desc = st.toggle("Descending", value=True, key="t20_order_desc_v33")
    if order_by.startswith("Role: "):
        order_role = order_by.replace("Role: ", "").strip()
else:
    oc1, oc2 = st.columns([2.0, 1.0])
    with oc1:
        order_by = st.selectbox("Order by", ["Score", "Best role (A→Z)", "Complete Score"], index=0, key="t20_order_by_all_v33")
    with oc2:
        order_dir_desc = st.toggle("Descending", value=True, key="t20_order_desc_all_v33")

# ============================== Build scoring pool (ONLY minutes affects pool) ==============================
df_base = df.copy()

# strip tags in key string columns (prevents <span ...> showing)
for c in ["Player", "Team", "League", "Position"]:
    if c in df_base.columns:
        df_base[c] = df_base[c].apply(_strip_tags)

for col in ["Age", "Minutes played"]:
    if col in df_base.columns:
        df_base[col] = pd.to_numeric(df_base[col], errors="coerce")

# youth leagues excluded by default (THIS AFFECTS POOL)
if ("League" in df_base.columns) and (not include_youth_leagues):
    df_base = df_base[~df_base["League"].astype(str).isin(YOUTH_LEAGUES)].copy()

# derive contract year / market value / foot / birth fields (for display filters)
if contract_col in df_base.columns:
    df_base["_contract_year"] = df_base[contract_col].apply(_contract_year)
else:
    df_base["_contract_year"] = np.nan

if mv_col in df_base.columns:
    df_base["_mv_m"] = df_base[mv_col].apply(_parse_market_value_to_m)
else:
    df_base["_mv_m"] = np.nan

if foot_col in df_base.columns:
    df_base["_foot"] = df_base[foot_col].apply(_norm_foot)
else:
    df_base["_foot"] = ""

if birth_col in df_base.columns:
    df_base["_birth_country"] = df_base[birth_col].apply(lambda x: _strip_tags(x))
else:
    df_base["_birth_country"] = ""

if "Position" in df_base.columns:
    df_base["_pos_tok"] = df_base["Position"].apply(_pos_token)
else:
    df_base["_pos_tok"] = ""

# scoring pool: minutes only
if "Minutes played" in df_base.columns:
    df_base = df_base[df_base["Minutes played"].fillna(0).astype(float) >= float(min_minutes_pool)].copy()

df_base["_league_strength"] = df_base["League"].astype(str).map(LEAGUE_STRENGTHS).fillna(0.0)
df_base["_region"] = df_base["League"].astype(str).apply(league_region) if "League" in df_base.columns else "Other"
df_base["_band"] = df_base["League"].astype(str).apply(gbe_league_band) if "League" in df_base.columns else 6

# ============================== Display df (filters do NOT change scoring pool) ==============================
df_disp = df_base.copy()

allowed_pos = POS_GROUPS_LOCAL[group_filter]
if allowed_pos is not None:
    df_disp = df_disp[df_disp["_pos_tok"].isin(allowed_pos)].copy()

# original display filters
if "Age" in df_disp.columns:
    df_disp = df_disp[df_disp["Age"].between(age_band[0], age_band[1])].copy()

if player_q.strip():
    df_disp = df_disp[df_disp["Player"].astype(str).str.contains(player_q.strip(), case=False, na=False)]
if team_q.strip():
    df_disp = df_disp[df_disp["Team"].astype(str).str.contains(team_q.strip(), case=False, na=False)]

# ✅ NEW display-only filters
# Contract year (keeps NaNs visible)
if "_contract_year" in df_disp.columns:
    lo_y, hi_y = int(contract_year_range[0]), int(contract_year_range[1])
    df_disp = df_disp[df_disp["_contract_year"].between(lo_y, hi_y) | df_disp["_contract_year"].isna()].copy()

# Foot
if foot_sel:
    df_disp = df_disp[df_disp["_foot"].isin(set(foot_sel))].copy()

# Birth country
if birth_sel:
    df_disp = df_disp[df_disp["_birth_country"].isin(set(birth_sel))].copy()

# Market value (€m) (keeps NaNs visible)
if "_mv_m" in df_disp.columns:
    lo_mv, hi_mv = float(mv_range[0]), float(mv_range[1])
    df_disp = df_disp[df_disp["_mv_m"].between(lo_mv, hi_mv) | df_disp["_mv_m"].isna()].copy()

# league display filters
if "League" in df_disp.columns:
    df_disp["_league"] = df_disp["League"].astype(str)

    if leagues_sel:
        df_disp = df_disp[df_disp["_league"].isin(set(leagues_sel))].copy()

    if region_sel:
        df_disp = df_disp[df_disp["_region"].isin(set(region_sel))].copy()

    if band_sel:
        df_disp = df_disp[df_disp["_band"].isin(set(band_sel))].copy()

    if band_max != "— None —":
        df_disp = df_disp[df_disp["_band"] <= int(band_max)].copy()

df_disp = df_disp[df_disp["_league_strength"].between(float(league_q[0]), float(league_q[1]))].copy()

# NEW: display-only RAW metric rules
for m, op, thr in raw_metric_rules:
    if m in df_disp.columns:
        s = pd.to_numeric(df_disp[m], errors="coerce")
        if op == ">=":
            df_disp = df_disp[s >= float(thr)].copy()
        else:
            df_disp = df_disp[s <= float(thr)].copy()

if df_disp.empty:
    st.info("No players match the display filters (note: minutes affects the pool).")
    st.stop()

# ============================== Scoring helpers ==============================
def _role_score_series_from_pcts(pcts: pd.DataFrame, spec: dict, idx: pd.Index) -> pd.Series:
    met_w = (spec or {}).get("metrics", {}) or {}
    cols, wts = [], []
    for met, w in met_w.items():
        if met in pcts.columns:
            cols.append(met)
            wts.append(float(w))
    if not cols or sum(wts) <= 0:
        return pd.Series(np.nan, index=idx)
    M = pcts.loc[idx, cols].astype(float)
    w = np.array(wts, dtype=float)
    s = (M.mul(w, axis=1).sum(axis=1) / w.sum())
    return pd.to_numeric(s, errors="coerce")

def _score_subset_against_league_pos_ref(
    df_subset: pd.DataFrame,
    rk: str,
    allowed_pos_tokens: set,
    selected_roles_for_scoring: set | None = None
):
    """
    Computes:
      - _best_role_badge + _best_raw_badge using eligible roles
        (default excludes BADGE_EXCLUDE + LABEL_ONLY, but you can include them if selected)
      - _best_role_display using ALL roles (including label-only + Target Man)
      - _complete_raw using your formula
      - role_scores_by_league for ordering by specific role
    Reference percentiles: df_base filtered to SAME LEAGUE + SAME POS GROUP
    (Outlier Search: uses RAW metric Z-scores mapped to 0..100)
    """
    specs = ROLE_BUCKETS.get(rk, {}) or {}
    out = pd.DataFrame(index=df_subset.index)

    if df_subset.empty:
        out["_best_role_badge"] = ""
        out["_best_raw_badge"] = 0.0
        out["_best_role_display"] = ""
        out["_complete_raw"] = np.nan
        out.attrs["role_scores_by_league"] = {}
        return out

    role_metrics = sorted({m for r in specs.values() for m in (r.get("metrics", {}) or {}).keys()}) if specs else []
    complete_metrics = sorted(set((COMPLETE_WEIGHTS.get(rk, {}) or {}).keys()))
    need_metrics = sorted(set(role_metrics) | set(complete_metrics))

    base_pos = df_base[df_base["_pos_tok"].isin(allowed_pos_tokens)].copy()

    # default eligible roles for numeric scoring
    eligible_default = [rname for rname in specs.keys() if (not _is_badge_excluded(rname)) and (not _is_label_only(rname))]

    # allow excluded/label-only roles to contribute ONLY if user selected them
    selected_roles_for_scoring = set(selected_roles_for_scoring or [])
    eligible_selected = [r for r in specs.keys() if r in selected_roles_for_scoring]

    eligible_roles_for_scoring = list(dict.fromkeys(eligible_default + eligible_selected))

    best_role_badge_s = pd.Series("", index=df_subset.index, dtype="object")
    best_raw_s        = pd.Series(0.0, index=df_subset.index, dtype="float")
    best_role_disp_s  = pd.Series("", index=df_subset.index, dtype="object")
    comp_s            = pd.Series(np.nan, index=df_subset.index, dtype="float")

    role_scores_by_league = {}

    for lg, part in df_subset.groupby(df_subset["League"].astype(str)):
        ref = base_pos[base_pos["League"].astype(str) == str(lg)]
        pcts = _scores_for_ref(ref, need_metrics, bool(outlier_search)) if (not ref.empty and need_metrics) else pd.DataFrame(index=ref.index)

        idx = part.index.intersection(pcts.index)
        if len(idx) == 0:
            continue

        # complete score
        comp_s.loc[idx] = _complete_score_from_pcts(pcts, rk, idx)

        # role scores table for all roles
        if specs and role_metrics:
            role_scores_tbl = pd.DataFrame(index=idx)
            for rname, spec in specs.items():
                role_scores_tbl[rname] = _role_score_series_from_pcts(pcts, spec, idx)

            role_scores_by_league[str(lg)] = role_scores_tbl

            # BEST ROLE DISPLAY: use ALL roles (includes label-only + target man)
            best_role_disp_s.loc[idx] = role_scores_tbl.idxmax(axis=1).astype(str).fillna("")

            # BADGE/base best role: eligible scoring roles only
            cols = [c for c in eligible_roles_for_scoring if c in role_scores_tbl.columns]
            if cols:
                elig_tbl = role_scores_tbl[cols]
                best_role_badge_s.loc[idx] = elig_tbl.idxmax(axis=1).astype(str).fillna("")
                best_raw_s.loc[idx]        = pd.to_numeric(elig_tbl.max(axis=1), errors="coerce").fillna(0.0)
        else:
            role_scores_by_league[str(lg)] = pd.DataFrame(index=idx)

    out["_best_role_badge"]   = best_role_badge_s.fillna("")
    out["_best_raw_badge"]    = pd.to_numeric(best_raw_s, errors="coerce").fillna(0.0)
    out["_best_role_display"] = best_role_disp_s.fillna("")
    out["_complete_raw"]      = pd.to_numeric(comp_s, errors="coerce")

    out.attrs["role_scores_by_league"] = role_scores_by_league
    return out

def _apply_role_thresholds(df_scored: pd.DataFrame, rk: str, allowed_pos_tokens: set, role_thresholds: dict, mode_any: bool) -> pd.DataFrame:
    if not role_thresholds:
        return df_scored

    specs_all = ROLE_BUCKETS.get(rk, {}) or {}
    role_thresholds = {r: float(t) for r, t in role_thresholds.items() if r in specs_all and r}
    if not role_thresholds:
        return df_scored

    need_metrics = sorted({m for r in role_thresholds.keys() for m in (specs_all[r].get("metrics", {}) or {}).keys()})
    base_pos = df_base[df_base["_pos_tok"].isin(allowed_pos_tokens)].copy()

    keep_mask = pd.Series(False if mode_any else True, index=df_scored.index)

    for lg, part in df_scored.groupby(df_scored["League"].astype(str)):
        ref = base_pos[base_pos["League"].astype(str) == str(lg)]
        pcts = _scores_for_ref(ref, need_metrics, bool(outlier_search)) if (not ref.empty and need_metrics) else pd.DataFrame(index=ref.index)

        idx = part.index.intersection(pcts.index)
        if len(idx) == 0:
            keep_mask.loc[part.index] = False
            continue

        hits = []
        for rname, thr in role_thresholds.items():
            rs = _role_score_series_from_pcts(pcts, specs_all[rname], idx)
            hits.append((rs >= float(thr)).fillna(False))

        if mode_any:
            ok = hits[0].copy()
            for h in hits[1:]:
                ok |= h
        else:
            ok = hits[0].copy()
            for h in hits[1:]:
                ok &= h

        keep_mask.loc[idx] = ok.reindex(idx).fillna(False).values
        miss = part.index.difference(idx)
        if len(miss) > 0:
            keep_mask.loc[miss] = False

    return df_scored[keep_mask].copy()

# ============================== Run scoring ==============================
if group_filter != "All positions":
    rk = role_key_for_group
    allowed_pos_tokens = allowed_pos

    df_scored = df_disp.copy()

    # IMPORTANT: allow label-only + target man to affect score ONLY if selected
    selected_for_scoring = set(role_thresholds.keys()) if role_thresholds else set()

    scored = _score_subset_against_league_pos_ref(df_scored, rk, allowed_pos_tokens, selected_for_scoring)

    df_scored["_best_role_badge"]   = scored["_best_role_badge"]
    df_scored["_best_raw_badge"]    = scored["_best_raw_badge"]
    df_scored["_best_role_display"] = scored["_best_role_display"]
    df_scored["_complete_raw"]      = scored["_complete_raw"]
    role_scores_by_league = scored.attrs.get("role_scores_by_league", {})

    if role_thresholds:
        df_scored = _apply_role_thresholds(df_scored, rk, allowed_pos_tokens, role_thresholds, mode_any)

    if df_scored.empty:
        st.info("No players meet the selected role threshold rules.")
        st.stop()

    if order_role:
        order_vals = pd.Series(np.nan, index=df_scored.index)
        for lg, part in df_scored.groupby(df_scored["League"].astype(str)):
            tbl = role_scores_by_league.get(str(lg), None)
            if isinstance(tbl, pd.DataFrame) and (order_role in tbl.columns):
                idx = part.index.intersection(tbl.index)
                order_vals.loc[idx] = pd.to_numeric(tbl.loc[idx, order_role], errors="coerce")
        df_scored["_order_role_val"] = pd.to_numeric(order_vals, errors="coerce").fillna(-1.0)

else:
    df_scored_chunks = []
    GROUPS = [
        ("CF", {"CF"}),
        ("CB", {"CB", "LCB", "RCB"}),
        ("FB", {"RB", "LB", "RWB", "LWB"}),
        ("CM", {"DMF", "CMF", "LCMF", "RCMF", "LDMF", "RDMF"}),
        ("ATT", {"RW", "RWF", "LW", "LWF", "AMF", "RAMF", "LAMF"}),
    ]
    for rk, toks in GROUPS:
        part = df_disp[df_disp["_pos_tok"].isin(toks)].copy()
        if part.empty:
            continue

        scored = _score_subset_against_league_pos_ref(part, rk, toks, set())  # no selected roles in all-positions mode

        part["_best_role_badge"]   = scored["_best_role_badge"]
        part["_best_raw_badge"]    = scored["_best_raw_badge"]
        part["_best_role_display"] = scored["_best_role_display"]
        part["_complete_raw"]      = scored["_complete_raw"]
        part["_rk"] = rk
        df_scored_chunks.append(part)

    if not df_scored_chunks:
        st.info("No players found across position groups for current display filters.")
        st.stop()

    df_scored = pd.concat(df_scored_chunks, axis=0)

# ============================== Base score (best-role vs complete) ==============================
best_role_raw = pd.to_numeric(df_scored["_best_raw_badge"], errors="coerce").fillna(0.0)
complete_raw  = pd.to_numeric(df_scored["_complete_raw"], errors="coerce")

base_raw = complete_raw.where(complete_raw.notna(), best_role_raw) if use_complete_base else best_role_raw
df_scored["_base_raw"] = pd.to_numeric(base_raw, errors="coerce").fillna(0.0)

# ============================== Final score (base → optional league blend) ==============================
ls = pd.to_numeric(df_scored["_league_strength"], errors="coerce").fillna(0.0)

df_scored["_score"] = ((1.0 - float(BETA_BADGE)) * df_scored["_base_raw"] + float(BETA_BADGE) * ls) if use_league_weighting else df_scored["_base_raw"]
df_scored["_score"] = pd.to_numeric(df_scored["_score"], errors="coerce").fillna(0.0)

# ============================== Ordering + top N ==============================
if order_by == "Best role (A→Z)":
    df_scored = df_scored.sort_values(["_best_role_display", "_score"], ascending=[True, False]).copy()
elif order_by == "Complete Score":
    tmp = pd.to_numeric(df_scored["_complete_raw"], errors="coerce")
    df_scored["_order_complete"] = tmp.where(tmp.notna(), df_scored["_base_raw"]).fillna(0.0)
    df_scored = df_scored.sort_values(["_order_complete", "_score"], ascending=[not order_dir_desc, not order_dir_desc]).copy()
elif order_role is not None and ("_order_role_val" in df_scored.columns):
    df_scored = df_scored.sort_values(["_order_role_val", "_score"], ascending=[not order_dir_desc, not order_dir_desc]).copy()
else:
    df_scored = df_scored.sort_values("_score", ascending=not order_dir_desc).copy()

df_scored = df_scored.head(int(top_n)).copy()
df_scored.insert(0, "Rank", range(1, len(df_scored) + 1))

# ============================== Images (safe) ==============================
def _photo_url(row) -> str:
    try:
        u = resolve_player_photo(
            str(row.get("Player", "")),
            str(row.get("Team", "")),
            str(row.get("League", "")),
        )
        return u if _is_http_url(u) else PLACEHOLDER_IMG
    except Exception:
        return PLACEHOLDER_IMG

def _crest_url(row) -> str:
    try:
        u = resolve_team_crest(
            str(row.get("Team", "")),
            str(row.get("League", "")),
        )
        return u if _is_http_url(u) else FALLBACK_BADGE
    except Exception:
        return FALLBACK_BADGE

# ============================== Tight CSS table ==============================
st.markdown(
    """
<style>
.t20-wrap{ margin-top:10px; }
.t20-table{
  width:100%;
  border-collapse:separate;
  border-spacing:0;
  background:#0b1220;
  border:1px solid rgba(148,163,184,.18);
  border-radius:14px;
  overflow:hidden;
}
.t20-table thead th{
  text-align:left;
  font-size:12px;
  letter-spacing:.02em;
  color:#cbd5e1;
  padding:10px 10px;
  background:rgba(15,23,42,.70);
  border-bottom:1px solid rgba(148,163,184,.16);
  white-space:nowrap;
}
.t20-table tbody td{
  padding:8px 10px;
  border-bottom:1px solid rgba(148,163,184,.10);
  color:#e5e7eb;
  font-size:13px;
  vertical-align:middle;
  white-space:nowrap;
}
.t20-table tbody tr:hover td{ background:rgba(59,130,246,.06); }

.t20-rank{ font-weight:900; width:48px; }

.t20-player{ display:flex; align-items:center; gap:8px; min-width:180px; }
.t20-photo{
  width:36px; height:36px; border-radius:9px; object-fit:cover;
  border:1px solid rgba(148,163,184,.22);
  background:#0f172a;
}
.t20-name{
  font-weight:900;
  line-height:1.00;
  display:flex;
  align-items:center;
  gap:6px;
}
.t20-flagimg{
  width:18px;
  height:12px;
  border-radius:3px;
  object-fit:cover;
  border:1px solid rgba(148,163,184,.22);
  background:#0f172a;
}

.t20-team{ display:flex; align-items:center; gap:8px; min-width:160px; }
.t20-crest{
  width:18px; height:18px; border-radius:6px; object-fit:contain;
  border:1px solid rgba(148,163,184,.22);
  background:#0f172a;
}
.t20-teamname{ font-weight:850; line-height:1.00; }

.t20-muted{ color:#94a3b8; font-weight:650; }
.t20-role{ font-weight:850; }

.t20-pill{
  display:inline-flex; align-items:center; justify-content:center;
  min-width:42px;
  padding:6px 10px;
  border-radius:12px;
  color:white;
  font-weight:900;
  border:1px solid rgba(255,255,255,.10);
}
</style>
""",
    unsafe_allow_html=True
)

# ============================== Render HTML table ==============================
rows_html = []
for _, r in df_scored.iterrows():
    player = _esc(_strip_tags(r.get("Player", "")))
    team   = _esc(_strip_tags(r.get("Team", "")))
    league = _esc(_strip_tags(r.get("League", "")))
    band   = _esc(r.get("_band", ""))
    pos    = _esc(_strip_tags(r.get("Position", "")))
    age    = _esc(r.get("Age", "—"))

    birth_country = _strip_tags(r.get("_birth_country", ""))
    flag_url = _country_to_flag_url(birth_country)
    flag_html = f"<img class='t20-flagimg' src='{_esc(flag_url)}' title='{_esc(birth_country)}' />" if _is_http_url(flag_url) else ""

    best_role_raw_name = _strip_tags(r.get("_best_role_display", ""))
    best_role_disp = _esc(_format_best_role_display(best_role_raw_name, r.get("Position", "")))

    score = float(pd.to_numeric(r.get("_score", 0.0), errors="coerce") or 0.0)
    pill_col = _div_color_hex(score)

    photo = _esc(_photo_url(r))
    crest = _esc(_crest_url(r))
    crest_img = f"<img class='t20-crest' src='{crest}' />" if crest else ""

    rows_html.append(
        f"""
<tr>
  <td class="t20-rank">{int(r.get("Rank",0))}</td>
  <td>
    <div class="t20-player">
      <img class="t20-photo" src="{photo}" />
      <span class="t20-name">{player}{flag_html}</span>
    </div>
  </td>
  <td>
    <div class="t20-team">
      {crest_img}
      <span class="t20-teamname">{team}</span>
    </div>
  </td>
  <td><span class="t20-muted">{league}</span></td>
  <td><span class="t20-muted">{band}</span></td>
  <td><span class="t20-muted">{pos}</span></td>
  <td>{age}</td>
  <td class="t20-role">{best_role_disp}</td>
  <td><span class="t20-pill" style="background:{pill_col};">{int(round(score))}</span></td>
</tr>
"""
    )

table_html = f"""
<div class="t20-wrap">
  <table class="t20-table">
    <thead>
      <tr>
        <th>Rank</th>
        <th>Player</th>
        <th>Team</th>
        <th>League</th>
        <th>Band</th>
        <th>Position</th>
        <th>Age</th>
        <th>Best role</th>
        <th>Score</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</div>
"""

st.markdown(table_html, unsafe_allow_html=True)

with st.expander("Debug (scoring fields)", expanded=False):
    show_cols = [
        "Player", "Team", "League", "_region", "_band", "Position", "Age",
        "_contract_year", "_foot", "_birth_country", "_mv_m",
        "_best_role_display", "_best_role_badge", "_best_raw_badge",
        "_complete_raw", "_base_raw", "_league_strength", "_score"
    ]
    # include raw-metric columns for sanity if user used those filters
    for m, _, _ in raw_metric_rules:
        if m not in show_cols:
            show_cols.append(m)

    show_cols = [c for c in show_cols if c in df_scored.columns]
    st.dataframe(df_scored[show_cols], use_container_width=True)

# ================= PLAYER DROP DEBUGGER =================
with st.expander("🕵️ Why is my player missing?", expanded=False):

    q = st.text_input("Type player name", "")

    if q.strip():
        q = q.strip()

        def find(df):
            return df[df["Player"].str.contains(q, case=False, na=False)]

        b = find(df_base)
        d = find(df_disp)
        s = find(df_scored)

        st.markdown("### Presence check")
        st.write({
            "In pool (df_base)": len(b),
            "After display filters (df_disp)": len(d),
            "After scoring/thresholds (df_scored)": len(s),
        })

        if not b.empty:
            st.markdown("### Pool row (df_base)")
            st.dataframe(
                b[["Player","Team","League","Position","Age","Minutes played"]]
                .head(3),
                use_container_width=True
            )

        if b.empty:
            st.error("❌ Player not even in scoring pool → MINUTES or youth league filter removed them")

        elif d.empty:
            st.error("❌ Player removed by DISPLAY FILTERS")
            st.dataframe(
                b[["Age","_contract_year","_foot","_birth_country","_mv_m",
                   "_region","_band","_league_strength"]]
                .head(3),
                use_container_width=True
            )

        elif s.empty:
            st.error("❌ Player removed during SCORING stage")
            st.info("Most common causes:")
            st.markdown("""
            • Wrong position token for selected group  
            • Role thresholds too strict  
            • No league+position reference group  
            • Raw metric rules  
            """)

            # show their role scores if available
            try:
                lg = d.iloc[0]["League"]
                tbl = role_scores_by_league.get(str(lg), None)
                if isinstance(tbl, pd.DataFrame):
                    pid = d.index[0]
                    st.markdown("### Their role scores")
                    st.dataframe(tbl.loc[[pid]], use_container_width=True)
            except:
                pass

        else:
            st.success("✅ Player survived all filters — just outside Top N")

            st.markdown("### Final row")
            st.dataframe(
                s[[
                    "Player","Team","League","Position","Age",
                    "_best_role_display","_best_raw_badge",
                    "_complete_raw","_score"
                ]],
                use_container_width=True
            )
# =======================================================


# ============================ END SHORTLIST (T20) — ELITE TABLE (v3.3++) ============================



# ======================== SECTION B (v2 — FULL, TITLES OFF, FULL LABELS) ========================
st.markdown("---")
st.header("Section B — League Comparison Radar & Tables")

import numpy as _np
import pandas as _pd
import io as _io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------- helpers ----------
def _safe_div(num, den):
    num = _pd.to_numeric(num, errors="coerce")
    den = _pd.to_numeric(den, errors="coerce").replace(0, _np.nan)
    return (num / den).fillna(0.0)

def _pct_from_rank(rank:int, total:int) -> int:
    if total <= 1:
        return 100
    return int(round((1 - (rank - 1) / (total - 1)) * 100))

def _polar_bars(metrics_labels, percentiles):
    # Bars only, no numeric labels, no figure title
    color_scale = ["#be2a3e", "#e25f48", "#f88f4d", "#f4d166", "#90b960", "#4b9b5f", "#22763f"]
    cmap = LinearSegmentedColormap.from_list("custom_scale", color_scale)
    normalized = [max(0, min(100, p))/100 for p in percentiles]
    bar_colors = [cmap(p) for p in normalized]

    N = len(metrics_labels)
    angles = _np.linspace(0, 2*_np.pi, N, endpoint=False)
    bar_width = (2*_np.pi/N)*0.85

    fig = plt.figure(figsize=(8.5, 6.5))
    fig.patch.set_facecolor('#0a0f1c')
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88], polar=True)
    ax.set_facecolor('#0a0f1c')
    ax.set_rlim(0, 100)

    # background segments
    for i in range(N):
        ax.bar(angles[i], 105, width=bar_width, color='#ffffff22', edgecolor=None, bottom=0, linewidth=0, zorder=0)

    # data bars — NO numeric value labels
    for i in range(N):
        ax.bar(angles[i], percentiles[i], width=bar_width, color=bar_colors[i],
               edgecolor='white', linewidth=1.4, zorder=2)

    # metric labels (full names)
    label_radius = 135
    for i, lab in enumerate(metrics_labels):
        ax.text(angles[i], label_radius, lab.upper(), ha='center', va='center',
                fontsize=10, weight='bold', color='white')

    ax.set_xticks([]); ax.set_yticks([]); ax.spines['polar'].set_visible(False); ax.grid(False)
    return fig

def _download_df_button(df_to_dl: _pd.DataFrame, fname: str, label: str):
    buf = _io.BytesIO()
    df_to_dl.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=fname, mime="text/csv")


# ---------- role configs (FULL LABELS) ----------
def _cfg(role_key: str):
    if role_key == "cb":
        require = ['Aerial duels per 90','Defensive duels per 90','Passes per 90','Forward passes per 90',
                   'Progressive passes per 90','Progressive runs per 90','PAdj Interceptions','Shots blocked per 90']
        def pos_ok(s): s=str(s).upper().strip(); return s.startswith(("CB","RCB","LCB"))
        def compute(df):
            out=df.copy()
            out["Pass Verticality"]= _safe_div(out['Forward passes per 90'], out['Passes per 90'])
            out["Pass Volume"]= _pd.to_numeric(out['Passes per 90'],errors="coerce")
            out["Positional Demand"]= _pd.to_numeric(out['PAdj Interceptions'],errors="coerce")+_pd.to_numeric(out['Shots blocked per 90'],errors="coerce")
            out["Defensive Volume"]= _pd.to_numeric(out['Defensive duels per 90'],errors="coerce")
            out["Progression Volume"]= _pd.to_numeric(out['Progressive passes per 90'],errors="coerce")+_pd.to_numeric(out['Progressive runs per 90'],errors="coerce")
            out["Aerial Volume"]= _pd.to_numeric(out['Aerial duels per 90'],errors="coerce")
            return out
        agg_cols=["Pass Volume","Pass Verticality","Progression Volume","Defensive Volume","Positional Demand","Aerial Volume"]
        label_map={
            "Pass Volume":"Pass Volume",
            "Pass Verticality":"Pass Verticality",
            "Progression Volume":"Progression Volume",
            "Defensive Volume":"Defensive Volume",
            "Positional Demand":"Positional Demand",
            "Aerial Volume":"Aerial Volume"
        }
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, label_map=label_map, title="Center Backs")

    if role_key == "fb":
        require=['Passes per 90','Forward passes per 90','Progressive passes per 90','Progressive runs per 90',
                 'Defensive duels per 90','PAdj Interceptions','Aerial duels per 90','xA per 90','Crosses per 90',
                 'Touches in box per 90','Shots per 90','Passes to penalty area per 90','Accurate passes, %','Dribbles per 90']
        def pos_ok(s): s=str(s).upper().strip(); return s.startswith(("LB","LWB","RB","RWB"))
        def compute(df):
            out=df.copy()
            out["Pass Verticality"]= _safe_div(out['Forward passes per 90'], out['Passes per 90'])
            out["Progression Volume"]= _pd.to_numeric(out['Progressive passes per 90'],errors="coerce")+_pd.to_numeric(out['Progressive runs per 90'],errors="coerce")
            out["Ball Carrying"]= 0.6*_pd.to_numeric(out['Dribbles per 90'],errors="coerce")+0.4*_pd.to_numeric(out['Progressive runs per 90'],errors="coerce")
            out["Attacking Contribution"]= (0.4*_pd.to_numeric(out['xA per 90'],errors="coerce")
                                           +0.2*_pd.to_numeric(out['Crosses per 90'],errors="coerce")
                                           +0.2*_pd.to_numeric(out['Touches in box per 90'],errors="coerce")
                                           +0.1*_pd.to_numeric(out['Shots per 90'],errors="coerce")
                                           +0.1*_pd.to_numeric(out['Passes to penalty area per 90'],errors="coerce"))
            out["Defensive Volume"]= (0.5*_pd.to_numeric(out['Defensive duels per 90'],errors="coerce")
                                     +0.3*_pd.to_numeric(out['PAdj Interceptions'],errors="coerce")
                                     +0.2*_pd.to_numeric(out['Aerial duels per 90'],errors="coerce"))
            out["Pass Volume"]= _pd.to_numeric(out['Passes per 90'],errors="coerce")
            out["Retention"]= _pd.to_numeric(out['Accurate passes, %'],errors="coerce")
            return out
        agg_cols=["Pass Volume","Pass Verticality","Progression Volume","Ball Carrying","Attacking Contribution","Defensive Volume","Retention"]
        label_map={
            "Pass Volume":"Pass Volume",
            "Pass Verticality":"Pass Verticality",
            "Progression Volume":"Progression Volume",
            "Ball Carrying":"Ball Carrying",
            "Attacking Contribution":"Attacking Contribution",
            "Defensive Volume":"Defensive Volume",
            "Retention":"Retention"
        }
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, label_map=label_map, title="Fullbacks")

    if role_key == "cm":
        require=['Passes per 90','Forward passes per 90','Progressive passes per 90','Progressive runs per 90',
                 'Defensive duels per 90','PAdj Interceptions','Touches in box per 90','Shots per 90','Accurate passes, %']
        def pos_ok(s): s=str(s).upper().strip(); return s.startswith(("DMF","LDMF","RDMF","LCMF","RCMF","CMF"))
        def compute(df):
            out=df.copy()
            out["Pass Verticality"]= _safe_div(out['Forward passes per 90'], out['Passes per 90'])
            out["Pass Volume"]= _pd.to_numeric(out['Passes per 90'],errors="coerce")
            out["Progression Volume"]= _pd.to_numeric(out['Progressive passes per 90'],errors="coerce")+_pd.to_numeric(out['Progressive runs per 90'],errors="coerce")
            out["Defensive Volume"]= _pd.to_numeric(out['Defensive duels per 90'],errors="coerce")
            out["Interception Volume"]= _pd.to_numeric(out['PAdj Interceptions'],errors="coerce")
            out["Retention"]= _pd.to_numeric(out['Accurate passes, %'],errors="coerce")
            return out
        agg_cols=["Pass Volume","Pass Verticality","Progression Volume","Defensive Volume","Interception Volume","Retention"]
        label_map={
            "Pass Volume":"Pass Volume",
            "Pass Verticality":"Pass Verticality",
            "Progression Volume":"Progression Volume",
            "Defensive Volume":"Defensive Volume",
            "Interception Volume":"Interception Volume",
            "Retention":"Retention"
        }
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, label_map=label_map, title="Central Midfielders")

    if role_key == "attack":
        require = [
            'Accurate passes, %','xG per 90','Non-penalty goals per 90','Touches in box per 90','xA per 90',
            'Passes to penalty area per 90','Passes per 90','Progressive passes per 90','Passes to final third per 90',
            'Dribbles per 90','Progressive runs per 90'
        ]

        import re

        def pos_ok(s: str) -> bool:
            # normalise
            s = str(s).upper().strip()

            # take the first chunk if "RW/LW", "RW, LW", "RW ST", etc.
            main = re.split(r'[/,]', s)[0].strip()
            main = main.split()[0]  # also cut anything after a space

            # pure wingers, but NOT wing-backs
            # RW ✅, LW ✅, RW/LW ✅, RW ST ✅
            # RWB ❌, LWB ❌, RB ❌, LB ❌
            if main in ("RW", "LW"):
                return True

            # wide forwards + AM roles
            prefixes = ("RWF", "LWF", "LAMF", "RAMF", "AMF")
            return main.startswith(prefixes)

        def compute(df):
            out = df.copy()
            out["Retention Style"] = _pd.to_numeric(out['Accurate passes, %'], errors="coerce")
            out["Goal Threat"] = (
                0.4 * _pd.to_numeric(out['xG per 90'], errors="coerce")
                + 0.4 * _pd.to_numeric(out['Non-penalty goals per 90'], errors="coerce")
                + 0.2 * _pd.to_numeric(out['Touches in box per 90'], errors="coerce")
            )
            out["Creativity Threat"] = (
                0.65 * _pd.to_numeric(out['xA per 90'], errors="coerce")
                + 0.35 * _pd.to_numeric(out['Passes to penalty area per 90'], errors="coerce")
            )
            out["Passing Volume"] = _pd.to_numeric(out['Passes per 90'], errors="coerce")
            out["Deeper Playmaking"] = (
                0.5 * _pd.to_numeric(out['Progressive passes per 90'], errors="coerce")
                + 0.5 * _pd.to_numeric(out['Passes to final third per 90'], errors="coerce")
            )
            out["Ball Carrying"] = (
                0.6 * _pd.to_numeric(out['Dribbles per 90'], errors="coerce")
                + 0.4 * _pd.to_numeric(out['Progressive runs per 90'], errors="coerce")
            )
            return out

        agg_cols = [
            "Retention Style","Goal Threat","Creativity Threat",
            "Passing Volume","Deeper Playmaking","Ball Carrying"
        ]
        label_map = {
            "Retention Style": "Retention Style",
            "Goal Threat": "Goal Threat",
            "Creativity Threat": "Creativity Threat",
            "Passing Volume": "Passing Volume",
            "Deeper Playmaking": "Deeper Playmaking",
            "Ball Carrying": "Ball Carrying"
        }
        return dict(
            pos_filter=pos_ok,
            require_cols=require,
            compute_metrics=compute,
            agg_cols=agg_cols,
            label_map=label_map,
            title="Attackers"
        )



    if role_key == "cf":
        require=['Touches in box per 90','xG per 90','Dribbles per 90','Progressive runs per 90',
                 'Aerial duels per 90','Aerial duels won, %','Passes per 90','Non-penalty goals per 90','Accurate passes, %']
        def pos_ok(s): return str(s).upper().strip().startswith("CF")
        def compute(df):
            out=df.copy()
            out["Opportunities"]= 0.7*_pd.to_numeric(out['Touches in box per 90'],errors="coerce")+0.3*_pd.to_numeric(out['xG per 90'],errors="coerce")
            out["Ball Carrying"]= 0.65*_pd.to_numeric(out['Dribbles per 90'],errors="coerce")+0.35*_pd.to_numeric(out['Progressive runs per 90'],errors="coerce")
            out["Aerial Requirement"]= _pd.to_numeric(out['Aerial duels per 90'],errors="coerce")*_pd.to_numeric(out['Aerial duels won, %'],errors="coerce")/100.0
            out["Passing Volume"]= _pd.to_numeric(out['Passes per 90'],errors="coerce")
            out["Goal Output"]= _pd.to_numeric(out['Non-penalty goals per 90'],errors="coerce")
            out["Retention"]= _pd.to_numeric(out['Accurate passes, %'],errors="coerce")
            return out
        agg_cols=["Opportunities","Ball Carrying","Aerial Requirement","Passing Volume","Goal Output","Retention"]
        label_map={
            "Opportunities":"Opportunities",
            "Ball Carrying":"Carrying Outlet",
            "Aerial Requirement":"Aerial Volume",
            "Passing Volume":"Passing Volume",
            "Goal Output":"Goal Output",
            "Retention":"Retention"
        }
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, label_map=label_map, title="Strikers")


# ---------- per-role UI/logic ----------
def _sectionB_for_role(role_key: str):
    cfg=_cfg(role_key)
    leagues = sorted([str(x) for x in df["League"].dropna().unique()])
    topA, topB, topC = st.columns([1.4, 1.6, 1.6])

    with topA:
        included_league = st.selectbox(f"League ({cfg['title']})", leagues, key=f"secB_league_{role_key}")

    league_df = df[df["League"].astype(str)==included_league].copy()
    missing = [c for c in cfg["require_cols"] if c not in league_df.columns]
    if missing:
        st.info(f"Missing columns for {cfg['title']}: {', '.join(missing)}")
        return

    # numeric cleaning for filters
    for c in ("Minutes played","Age","Goals"):
        if c in league_df.columns: league_df[c]=_pd.to_numeric(league_df[c], errors="coerce")

    # position subset
    league_df = league_df[league_df["Position"].apply(cfg["pos_filter"])].dropna(subset=cfg["require_cols"])
    if league_df.empty:
        st.info(f"No {cfg['title']} in this league with required stats.")
        return

    # filters (no market value)
    fl1, fl2, fl3 = st.columns([1.6,1.6,1.6])
    with fl1:
        teams = sorted(league_df["Team"].dropna().astype(str).unique())
        teams_selected = st.multiselect(
                 "Filter teams",
                   teams,
                   default=teams,
                   key=f"secB_teams_{role_key}_{included_league}"
)

        min_minutes, max_minutes = st.slider("Minutes played", 0, 6000, (750, 6000), key=f"secB_min_{role_key}")
        a_min = int(_np.nanmin(league_df["Age"])) if league_df["Age"].notna().any() else 16
        a_max = int(_np.nanmax(league_df["Age"])) if league_df["Age"].notna().any() else 50
        min_age, max_age = st.slider("Age", a_min, a_max, (16, 50), key=f"secB_age_{role_key}")
    with fl3:
        q = st.text_input("Quick player search (optional)", "", key=f"secB_q_{role_key}")

    pool = league_df[
        league_df["Team"].astype(str).isin(teams_selected)
        & league_df["Minutes played"].between(min_minutes, max_minutes)
        & league_df["Age"].between(min_age, max_age)
    ].copy()
    if q.strip():
        s=q.strip().lower()
        pool = pool[pool["Player"].astype(str).str.lower().str.contains(s, na=False)]

    if pool.empty:
        st.info("No players after filters.")
        return

    # compute metrics
    pool = cfg["compute_metrics"](pool).copy()

    # choose mode/team/player
    left, right = st.columns([1.8, 2.2])
    with left:
        compare_mode = st.radio("Compare", ["Team average","Specific player"], horizontal=True, key=f"secB_mode_{role_key}")
        teams_pool = sorted(pool["Team"].dropna().astype(str).unique())
        target_team = st.selectbox("Target team", teams_pool, key=f"secB_team_{role_key}")

    # team averages for league (after filters)
    agg = pool.groupby("Team")[cfg["agg_cols"]].mean().reset_index()

    if compare_mode == "Team average":
        if target_team not in agg["Team"].values:
            st.info("Target team has no eligible players in filtered set.")
            return
        target_vals = agg.set_index("Team").loc[target_team, cfg["agg_cols"]].to_dict()
        label_subject = f"{target_team} AVG"
        exclude_label = target_team  # drop real team before adding pseudo
        team_players_used = pool[pool["Team"].astype(str) == target_team].copy()
    else:
        players_pool = sorted(pool[pool["Team"].astype(str) == target_team]["Player"].dropna().astype(str).unique())
        if not players_pool:
            st.info("No eligible players on selected team under current filters.")
            return
        sel_player = st.selectbox("Player", players_pool, key=f"secB_player_{role_key}")
        prow = pool[pool["Player"].astype(str)==sel_player].head(1)
        if prow.empty:
            st.info("Pick a player above.")
            return
        target_vals = prow[cfg["agg_cols"]].iloc[0].to_dict()
        label_subject = sel_player
        exclude_label = str(prow["Team"].iloc[0])  # exclude player's team from benchmark table
        team_players_used = None

    # ---------- right side: team averages table ----------
    with right:
        st.markdown("**Team role averages (post-filter league scope)**")
        sort_col = st.selectbox("Sort by metric", ["Team"] + cfg["agg_cols"], index=0, key=f"secB_sort_{role_key}")
        asc = st.checkbox("Ascending", False, key=f"secB_sort_asc_{role_key}")
        st.dataframe(agg.sort_values(sort_col, ascending=asc), use_container_width=True)
        _download_df_button(agg, f"{cfg['title'].replace(' ','_')}_team_averages.csv", "⬇️ Download team averages (CSV)")

    # ---------- ranks & percentiles (no N+1) ----------
    rows=[]
    for met in cfg["agg_cols"]:
        temp = agg[["Team",met]].copy()
        temp = temp[temp["Team"] != exclude_label].copy()  # remove the team we're substituting/excluding
        val = float(target_vals[met])
        pseudo = _pd.DataFrame({"Team":[label_subject], met:[val]})
        temp = _pd.concat([temp, pseudo], ignore_index=True)
        temp = temp.drop_duplicates(subset="Team", keep="last")
        temp = temp.sort_values(by=met, ascending=False, kind="mergesort").reset_index(drop=True)

        rk = int(temp.index[temp["Team"]==label_subject][0]) + 1
        tot = int(temp.shape[0])
        pct = _pct_from_rank(rk, tot)
        rows.append((met, rk, tot, pct, val))

    rank_df = _pd.DataFrame(rows, columns=["Metric","Rank","Total teams","Percentile","Target value"])
    rank_df["Metric"] = rank_df["Metric"].map(lambda x: cfg["label_map"].get(x,x))

    # ---------- per-metric descending league tables ----------
    st.markdown("### 📊 Per-metric league tables (descending)")
    for met_key in cfg["agg_cols"]:
        pretty = cfg["label_map"].get(met_key, met_key)
        tmp = agg[["Team", met_key]].copy().rename(columns={met_key: pretty})
        tmp = tmp.sort_values(by=pretty, ascending=False).reset_index(drop=True)
        tmp.insert(0, "Rank", _np.arange(1, len(tmp)+1))
        with st.expander(f"{pretty} — league table"):
            st.dataframe(tmp, use_container_width=True)

    # ---------- header + summary ----------
    who = f"{label_subject} ({cfg['title']}) vs league team averages"
    st.subheader(f"📈 {who}")
    st.dataframe(rank_df, use_container_width=True)
    _download_df_button(rank_df, f"{cfg['title'].replace(' ','_')}_rank_summary_{label_subject.replace(' ','_')}.csv",
                        "⬇️ Download ranking summary (CSV)")

    # ---------- players used (team-average mode only) ----------
    if team_players_used is not None and not team_players_used.empty:
        st.markdown("### 🧩 Players used for team average (with minutes & in-team ranks)")
        for c in cfg["agg_cols"]:
            team_players_used[f"Rank in team ({cfg['label_map'].get(c,c)})"] = team_players_used[c].rank(ascending=False, method="dense")
        cols_show = ["Player","Minutes played","Age","Position"] + cfg["agg_cols"] + \
                    [f"Rank in team ({cfg['label_map'].get(c,c)})" for c in cfg["agg_cols"]]
        cols_show = [c for c in cols_show if c in team_players_used.columns]
        st.dataframe(team_players_used[cols_show].sort_values("Minutes played", ascending=False), use_container_width=True)
        _download_df_button(team_players_used[cols_show],
                            f"{cfg['title'].replace(' ','_')}_{target_team.replace(' ','_')}_players_used.csv",
                            "⬇️ Download players used (CSV)")

    # ---------- radar (bars only; NO title; FULL labels) ----------
    labels = [cfg["label_map"].get(m,m) for m in cfg["agg_cols"]]
    percentiles = [int(x) for x in rank_df["Percentile"].tolist()]
    fig = _polar_bars(labels, percentiles)
    st.pyplot(fig, use_container_width=True)
    buf=_io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor()); buf.seek(0)
    st.download_button("⬇️ Download radar", data=buf.getvalue(),
                       file_name=f"SectionB_{cfg['title'].replace(' ','_')}_{label_subject.replace(' ','_')}.png",
                       mime="image/png", key=f"dl_{role_key}_{label_subject}")
    plt.close(fig)

# ---------- Tabs in requested order ----------
tab_cb, tab_fb, tab_cm, tab_att, tab_st = st.tabs(
    ["Center Backs", "Fullbacks", "Central Midfielders", "Attackers", "Strikers"]
)
with tab_cb:  _sectionB_for_role("cb")
with tab_fb:  _sectionB_for_role("fb")
with tab_cm:  _sectionB_for_role("cm")
with tab_att: _sectionB_for_role("attack")
with tab_st:  _sectionB_for_role("cf")


# ======================== SECTION B (v4.6 — ONE CHART, EVEN SPLIT, MANUAL TEAM RANKS + AUTO ROLE) ========================
# ✅ Flow: Select League -> (minutes/age/search) -> Target Team (NO league team filtering required)
# ✅ Pool = ALL teams in league (post minutes/age/search + position filter)
# ✅ Team Style ranks use N teams from that same pool and N is displayed
# ✅ Single polar chart split TEAM(5) vs ROLE(5) unchanged

st.markdown("---")
st.header("Section B — League Comparison (Single Split Radar)")

import numpy as _np
import pandas as _pd
import io as _io
import re as _re
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------- helpers ----------
def _safe_div(num, den):
    num = _pd.to_numeric(num, errors="coerce")
    den = _pd.to_numeric(den, errors="coerce").replace(0, _np.nan)
    return (num / den).fillna(0.0)

def _pct_from_rank(rank: int, total: int) -> int:
    if total <= 1:
        return 100
    rank = max(1, min(int(rank), int(total)))
    return int(round((1 - (rank - 1) / (total - 1)) * 100))

def _keyify(s: str) -> str:
    s = str(s or "").strip().lower()
    s = _re.sub(r"[^a-z0-9_]+", "_", s)
    return s[:80] if s else "x"

# ---------- TEAM STYLE metric LABELS ONLY (manual ranks) ----------
TEAM_STYLE_LABELS = {
    "cb":     ["Possession", "Passes", "Direct Speed", "xGA", "Goals vs"],
    "fb":     ["Possession", "Passes", "Pressing", "Direct Speed", "xGA"],
    "cm":     ["Possession", "Passes", "Pressing", "Direct Speed", "Passes to Final 3rd"],
    "attack": ["Possession", "Passes", "Pressing", "Direct Speed", "xG"],
    "cf":     ["Possession", "Passes", "Pressing", "Long Balls", "xG"],
}

# ---------- upgraded single split polar chart ----------
def _single_split_polar(team_labels, team_pcts, role_labels, role_pcts):
    value_colors = ["#be2a3e", "#e25f48", "#f88f4d", "#f4d166", "#90b960", "#4b9b5f", "#22763f"]
    cmap = LinearSegmentedColormap.from_list("pct_scale", value_colors)

    TEAM_TRACK = "#2b3646"   # blue-grey
    ROLE_TRACK = "#362b46"   # purple-grey

    fig = plt.figure(figsize=(9.2, 8.2))
    fig.patch.set_facecolor("#0a0f1c")

    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88], polar=True)
    ax.set_facecolor("#0a0f1c")

    RMAX = 110
    ax.set_rlim(0, RMAX)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    n_team, n_role = 5, 5
    step_team = _np.pi / n_team
    step_role = _np.pi / n_role

    angles_team = (_np.pi/2) + (step_team/2) + _np.arange(n_team) * step_team
    angles_role = (-_np.pi/2) + (step_role/2) + _np.arange(n_role) * step_role

    angles = _np.concatenate([angles_team, angles_role])
    labels = list(team_labels) + list(role_labels)
    values = list(team_pcts) + list(role_pcts)

    width_team = step_team * 0.76
    width_role = step_role * 0.76
    widths = [width_team]*n_team + [width_role]*n_role

    theta_ring = _np.linspace(0, 2*_np.pi, 361)
    for r, alpha, lw in [(25, 0.08, 0.8), (50, 0.22, 1.8), (75, 0.08, 0.8)]:
        ax.plot(theta_ring, _np.full_like(theta_ring, r),
                color="white", alpha=alpha, linewidth=lw, zorder=0)

    for th in (_np.pi/2, 3*_np.pi/2):
        ax.plot([th, th], [0, RMAX],
                color="white", linewidth=4.8, alpha=0.65, zorder=0)

    for i, (th, w) in enumerate(zip(angles, widths)):
        is_team = i < n_team
        ax.bar(th, 100, width=w, bottom=0,
               color=(TEAM_TRACK if is_team else ROLE_TRACK),
               alpha=0.78, edgecolor="none", zorder=1)

    for th, w, v in zip(angles, widths, values):
        v = int(max(0, min(100, v)))
        ax.bar(th, v, width=w, bottom=0,
               color=cmap(v/100.0),
               edgecolor="white", linewidth=1.4, zorder=3)

    label_radius = 128
    for th, lab in zip(angles, labels):
        ax.text(th, label_radius, str(lab).upper(),
                ha="center", va="center",
                fontsize=9.6, fontweight="bold",
                color="white", alpha=0.95, zorder=4)

    corner_col = "#f472b6"
    fig.text(0.04, 0.965, "TEAM", ha="left", va="top",
             fontsize=18, fontweight="900", color=corner_col)
    fig.text(0.96, 0.965, "ROLE", ha="right", va="top",
             fontsize=18, fontweight="900", color=corner_col)

    return fig

# ---------- ROLE STYLE configs ----------
def _cfg(role_key: str):
    role_key = str(role_key).lower().strip()

    if role_key == "cb":
        require = [
            "Aerial duels per 90","Defensive duels per 90","Passes per 90","Forward passes per 90",
            "Progressive passes per 90","Progressive runs per 90","PAdj Interceptions","Shots blocked per 90"
        ]
        def pos_ok(s): s=str(s).upper().strip(); return s.startswith(("CB","RCB","LCB"))
        def compute(df):
            out = df.copy()
            out["Pass Verticality"] = _safe_div(out["Forward passes per 90"], out["Passes per 90"])
            out["Passing Volume"]   = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Defensive Volume"] = _pd.to_numeric(out["Defensive duels per 90"], errors="coerce")
            out["Progression Volume"] = (
                _pd.to_numeric(out["Progressive passes per 90"], errors="coerce")
                + _pd.to_numeric(out["Progressive runs per 90"], errors="coerce")
            )
            out["Aerial Volume"] = _pd.to_numeric(out["Aerial duels per 90"], errors="coerce")
            return out
        agg_cols = ["Defensive Volume","Aerial Volume","Passing Volume","Progression Volume","Pass Verticality"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Center Backs")

    if role_key == "fb":
        require = [
            "Passes per 90","Forward passes per 90","Progressive passes per 90","Progressive runs per 90",
            "Defensive duels per 90","PAdj Interceptions","Aerial duels per 90","xA per 90","Crosses per 90",
            "Touches in box per 90","Shots per 90","Passes to penalty area per 90","Accurate passes, %","Dribbles per 90"
        ]
        def pos_ok(s): s=str(s).upper().strip(); return s.startswith(("LB","LWB","RB","RWB"))
        def compute(df):
            out = df.copy()
            out["Progression Volume"] = (
                _pd.to_numeric(out["Progressive passes per 90"], errors="coerce")
                + _pd.to_numeric(out["Progressive runs per 90"], errors="coerce")
            )
            out["Attacking Contribution"] = (
                0.4 * _pd.to_numeric(out["xA per 90"], errors="coerce")
                + 0.2 * _pd.to_numeric(out["Crosses per 90"], errors="coerce")
                + 0.2 * _pd.to_numeric(out["Touches in box per 90"], errors="coerce")
                + 0.1 * _pd.to_numeric(out["Shots per 90"], errors="coerce")
                + 0.1 * _pd.to_numeric(out["Passes to penalty area per 90"], errors="coerce")
            )
            out["Defensive Volume"] = (
                0.5 * _pd.to_numeric(out["Defensive duels per 90"], errors="coerce")
                + 0.3 * _pd.to_numeric(out["PAdj Interceptions"], errors="coerce")
                + 0.2 * _pd.to_numeric(out["Aerial duels per 90"], errors="coerce")
            )
            out["Pass Volume"] = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Retention"] = _pd.to_numeric(out["Accurate passes, %"], errors="coerce")
            return out
        agg_cols = ["Defensive Volume","Pass Volume","Attacking Contribution","Retention","Progression Volume"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Fullbacks")

    if role_key == "cm":
        require = [
            "Passes per 90","Forward passes per 90","Progressive passes per 90","Progressive runs per 90",
            "Defensive duels per 90","PAdj Interceptions","Accurate passes, %"
        ]
        def pos_ok(s): s=str(s).upper().strip(); return s.startswith(("DMF","LDMF","RDMF","LCMF","RCMF","CMF"))
        def compute(df):
            out = df.copy()
            out["Pass Verticality"] = _safe_div(out["Forward passes per 90"], out["Passes per 90"])
            out["Pass Volume"] = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Progressive Volume"] = (
                _pd.to_numeric(out["Progressive passes per 90"], errors="coerce")
                + _pd.to_numeric(out["Progressive runs per 90"], errors="coerce")
            )
            out["Defensive Volume"] = _pd.to_numeric(out["Defensive duels per 90"], errors="coerce")
            out["Retention"] = _pd.to_numeric(out["Accurate passes, %"], errors="coerce")
            return out
        agg_cols = ["Defensive Volume","Pass Volume","Progressive Volume","Retention","Pass Verticality"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Central Midfielders")

    if role_key == "attack":
        require = [
            "Accurate passes, %","xG per 90","Non-penalty goals per 90","Touches in box per 90","xA per 90",
            "Passes to penalty area per 90","Passes per 90","Progressive passes per 90","Passes to final third per 90",
            "Dribbles per 90","Progressive runs per 90"
        ]
        def pos_ok(s: str) -> bool:
            s = str(s).upper().strip()
            main = _re.split(r"[/,]", s)[0].strip().split()[0]
            if main in ("RW", "LW"): return True
            return main.startswith(("RWF","LWF","LAMF","RAMF","AMF"))
        def compute(df):
            out = df.copy()
            out["Goal Threat"] = (
                0.4 * _pd.to_numeric(out["xG per 90"], errors="coerce")
                + 0.4 * _pd.to_numeric(out["Non-penalty goals per 90"], errors="coerce")
                + 0.2 * _pd.to_numeric(out["Touches in box per 90"], errors="coerce")
            )
            out["Creative Threat"] = (
                0.65 * _pd.to_numeric(out["xA per 90"], errors="coerce")
                + 0.35 * _pd.to_numeric(out["Passes to penalty area per 90"], errors="coerce")
            )
            out["Pass Volume"] = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Deep Playmaking"] = (
                0.5 * _pd.to_numeric(out["Progressive passes per 90"], errors="coerce")
                + 0.5 * _pd.to_numeric(out["Passes to final third per 90"], errors="coerce")
            )
            out["Ball Carrying"] = (
                0.6 * _pd.to_numeric(out["Dribbles per 90"], errors="coerce")
                + 0.4 * _pd.to_numeric(out["Progressive runs per 90"], errors="coerce")
            )
            return out
        agg_cols = ["Pass Volume","Deep Playmaking","Ball Carrying","Goal Threat","Creative Threat"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Attackers")

    if role_key == "cf":
        require = [
            "Touches in box per 90","xG per 90","Dribbles per 90","Progressive runs per 90",
            "Aerial duels per 90","Aerial duels won, %","Passes per 90","Accurate passes, %"
        ]
        def pos_ok(s): return str(s).upper().strip().startswith("CF")
        def compute(df):
            out = df.copy()
            out["Opportunities"] = (
                0.7 * _pd.to_numeric(out["Touches in box per 90"], errors="coerce")
                + 0.3 * _pd.to_numeric(out["xG per 90"], errors="coerce")
            )
            out["Carrying Outlet"] = (
                0.65 * _pd.to_numeric(out["Dribbles per 90"], errors="coerce")
                + 0.35 * _pd.to_numeric(out["Progressive runs per 90"], errors="coerce")
            )
            out["Aerial Volume"] = (
                _pd.to_numeric(out["Aerial duels per 90"], errors="coerce")
                * _pd.to_numeric(out["Aerial duels won, %"], errors="coerce") / 100.0
            )
            out["Pass Volume"] = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Retention"] = _pd.to_numeric(out["Accurate passes, %"], errors="coerce")
            return out
        agg_cols = ["Aerial Volume","Pass Volume","Opportunities","Retention","Carrying Outlet"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Strikers")

    return None

# ---------- per-role UI/logic ----------
def _sectionB_singlechart_manualteam_for_role(role_key: str, kbase: str = "secB_SINGLE_v46"):
    cfg = _cfg(role_key)
    if not cfg:
        st.info("Unknown role key.")
        return

    K = f"{kbase}_{_keyify(role_key)}"
    leagues = sorted([str(x) for x in df["League"].dropna().unique()])

    included_league = st.selectbox(f"League ({cfg['title']})", leagues, key=f"{K}_league")

    league_df = df[df["League"].astype(str) == included_league].copy()

    missing_role = [c for c in cfg["require_cols"] if c not in league_df.columns]
    if missing_role:
        st.info(f"Missing ROLE columns for {cfg['title']}: {', '.join(missing_role)}")
        return

    for c in ("Minutes played", "Age", "Goals"):
        if c in league_df.columns:
            league_df[c] = _pd.to_numeric(league_df[c], errors="coerce")

    if "Position" not in league_df.columns:
        st.info("Missing Position column.")
        return

    # position subset + required cols
    league_df = league_df[league_df["Position"].apply(cfg["pos_filter"])].dropna(subset=cfg["require_cols"])
    if league_df.empty:
        st.info(f"No {cfg['title']} in this league with required stats.")
        return

    # Filters (ONLY minutes/age/search; no team filtering)
    f1, f2 = st.columns([2.0, 1.4])
    with f1:
        q = st.text_input("Quick player search (optional)", "", key=f"{K}_q")
    with f2:
        min_minutes, max_minutes = st.slider("Minutes played", 0, 6000, (750, 6000), key=f"{K}_mins")
        a_min = int(_np.nanmin(league_df["Age"])) if league_df["Age"].notna().any() else 16
        a_max = int(_np.nanmax(league_df["Age"])) if league_df["Age"].notna().any() else 50
        min_age, max_age = st.slider("Age", a_min, a_max, (16, 50), key=f"{K}_age")

    pool = league_df[
        league_df["Minutes played"].between(min_minutes, max_minutes)
        & league_df["Age"].between(min_age, max_age)
    ].copy()

    if q.strip():
        s = q.strip().lower()
        pool = pool[pool["Player"].astype(str).str.lower().str.contains(s, na=False)]

    if pool.empty:
        st.info("No players after filters.")
        return

    pool = cfg["compute_metrics"](pool).copy()

    # ✅ Target team directly from league pool
    compare_mode = st.radio("Compare", ["Team average", "Specific player"], horizontal=True, key=f"{K}_mode")
    teams_pool = sorted(pool["Team"].dropna().astype(str).unique())
    if not teams_pool:
        st.info("No teams found in pool.")
        return
    target_team = st.selectbox("Target team", teams_pool, key=f"{K}_team")

    agg_role = pool.groupby("Team")[cfg["agg_cols"]].mean().reset_index()

    if compare_mode == "Team average":
        target_vals_role = agg_role.set_index("Team").loc[target_team, cfg["agg_cols"]].to_dict()
        label_subject = f"{target_team} AVG"
        exclude_label = target_team
    else:
        players_pool = sorted(pool[pool["Team"].astype(str) == target_team]["Player"].dropna().astype(str).unique())
        if not players_pool:
            st.info("No eligible players on selected team under current filters.")
            return
        sel_player = st.selectbox("Player", players_pool, key=f"{K}_player")
        prow = pool[pool["Player"].astype(str) == sel_player].head(1)
        if prow.empty:
            st.info("Pick a player above.")
            return
        target_vals_role = prow[cfg["agg_cols"]].iloc[0].to_dict()
        label_subject = sel_player
        exclude_label = str(prow["Team"].iloc[0])

    # ROLE percentiles
    rows_role = []
    for met in cfg["agg_cols"]:
        temp = agg_role[["Team", met]].copy()
        temp = temp[temp["Team"] != exclude_label].copy()
        val = float(target_vals_role.get(met, _np.nan))
        pseudo = _pd.DataFrame({"Team": [label_subject], met: [val]})
        temp = _pd.concat([temp, pseudo], ignore_index=True)
        temp = temp.drop_duplicates(subset="Team", keep="last")
        temp = temp.sort_values(by=met, ascending=False, kind="mergesort").reset_index(drop=True)
        rk = int(temp.index[temp["Team"] == label_subject][0]) + 1
        tot = int(temp.shape[0])
        pct = _pct_from_rank(rk, tot)
        rows_role.append((met, rk, tot, pct, val))

    rank_role_df = _pd.DataFrame(rows_role, columns=["Metric", "Rank", "Total teams", "Percentile", "Target value"])

    # ✅ N teams from role benchmark pool
    N_TEAMS = int(rank_role_df["Total teams"].max()) if not rank_role_df.empty else int(agg_role["Team"].nunique())
    N_TEAMS = max(1, N_TEAMS)

    # Manual TEAM ranks + show N
    st.markdown("### ✍️ Team Style — enter ranks (1 = best, N = worst)")
    st.caption(f"League benchmark size used for ranks: **N = {N_TEAMS} teams** (post minutes/age/search filters)")

    team_labels = TEAM_STYLE_LABELS.get(str(role_key).lower().strip(), ["Metric1","Metric2","Metric3","Metric4","Metric5"])[:5]
    cols = st.columns(5)
    team_ranks = []
    for col, lab in zip(cols, team_labels):
        with col:
            r = st.number_input(
                f"{lab} (rank 1–{N_TEAMS})",
                min_value=1, max_value=int(N_TEAMS),
                value=1,
                step=1,
                key=f"{K}_teamrank_{_keyify(lab)}"
            )
            team_ranks.append(int(r))
    team_pcts = [_pct_from_rank(r, N_TEAMS) for r in team_ranks]

    # Build chart
    role_labels = list(cfg["agg_cols"])
    role_pcts = [int(x) for x in rank_role_df["Percentile"].tolist()]

    fig = _single_split_polar(team_labels, team_pcts, role_labels, role_pcts)
    st.pyplot(fig, use_container_width=True)

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.download_button(
        "⬇️ Download radar",
        data=buf.getvalue(),
        file_name=f"SectionB_SingleSplitRadar_{cfg['title'].replace(' ','_')}_{_keyify(label_subject)}.png",
        mime="image/png",
        key=f"{K}_dl_radar_{_keyify(label_subject)}"
    )
    plt.close(fig)

# ---------- Tabs ----------
tab_cb, tab_fb, tab_cm, tab_att, tab_st = st.tabs(
    ["Center Backs", "Fullbacks", "Central Midfielders", "Attackers", "Strikers"]
)
with tab_cb:  _sectionB_singlechart_manualteam_for_role("cb", kbase="secB_SINGLE_v46")
with tab_fb:  _sectionB_singlechart_manualteam_for_role("fb", kbase="secB_SINGLE_v46")
with tab_cm:  _sectionB_singlechart_manualteam_for_role("cm", kbase="secB_SINGLE_v46")
with tab_att: _sectionB_singlechart_manualteam_for_role("attack", kbase="secB_SINGLE_v46")
with tab_st:  _sectionB_singlechart_manualteam_for_role("cf", kbase="secB_SINGLE_v46")
# ======================== END SECTION B (v4.6) ========================








# ============================== FEATURE R — SQUAD PROFILE ==============================
from io import BytesIO
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import patheffects as pe

st.markdown("---")
st.header("📊 Feature R — Squad Profile")

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
CONTRACT_COL = "Contract expires"   # <= 2026 -> default red highlight

# Optional smart label library
try:
    from adjustText import adjust_text
    HAVE_ADJUSTTEXT = True
except ImportError:
    HAVE_ADJUSTTEXT = False

# --------------------------------------------------------------------------------------
# SETTINGS PANEL
# --------------------------------------------------------------------------------------
with st.expander("Squad Profile settings", expanded=False):

    # --- Squad selection ---
    teams_available = sorted(df["Team"].dropna().unique())

    default_team = None
    selected_player_name = None
    player_row_obj = globals().get("player_row", pd.DataFrame())
    if isinstance(player_row_obj, pd.DataFrame) and not player_row_obj.empty:
        default_team = player_row_obj.iloc[0].get("Team", None)
        selected_player_name = player_row_obj.iloc[0].get("Player", None)

    default_idx = teams_available.index(default_team) if default_team in teams_available else 0

    squad_team = st.selectbox(
        "Squad (team)",
        options=teams_available,
        index=default_idx,
        key="sq_team",
    )

    # --- Axis filters ---
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Minutes 0–5000, default 0–4000
    min_minutes_s, max_minutes_s = st.slider(
        "Minutes range (for axis & filter)",
        0, 5000,
        (0, 4000),
        step=250,
        key="sq_min",
    )

    # Age 14–45, default 16–40
    min_age_s, max_age_s = st.slider(
        "Age range (for axis & filter)",
        14, 45,
        (16, 40),
        key="sq_age",
    )

    # --- Minutes bands (horizontal lines) ---
    st.markdown("**Minutes bands (horizontal dashed lines)**")
    important_line = st.slider(
        "Important Player line (minutes)",
        0, 5000, 500, step=250, key="sq_line_important",
    )
    crucial_line = st.slider(
        "Crucial Player line (minutes)",
        0, 5000, 1000, step=250, key="sq_line_crucial",
    )

    band_lines = sorted(
        [("Important Player", important_line),
         ("Crucial Player", crucial_line)],
        key=lambda x: x[1],
    )

    # --- Contract highlight & custom red players ---
    auto_contract_red = st.checkbox(
        "Highlight players with contract ≤ 2026 in red",
        value=True,
        key="sq_auto_contract",
    )

    team_players_all = sorted(
        df[df["Team"] == squad_team]["Player"].dropna().unique().tolist()
    )
    custom_red_players = st.multiselect(
        "Force-highlight specific players in red",
        options=team_players_all,
        default=[],
        key="sq_custom_red",
    )

    # --- Labels & points ---
    show_labels = st.toggle("Show labels", value=True, key="sq_show_labels")
    label_size = st.slider("Label size", 8, 22, 15, 1, key="sq_lblsize")  # default 15
    point_size = st.slider("Point size", 24, 300, 300, 2, key="sq_pts")   # default 300
    point_alpha = st.slider("Point opacity", 0.2, 1.0, 0.92, 0.02, key="sq_alpha")

    # --- Theme & canvas (same style as Feature Q) ---
    PAGE_BG = "#0a0f1c"
    PLOT_BG = "#0a0f1c"
    GRID_MAJ = "#3a4050"
    txt_col = "#f1f5f9"

    canvas_preset = st.selectbox(
        "Canvas size",
        ["1280×720", "1600×900", "1920×820", "1920×1080"],
        index=1,
        key="sq_canvas",
    )
    w_px, h_px = map(int, canvas_preset.replace("×", "x").split("x"))

    top_gap_px = st.slider("Top gap (px)", 0, 240, 80, 5, key="sq_gap")
    render_exact = st.checkbox("Render exact pixels (PNG)", value=True, key="sq_exact")

# --------------------------------------------------------------------------------------
# FILTER SQUAD
# --------------------------------------------------------------------------------------
squad = df[df["Team"] == squad_team].copy()
if squad.empty:
    st.info("No players found for this squad.")
    st.stop()

squad["Minutes played"] = pd.to_numeric(squad["Minutes played"], errors="coerce")
squad["Age"] = pd.to_numeric(squad["Age"], errors="coerce")

squad = squad[
    squad["Minutes played"].between(min_minutes_s, max_minutes_s)
    & squad["Age"].between(min_age_s, max_age_s)
]

if squad.empty:
    st.info("No players after applying filters.")
    st.stop()

# --------------------------------------------------------------------------------------
# CONTRACT HIGHLIGHT & CUSTOM HIGHLIGHT
# --------------------------------------------------------------------------------------
if auto_contract_red and CONTRACT_COL in squad.columns:
    contract_year = (
        squad[CONTRACT_COL]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
        .astype(float)
    )
    squad["ContractYear"] = contract_year
    squad["AutoRed"] = squad["ContractYear"].le(2026)
else:
    squad["ContractYear"] = np.nan
    squad["AutoRed"] = False

squad["Selected"] = False
if selected_player_name:
    squad["Selected"] = squad["Player"] == selected_player_name

squad["CustomRed"] = squad["Player"].isin(custom_red_players)
squad["IsRed"] = squad["AutoRed"] | squad["Selected"] | squad["CustomRed"]

# --------------------------------------------------------------------------------------
# SCATTER BASE
# --------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
fig.patch.set_facecolor(PAGE_BG)
ax.set_facecolor(PLOT_BG)

ax.set_xlim(min_age_s, max_age_s)
ax.set_ylim(min_minutes_s, max_minutes_s)

ax.set_xlabel("Age", fontsize=16, fontweight="semibold", color=txt_col)
ax.xaxis.labelpad = 14
ax.set_ylabel("Minutes Played", fontsize=16, fontweight="semibold", color=txt_col)

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(250))

for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("semibold")
    tick.set_color(txt_col)
    tick.set_fontsize(14)

ax.grid(True, color=GRID_MAJ, linewidth=0.6)
for s in ax.spines.values():
    s.set_color("#e5e7eb")
    s.set_linewidth(1.1)

# --------------------------------------------------------------------------------------
# AGE BANDS (flexible titles, ASCENT 21–24)
# --------------------------------------------------------------------------------------
line_col = "#FFFFFF"

# Fixed conceptual bands: 16–20, 21–24, 25–28, 29–32, 33–45
AGE_BAND_LABELS = ["YOUTH", "ASCENT", "PRIME", "EXPERIENCED", "OLD"]
AGE_BAND_EDGES = [16, 21, 25, 29, 33, 45]

# Vertical lines at 21, 25, 29, 33
for al in [21, 25, 29, 33]:
    if min_age_s <= al <= max_age_s:
        ax.axvline(al, color=line_col, linestyle=(0, (4, 4)), lw=1.5)

# Titles centred within the *visible* part of each band
for i, label in enumerate(AGE_BAND_LABELS):
    band_start = AGE_BAND_EDGES[i]
    band_end = AGE_BAND_EDGES[i + 1]

    visible_start = max(band_start, min_age_s)
    visible_end = min(band_end, max_age_s)

    if visible_start >= visible_end or max_age_s == min_age_s:
        continue

    center = (visible_start + visible_end) / 2.0
    x_frac = (center - min_age_s) / float(max_age_s - min_age_s)

    ax.text(
        x_frac,
        1.01,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        color=txt_col,
        ha="center",
        va="bottom",
    )

# --------------------------------------------------------------------------------------
# MINUTES BANDS
# --------------------------------------------------------------------------------------
for name, y_val in band_lines:
    if min_minutes_s <= y_val <= max_minutes_s:
        ax.axhline(y_val, color=line_col, linestyle=(0, (4, 4)), lw=1.5)
        ax.text(
            min_age_s + 0.2,
            y_val + (max_minutes_s - min_minutes_s) * 0.01,
            name,
            fontsize=14,
            fontweight="bold",
            color="#020617",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#e5e7eb",
                edgecolor="none",
                alpha=0.95,
            ),
            va="bottom",
        )

# --------------------------------------------------------------------------------------
# POINTS
# --------------------------------------------------------------------------------------
effective_point_size = point_size * 1.1
for is_red, grp in squad.groupby("IsRed"):
    ax.scatter(
        grp["Age"],
        grp["Minutes played"],
        s=effective_point_size,
        c="#ef4444" if is_red else "#e5e7eb",
        alpha=point_alpha,
        edgecolors="none",
        linewidth=0,
        zorder=3 if is_red else 2,
    )

# --------------------------------------------------------------------------------------
# LABELS – ALL PLAYERS
# --------------------------------------------------------------------------------------
if show_labels:
    label_df = squad.copy()

    axis_height = max_minutes_s - min_minutes_s
    top_margin = axis_height * 0.04
    bottom_margin = axis_height * 0.03

    # --------------------- PATH 1: adjustText available -------------------------------
    if HAVE_ADJUSTTEXT:
        texts = []
        xs = label_df["Age"].values
        ys = label_df["Minutes played"].values

        for x, y, name, is_red in zip(xs, ys, label_df["Player"], label_df["IsRed"]):
            t = ax.text(
                x,
                y,
                name,
                fontsize=label_size,
                color=txt_col,
                weight="semibold",
                ha="center",
                va="bottom",
                zorder=6 if is_red else 5,
            )
            t.set_path_effects([
                pe.withStroke(linewidth=2, foreground="#020617", alpha=0.9)
            ])
            texts.append(t)

        adjust_text(
            texts,
            x=xs,
            y=ys,
            ax=ax,
            autoalign="y",
            only_move={"points": "y", "text": "xy"},
            force_points=0.7,
            force_text=0.7,
            expand_points=(1.1, 1.5),
            expand_text=(1.1, 1.5),
            arrowprops=dict(
                arrowstyle="-",
                lw=0.6,
                color=txt_col,
                alpha=0.6,
            ),
        )

        # Clamp labels back inside vertical bounds (below band titles)
        for t in texts:
            x_lab, y_lab = t.get_position()
            y_lab = max(min_minutes_s + bottom_margin, min(y_lab, max_minutes_s - top_margin))
            t.set_position((x_lab, y_lab))

    # --------------------- PATH 2: manual fallback ------------------------------------
    else:
        axis_height = max_minutes_s - min_minutes_s
        base_offset = axis_height * 0.015      # starting gap above dot
        min_y_delta = axis_height * 0.05       # min vertical separation
        age_tol = 0.7                          # "same column" width in age units
        x_jitter = 0.25                        # horizontal nudge when stacked

        label_df_sorted = label_df.sort_values("Minutes played")
        placed = []
        positions = {}

        for _, r in label_df_sorted.iterrows():
            x = float(r["Age"])
            y = float(r["Minutes played"])

            x_lab = x
            y_lab = y + base_offset
            y_lab = max(min_minutes_s + bottom_margin, min(y_lab, max_minutes_s - top_margin))

            direction_y = 1
            direction_x = 1
            attempts = 0
            max_attempts = 80

            while attempts < max_attempts:
                collision = False
                for (px, py) in placed:
                    if abs(x_lab - px) < age_tol and abs(y_lab - py) < min_y_delta:
                        collision = True
                        break

                if not collision:
                    break

                y_lab += direction_y * min_y_delta
                x_lab += direction_x * x_jitter

                direction_y *= -1
                direction_x *= -1

                y_lab = max(min_minutes_s + bottom_margin, min(y_lab, max_minutes_s - top_margin))
                x_lab = max(min_age_s + 0.2, min(x_lab, max_age_s - 0.2))

                attempts += 1

            placed.append((x_lab, y_lab))
            positions[r["Player"]] = (x_lab, y_lab)

        # Draw labels + leader lines
        for _, r in label_df.iterrows():
            x = float(r["Age"])
            y = float(r["Minutes played"])
            x_lab, y_lab = positions.get(r["Player"], (x, y + base_offset))

            if abs(x_lab - x) > 0.05 or abs(y_lab - (y + base_offset)) > 0.05:
                ax.plot(
                    [x, x_lab],
                    [y, y_lab],
                    linestyle="-",
                    linewidth=0.5,
                    color=txt_col,
                    alpha=0.5,
                    zorder=5,
                )

            z = 6 if r["IsRed"] else 5
            t = ax.annotate(
                r["Player"],
                xy=(x_lab, y_lab),
                textcoords="data",
                fontsize=label_size,
                color=txt_col,
                weight="semibold",
                ha="center",
                va="bottom",
                zorder=z,
            )
            t.set_path_effects([
                pe.withStroke(linewidth=2, foreground="#020617", alpha=0.9)
            ])

# --------------------------------------------------------------------------------------
# LAYOUT & RENDER
# --------------------------------------------------------------------------------------
fig.subplots_adjust(
    left=0.06,
    right=0.98,
    bottom=0.11,
    top=1.02 - top_gap_px / float(h_px),
)

if render_exact:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=PAGE_BG)
    buf.seek(0)
    st.image(buf, width=w_px)
    st.download_button(
        "⬇️ Download Squad Profile (PNG)",
        data=buf.getvalue(),
        file_name=f"squad_profile_{uuid.uuid4().hex[:6]}.png",
        mime="image/png",
    )
else:
    st.pyplot(fig)

plt.close(fig)
# ============================== END FEATURE R ==========================================


































































