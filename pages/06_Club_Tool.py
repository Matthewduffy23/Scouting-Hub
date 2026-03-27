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
    'USA 1.','USA 2.','Uruguay 1.','Uzbekistan 1.','Venezuela 1.','Wales 1.' 'Faroe Islands 1.'
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
'Estonia 2.':3, 'Ireland 2.':10, 'Faroe Islands 1.':35.02,
}

# ========================= GBE BANDS + REGIONS (PRESETS) =========================
GBE_LEAGUE_BANDS = {
    "England 1.": 1, "England 2.": 1, "England 3.": 1, "England 4.": 1,
    "England 5.": 1, "England 6.": 1, "England 7.": 1, "England 8.": 1,
    "England 9.": 1, "England 10.": 1,
    "Scotland 1.": 1, "Scotland 2.": 1, "Scotland 3.": 1,
    "Wales 1.": 1,
    "Ireland 1.": 1,
    "Northern Ireland 1.": 1,
    "Spain 1.": 1, "Germany 1.": 1, "Italy 1.": 1, "France 1.": 1,
    "Portugal 1.": 2, "Netherlands 1.": 2, "Belgium 1.": 2, "Turkey 1.": 2,
    "USA 1.": 3, "Brazil 1.": 3, "Argentina 1.": 3, "Mexico 1.": 3,
    "Czech 1.": 4, "Croatia 1.": 4, "Switzerland 1.": 4,
    "Spain 2.": 4, "Germany 2.": 4,
    "Ukraine 1.": 4, "Greece 1.": 4, "Colombia 1.": 4,
    "Austria 1.": 4, "Denmark 1.": 4, "France 2.": 4, "Russia 1.": 4,
    "Serbia 1.": 5, "Poland 1.": 5, "Slovenia 1.": 5, "Chile 1.": 5, "Uruguay 1.": 5,
    "Sweden 1.": 5, "Norway 1.": 5, "Italy 2.": 5, "Hungary 1.": 5, "Japan 1.": 5,
    "Korea 1.": 5, "Australia 1.": 5,
}

def gbe_league_band(league_name: str) -> int:
    league_name = str(league_name).strip()
    return int(GBE_LEAGUE_BANDS.get(league_name, 6))

@st.cache_data(show_spinner=False)
def league_strength_band_df(max_band: int = 6) -> pd.DataFrame:
    rows = []
    for league, strength in LEAGUE_STRENGTHS.items():
        band = gbe_league_band(league)
        if band <= max_band:
            rows.append({"League": league, "League strength": strength, "GBE band": band})
    df_ls = pd.DataFrame(rows)
    return df_ls.sort_values(["League strength"], ascending=[False])

COUNTRY_TO_REGION = {
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
    "Brazil": "South America", "Argentina": "South America", "Colombia": "South America",
    "Ecuador": "South America", "Paraguay": "South America", "Uruguay": "South America",
    "Chile": "South America", "Bolivia": "South America", "Peru": "South America",
    "Venezuela": "South America",
    "USA": "North America", "Mexico": "North America", "Costa Rica": "North America",
    "Canada": "North America",
    "Morocco": "Africa", "Algeria": "Africa", "Egypt": "Africa", "Nigeria": "Africa",
    "Tunisia": "Africa", "South Africa": "Africa",
    "Japan": "Asia", "Korea": "Asia", "Saudi": "Asia",
    "UAE": "Asia", "Qatar": "Asia", "Uzbekistan": "Asia", "Israel": "Asia",
    "Turkey": "Asia", "Azerbaijan": "Asia",
    "Australia": "Asia",
}

def league_country(league: str) -> str:
    s = str(league).strip()
    m = re.match(r"^(.*)\s\d+\.\s*$", s)
    return m.group(1).strip() if m else s

def league_region(league: str) -> str:
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

seed = set()
if use_top5:
    seed |= PRESET_LEAGUES["Top 5 Europe"]
if use_top20:
    seed |= PRESET_LEAGUES["Top 20 Europe"]
if use_efl:
    seed |= PRESET_LEAGUES["EFL (England 2–4)"]
if region_picks:
    seed |= {lg for lg in INCLUDED_LEAGUES if league_region(lg) in set(region_picks)}
if band_picks:
    seed |= {lg for lg in INCLUDED_LEAGUES if gbe_league_band(lg) in set(band_picks)}
if band_max != "— None —":
    seed |= {lg for lg in INCLUDED_LEAGUES if gbe_league_band(lg) <= int(band_max)}

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
        pool_min_value, pool_max_value = st.slider("MV range (€)", 0, mv_cap, (0, min(mv_cap, 400_000_000)), step=500_000)

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
def format_market_value_gbp(v) -> str:
    if v is None:
        return "—"
    v = pd.to_numeric(v, errors="coerce")
    if not np.isfinite(v):
        return "—"
    sign = "-" if v < 0 else ""
    v = abs(float(v))
    if v >= 1_000_000:
        s = f"{v/1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{sign}£{s}m"
    if v >= 1_000:
        s = f"{v/1_000:.0f}"
        return f"{sign}£{s}k"
    return f"{sign}£{int(v):,}"

def build_base_pool():
    p = df.copy()
    p = p[p["League"].isin(leagues_sel)]
    for c in ["Minutes played", "Age", "Market value", "Goals"]:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")
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

def _percentile_of_value(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.5
    return float((s <= float(value)).mean())

def _percentile_vector_for_league(ref_df: pd.DataFrame, metric_cols: List[str], values: dict) -> dict:
    out = {}
    for m in metric_cols:
        s = pd.to_numeric(ref_df.get(m), errors="coerce").dropna()
        v = float(values.get(m, np.nan))
        if s.empty or not np.isfinite(v):
            out[m] = 50.0
        else:
            out[m] = _percentile_of_value(s, v) * 100.0
    return out

def _add_candidate_percentiles_per_league(pool: pd.DataFrame, ref_all: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    out = pool.copy()
    for m in metric_cols:
        out[f"{m} %ile"] = 50.0
    if out.empty:
        return out
    leagues = sorted(out["League"].dropna().astype(str).unique())
    for lg in leagues:
        idx = out["League"].astype(str) == lg
        ref_lg = ref_all[ref_all["League"].astype(str) == lg]
        if ref_lg.empty:
            continue
        for m in metric_cols:
            s = pd.to_numeric(ref_lg.get(m), errors="coerce").dropna()
            if s.empty:
                continue
            out.loc[idx, f"{m} %ile"] = pd.to_numeric(out.loc[idx, m], errors="coerce").map(
                lambda v: _percentile_of_value(s, v) * 100.0
            )
    return out

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
    if tmpl_src.empty or pool.empty:
        return pd.DataFrame()

    tmpl_numeric = tmpl_src.copy()
    for f in sim_features:
        tmpl_numeric[f] = pd.to_numeric(tmpl_numeric.get(f), errors="coerce")
    tmpl_numeric = tmpl_numeric.dropna(subset=sim_features)
    if tmpl_numeric.empty:
        return pd.DataFrame()

    target_vals = tmpl_numeric[sim_features].mean().astype(float).values

    cand = pool.copy()
    for f in sim_features:
        cand[f] = pd.to_numeric(cand.get(f), errors="coerce")
    cand = cand.dropna(subset=sim_features)

    if cand.empty:
        return pd.DataFrame()

    cand["Minutes played"] = pd.to_numeric(cand.get("Minutes played"), errors="coerce")
    cand["League strength"] = cand["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
    cand = (
        cand.sort_values(["Player", "Minutes played", "League strength"], ascending=[True, False, False])
            .drop_duplicates(subset=["Player"], keep="first")
    )

    weights_vec = np.array([float(weights_dict.get(f, 1.0)) for f in sim_features], dtype=float)

    league_block = df.loc[df["League"].astype(str) == str(target_league), sim_features].copy()
    for f in sim_features:
        league_block[f] = pd.to_numeric(league_block.get(f), errors="coerce")
    target_pct = []
    for i, f in enumerate(sim_features):
        target_pct.append(_percentile_of_value(league_block[f], float(target_vals[i])))
    target_pct = np.asarray(target_pct, dtype=float)

    percl = cand.groupby("League")[sim_features].rank(pct=True).values

    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(cand[sim_features])
    target_features_standardized = scaler.transform([target_vals])

    percentile_distances = np.linalg.norm((percl - target_pct) * weights_vec, axis=1)
    actual_value_distances = np.linalg.norm((standardized_features - target_features_standardized) * weights_vec, axis=1)
    combined = percentile_distances * float(percentile_weight) + actual_value_distances * (1.0 - float(percentile_weight))

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

    tmpl_src = _template_rows_for_role(lambda p: p.strip().upper().startswith("CF")).dropna(subset=feats)

    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (ST)", ["— Select —"] + players, index=0, key="st_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]

    if tmpl_src.empty:
        st.error("No strikers found for template conditions.")
        return pd.DataFrame(), "Strikers (CF)", pd.DataFrame()

    f = tmpl_src.copy()
    f["Opportunities"]     = 0.7*f['Touches in box per 90'] + 0.3*f['xG per 90']
    f["Ball Carrying"]     = 0.65*f['Dribbles per 90'] + 0.35*f['Progressive runs per 90']
    f["Aerial Requirement"]= f['Aerial duels per 90'] * f['Aerial duels won, %'] / 100.0
    f["Passing Volume"]    = f['Passes per 90']
    f["Goal Output"]       = f['Non-penalty goals per 90']
    f["Retention"]         = f['Accurate passes, %']
    tmpl_vec = f[["Opportunities","Ball Carrying","Aerial Requirement","Passing Volume","Goal Output","Retention"]].mean()

    base_pool = build_base_pool()
    pool = base_pool.copy()
    pool = pool[pool["Position"].str.upper().str.startswith("CF")]
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))].copy()

    for c in feats:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Opportunities"]      = 0.7*pool['Touches in box per 90'] + 0.3*pool['xG per 90']
    pool["Ball Carrying"]      = 0.65*pool['Dribbles per 90'] + 0.35*pool['Progressive runs per 90']
    pool["Aerial Requirement"] = pool['Aerial duels per 90'] * pool['Aerial duels won, %'] / 100.0
    pool["Passing Volume"]     = pool['Passes per 90']
    pool["Goal Output"]        = pool['Non-penalty goals per 90']
    pool["Retention"]          = pool['Accurate passes, %']

    cols = ["Opportunities","Ball Carrying","Aerial Requirement","Passing Volume","Goal Output","Retention"]

    ref_all = build_base_pool().copy()
    ref_all = ref_all[ref_all["Position"].str.upper().str.startswith("CF")].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all[c], errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    ref_all["Opportunities"]      = 0.7*ref_all['Touches in box per 90'] + 0.3*ref_all['xG per 90']
    ref_all["Ball Carrying"]      = 0.65*ref_all['Dribbles per 90'] + 0.35*ref_all['Progressive runs per 90']
    ref_all["Aerial Requirement"] = ref_all['Aerial duels per 90'] * ref_all['Aerial duels won, %'] / 100.0
    ref_all["Passing Volume"]     = ref_all['Passes per 90']
    ref_all["Goal Output"]        = ref_all['Non-penalty goals per 90']
    ref_all["Retention"]          = ref_all['Accurate passes, %']

    ref_tmpl = ref_all[ref_all["League"].astype(str) == str(template_league)].copy()
    tmpl_pct = _percentile_vector_for_league(ref_tmpl, cols, tmpl_vec.to_dict())

    pool = _add_candidate_percentiles_per_league(pool, ref_all, cols)

    for c in cols:
        pool[f"__tmpl__{c}"] = tmpl_pct[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[f"{c} %ile"]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

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
        return pd.DataFrame(), f"Attackers ({role_choice})", pd.DataFrame()

    f = tmpl_src.copy()
    f["Retention Style"]    = f['Accurate passes, %']
    f["Goal Threat"]        = 0.4*f['xG per 90'] + 0.4*f['Non-penalty goals per 90'] + 0.2*f['Touches in box per 90']
    f["Creativity Threat"]  = 0.65*f['xA per 90'] + 0.35*f['Passes to penalty area per 90']
    f["Passing Volume"]     = f['Passes per 90']
    f["Deeper Playmaking"]  = 0.5*f['Progressive passes per 90'] + 0.5*f['Passes to final third per 90']
    f["Ball Carrying"]      = 0.6*f['Dribbles per 90'] + 0.4*f['Progressive runs per 90']
    cols = ["Retention Style","Goal Threat","Creativity Threat","Passing Volume","Deeper Playmaking","Ball Carrying"]
    tmpl_vec = f[cols].mean()

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]

    for c in feats:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Retention Style"]   = pool['Accurate passes, %']
    pool["Goal Threat"]       = 0.4*pool['xG per 90'] + 0.4*pool['Non-penalty goals per 90'] + 0.2*pool['Touches in box per 90']
    pool["Creativity Threat"] = 0.65*pool['xA per 90'] + 0.35*pool['Passes to penalty area per 90']
    pool["Passing Volume"]    = pool['Passes per 90']
    pool["Deeper Playmaking"] = 0.5*pool['Progressive passes per 90'] + 0.5*pool['Passes to final third per 90']
    pool["Ball Carrying"]     = 0.6*pool['Dribbles per 90'] + 0.4*pool['Progressive runs per 90']

    ref_all = build_base_pool().copy()
    ref_all = ref_all[ref_all["Position"].apply(pos_ok)].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all[c], errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    ref_all["Retention Style"]   = ref_all['Accurate passes, %']
    ref_all["Goal Threat"]       = 0.4*ref_all['xG per 90'] + 0.4*ref_all['Non-penalty goals per 90'] + 0.2*ref_all['Touches in box per 90']
    ref_all["Creativity Threat"] = 0.65*ref_all['xA per 90'] + 0.35*ref_all['Passes to penalty area per 90']
    ref_all["Passing Volume"]    = ref_all['Passes per 90']
    ref_all["Deeper Playmaking"] = 0.5*ref_all['Progressive passes per 90'] + 0.5*ref_all['Passes to final third per 90']
    ref_all["Ball Carrying"]     = 0.6*ref_all['Dribbles per 90'] + 0.4*ref_all['Progressive runs per 90']

    ref_tmpl = ref_all[ref_all["League"].astype(str) == str(template_league)].copy()
    tmpl_pct = _percentile_vector_for_league(ref_tmpl, cols, tmpl_vec.to_dict())

    pool = _add_candidate_percentiles_per_league(pool, ref_all, cols)

    for c in cols:
        pool[f"__tmpl__{c}"] = tmpl_pct[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[f"{c} %ile"]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

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
        return pd.DataFrame(), "Central Midfield", pd.DataFrame()

    f = tmpl_src.copy()
    f["Pass Verticality"]    = _safe_verticality(f['Forward passes per 90'], f['Passes per 90'])
    f["Progression Volume"]  = f['Progressive passes per 90'] + f['Progressive runs per 90']
    f["Attacking Contribution"] = f['Touches in box per 90'] + f['Shots per 90']
    f["Defensive Volume"]    = f['Defensive duels per 90']
    f["Interception Volume"] = f['PAdj Interceptions']
    f["Retention"]           = f['Accurate passes, %']

    cols = ["Passes per 90","Pass Verticality","Progression Volume","Defensive Volume","Interception Volume","Attacking Contribution","Retention"]
    tmpl_vec = f[cols].mean()

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]

    for c in feats:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Pass Verticality"]    = _safe_verticality(pool['Forward passes per 90'], pool['Passes per 90'])
    pool["Progression Volume"]  = pool['Progressive passes per 90'] + pool['Progressive runs per 90']
    pool["Attacking Contribution"] = pool['Touches in box per 90'] + pool['Shots per 90']
    pool["Defensive Volume"]    = pool['Defensive duels per 90']
    pool["Interception Volume"] = pool['PAdj Interceptions']
    pool["Retention"]           = pool['Accurate passes, %']

    ref_all = build_base_pool().copy()
    ref_all = ref_all[ref_all["Position"].apply(pos_ok)].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all[c], errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    ref_all["Pass Verticality"]    = _safe_verticality(ref_all['Forward passes per 90'], ref_all['Passes per 90'])
    ref_all["Progression Volume"]  = ref_all['Progressive passes per 90'] + ref_all['Progressive runs per 90']
    ref_all["Attacking Contribution"] = ref_all['Touches in box per 90'] + ref_all['Shots per 90']
    ref_all["Defensive Volume"]    = ref_all['Defensive duels per 90']
    ref_all["Interception Volume"] = ref_all['PAdj Interceptions']
    ref_all["Retention"]           = ref_all['Accurate passes, %']

    ref_tmpl = ref_all[ref_all["League"].astype(str) == str(template_league)].copy()
    tmpl_pct = _percentile_vector_for_league(ref_tmpl, cols, tmpl_vec.to_dict())

    pool = _add_candidate_percentiles_per_league(pool, ref_all, cols)

    for c in cols:
        pool[f"__tmpl__{c}"] = tmpl_pct[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[f"{c} %ile"] - r[f"__tmpl__{c}"] for c in cols]), axis=1)

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
        return pd.DataFrame(), f"Fullbacks ({role_choice})", pd.DataFrame()

    f = tmpl_src.copy()
    f["Pass Verticality"]    = _safe_verticality(f['Forward passes per 90'], f['Passes per 90'])
    f["Progression Volume"]  = f['Progressive passes per 90'] + f['Progressive runs per 90']
    f["Attacking Contribution"]= 0.4*f['xA per 90'] + 0.2*f['Crosses per 90'] + 0.2*f['Touches in box per 90'] + 0.1*f['Shots per 90'] + 0.1*f['Passes to penalty area per 90']
    f["Defensive Volume"]    = 0.5*f['Defensive duels per 90'] + 0.3*f['PAdj Interceptions'] + 0.2*f['Aerial duels per 90']
    f["Retention"]           = f['Accurate passes, %']

    cols = ["Passes per 90","Pass Verticality","Progression Volume","Attacking Contribution","Defensive Volume","Retention"]
    tmpl_vec = f[cols].mean()

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]

    for c in feats:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Pass Verticality"]     = _safe_verticality(pool['Forward passes per 90'], pool['Passes per 90'])
    pool["Progression Volume"]   = pool['Progressive passes per 90'] + pool['Progressive runs per 90']
    pool["Attacking Contribution"]= 0.4*pool['xA per 90'] + 0.2*pool['Crosses per 90'] + 0.2*pool['Touches in box per 90'] + 0.1*pool['Shots per 90'] + 0.1*pool['Passes to penalty area per 90']
    pool["Defensive Volume"]     = 0.5*pool['Defensive duels per 90'] + 0.3*pool['PAdj Interceptions'] + 0.2*pool['Aerial duels per 90']
    pool["Retention"]            = pool['Accurate passes, %']

    ref_all = build_base_pool().copy()
    ref_all = ref_all[ref_all["Position"].apply(pos_ok)].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all[c], errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    ref_all["Pass Verticality"]     = _safe_verticality(ref_all['Forward passes per 90'], ref_all['Passes per 90'])
    ref_all["Progression Volume"]   = ref_all['Progressive passes per 90'] + ref_all['Progressive runs per 90']
    ref_all["Attacking Contribution"]= 0.4*ref_all['xA per 90'] + 0.2*ref_all['Crosses per 90'] + 0.2*ref_all['Touches in box per 90'] + 0.1*ref_all['Shots per 90'] + 0.1*ref_all['Passes to penalty area per 90']
    ref_all["Defensive Volume"]     = 0.5*ref_all['Defensive duels per 90'] + 0.3*ref_all['PAdj Interceptions'] + 0.2*ref_all['Aerial duels per 90']
    ref_all["Retention"]            = ref_all['Accurate passes, %']

    ref_tmpl = ref_all[ref_all["League"].astype(str) == str(template_league)].copy()
    tmpl_pct = _percentile_vector_for_league(ref_tmpl, cols, tmpl_vec.to_dict())

    pool = _add_candidate_percentiles_per_league(pool, ref_all, cols)

    for c in cols:
        pool[f"__tmpl__{c}"] = tmpl_pct[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[f"{c} %ile"] - r[f"__tmpl__{c}"] for c in cols]), axis=1)

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
        return pd.DataFrame(), "Center Backs", pd.DataFrame()

    f = tmpl_src.copy()
    f["Passing Verticality"] = _safe_verticality(f['Forward passes per 90'], f['Passes per 90'])
    f["Passing Volume"]      = f['Passes per 90']
    f["Positional Demand"]   = f['PAdj Interceptions'] + f['Shots blocked per 90']
    f["Progression Volume"]  = f['Progressive passes per 90'] + f['Progressive runs per 90']

    cols = ["Aerial duels per 90","Defensive duels per 90","Positional Demand","Passing Volume","Passing Verticality","Progression Volume"]
    tmpl_vec = f[cols].mean()

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]

    for c in feats:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Passing Verticality"] = _safe_verticality(pool['Forward passes per 90'], pool['Passes per 90'])
    pool["Passing Volume"]      = pool['Passes per 90']
    pool["Positional Demand"]   = pool['PAdj Interceptions'] + pool['Shots blocked per 90']
    pool["Progression Volume"]  = pool['Progressive passes per 90'] + pool['Progressive runs per 90']

    ref_all = build_base_pool().copy()
    ref_all = ref_all[ref_all["Position"].apply(pos_ok)].copy()
    for c in feats:
        ref_all[c] = pd.to_numeric(ref_all[c], errors="coerce")
    ref_all = ref_all.dropna(subset=feats)

    ref_all["Passing Verticality"] = _safe_verticality(ref_all['Forward passes per 90'], ref_all['Passes per 90'])
    ref_all["Passing Volume"]      = ref_all['Passes per 90']
    ref_all["Positional Demand"]   = ref_all['PAdj Interceptions'] + ref_all['Shots blocked per 90']
    ref_all["Progression Volume"]  = ref_all['Progressive passes per 90'] + ref_all['Progressive runs per 90']

    ref_tmpl = ref_all[ref_all["League"].astype(str) == str(template_league)].copy()
    tmpl_pct = _percentile_vector_for_league(ref_tmpl, cols, tmpl_vec.to_dict())

    pool = _add_candidate_percentiles_per_league(pool, ref_all, cols)

    for c in cols:
        pool[f"__tmpl__{c}"] = tmpl_pct[c]
    pool["BaseDist"] = pool.apply(
        lambda r: norm([r[f"{c} %ile"] - r[f"__tmpl__{c}"] for c in cols]),
        axis=1
    )

    ranked = _score_block(pool.copy())
    return ranked, "Center Backs", tmpl_src


# ========================= FOTMOB PHOTO + CREST =========================
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
    key_id = f"{player}|||{team}|||{league}"
    override = st.session_state.get("photo_map", {}).get(key_id, "")
    if override:
        return override
    try:
        from photo_utils import get_player_photo_url
        return get_player_photo_url(player, team)
    except Exception:
        return PLACEHOLDER_IMG
    squad = _fotmob_team_squad(tid)
    target_surname = _slug_name(_player_surname(player))
    target_full = _slug_name(player)
    best_id = ""
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
    if not best_id and target_full:
        for m in squad:
            name = m.get("name") or m.get("playerName") or ""
            pid = m.get("id") or m.get("playerId") or m.get("primaryId") or ""
            if not pid:
                continue
            if target_full in _slug_name(name):
                best_id = str(pid)
                break
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
st.markdown(
    """
<style>
:root{ --bg:#0f1115; --card:#161a22; --stroke:#252b3a; --muted:#a8b3cf; --soft:#202633; }
.tiles{ display:grid; grid-template-columns:repeat(auto-fill, minmax(330px, 1fr)); gap:14px; }
.tile{ position:relative; background:var(--card); border:1px solid var(--stroke); border-radius:16px; padding:14px; overflow:hidden; box-shadow: 0 2px 12px rgba(0,0,0,.22); }
.row{ display:flex; gap:12px; align-items:flex-start; }
.avatar{ width:72px; height:72px; border-radius:14px; object-fit:cover; background:#0b0d12; border:1px solid #2a3145; }
.name{ font-weight:900; font-size:18px; color:#e8ecff; line-height:1.1; }
.teamline{ margin-top:6px; color:#cbd5f5; font-size:13px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.crest{ width:20px; height:20px; object-fit:contain; border-radius:4px; background:#0b0d12; border:1px solid #2a3145; }
.meta{ margin-top:8px; color:var(--muted); font-size:12px; display:flex; gap:8px; flex-wrap:wrap; }
.chip{ background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:2px 8px; border-radius:10px; }
.match{ position:absolute; top:10px; right:10px; background:#0b0d12; border:1px solid #2a3145; color:#e8ecff; border-radius:12px; padding:6px 10px; font-weight:900; }
.match small{ display:block; font-size:10px; color:var(--muted); font-weight:700; margin-top:1px; text-align:right; }
</style>
""",
    unsafe_allow_html=True
)


# ========================= RENDER TILES (top_n_override parameter — NO globals hack) =========================
def render_tiles(
    ranked: pd.DataFrame,
    role_title: str,
    score_col: str = "Role Fit Score",
    badge_label: str = "Match",
    top_n_override: int = None,
):
    n = int(top_n_override) if top_n_override is not None else int(top_n)

    df_view = ranked.copy()

    if "Age" in df_view.columns:
        df_view["Age"] = pd.to_numeric(df_view["Age"], errors="coerce")
        df_view = df_view[df_view["Age"].between(min_age, max_age)]

    if "Market value" in df_view.columns:
        df_view["Market value"] = pd.to_numeric(df_view["Market value"], errors="coerce")
        df_view = df_view[df_view["Market value"].between(pool_min_value, pool_max_value)]

    df_view = df_view.head(n).copy()
    if df_view.empty:
        st.info("No matches.")
        return

    html = ["<div class='tiles'>"]
    for _, row in df_view.iterrows():
        player = str(row.get("Player", ""))
        team = str(row.get("Team", ""))
        league = str(row.get("League", ""))
        pos = str(row.get("Position", ""))
        age = row.get("Age", "")
        minutes = row.get("Minutes played", "")
        foot = str(row.get("Foot", "")).strip()
        mv = row.get("Market value", np.nan)
        mv_txt = format_market_value_gbp(mv)
        score = float(pd.to_numeric(row.get(score_col, 0.0), errors="coerce") or 0.0)
        match_pct = max(0, min(100, int(round(score))))

        avatar = resolve_player_photo(player, team, league)
        crest = resolve_team_crest(team, league)

        if DEBUG_PHOTOS:
            st.write(player, team, "→", avatar)

        crest_html = f"<img class='crest' src='{crest}' />" if crest else ""
        html.append(
            f"""
<div class="tile">
  <div class="match">{match_pct}%<small>{badge_label}</small></div>
  <div class="row">
    <img class="avatar" src="{avatar}" />
    <div>
      <div class="name">{player}</div>
      <div class="teamline">{crest_html}<span>{team} · {league}</span></div>
      <div class="meta">
        <span class="chip">{pos}</span>
        <span class="chip">Age {age}</span>
        <span class="chip">{int(minutes) if str(minutes).isdigit() else minutes} min</span>
        <span class="chip">{foot}</span>
        <span class="chip">{mv_txt}</span>
      </div>
    </div>
  </div>
</div>
"""
        )

    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_matches_table(
    ranked: pd.DataFrame,
    top_n_override: int = None,
    score_col: str = "Role Fit Score"
):
    n = int(top_n_override) if top_n_override is not None else int(top_n)
    df_view = ranked.copy()

    if "Age" in df_view.columns:
        df_view["Age"] = pd.to_numeric(df_view["Age"], errors="coerce")
        df_view = df_view[df_view["Age"].between(min_age, max_age)]

    if "Market value" in df_view.columns:
        df_view["Market value"] = pd.to_numeric(df_view["Market value"], errors="coerce")
        df_view = df_view[df_view["Market value"].between(pool_min_value, pool_max_value)]

    cols = [
        c for c in
        ["Player", "Team", "League", "Position", "Age", "Minutes played", "Market value", score_col]
        if c in df_view.columns
    ]
    st.dataframe(df_view[cols].head(n), use_container_width=True)


# ========================= SIMILARITY UI (per tab) =========================
def similarity_settings_ui(sim_key_prefix: str, default_leagues: List[str]):
    LS_MAP = globals().get('LEAGUE_STRENGTHS', globals().get('league_strengths', {}))

    _leagues_from_df = df['League'].dropna().unique().tolist() if 'League' in df.columns else []
    _included_from_global = list(globals().get('INCLUDED_LEAGUES', []))
    _included_leagues_cf = sorted(set(_included_from_global) | set(_leagues_from_df))

    _PRESET_LEAGUES_SAFE = globals().get('PRESET_LEAGUES', {})
    _PRESETS_SIM = {
        "All listed leagues": _included_leagues_cf,
        "T5":  sorted(list(_PRESET_LEAGUES_SAFE.get("Top 5 Europe", []))),
        "T20": sorted(list(_PRESET_LEAGUES_SAFE.get("Top 20 Europe", []))),
        "EFL": sorted(list(_PRESET_LEAGUES_SAFE.get("EFL (England 2–4)", []))),
        "Custom": None,
    }

    with st.expander("Similarity settings", expanded=False):
        candidate_league_options = sorted(_included_leagues_cf or _leagues_from_df)
        default_sel = default_leagues if default_leagues else candidate_league_options

        sim_preset_choices = list(_PRESETS_SIM.keys())
        sim_preset = st.selectbox(
            "Candidate league preset",
            sim_preset_choices,
            index=sim_preset_choices.index("All listed leagues"),
            key=f"{sim_key_prefix}_sim_preset"
        )

        preset_vals_raw = _PRESETS_SIM.get(sim_preset) or []
        preset_vals = sorted([lg for lg in preset_vals_raw if lg in candidate_league_options])

        _last_key = f"{sim_key_prefix}__last_sim_preset"
        if st.session_state.get(_last_key) != sim_preset:
            st.session_state[f"{sim_key_prefix}_sim_leagues"] = preset_vals if preset_vals else default_sel
            st.session_state[_last_key] = sim_preset

        sim_leagues = st.multiselect(
            "Candidate leagues",
            candidate_league_options,
            default=st.session_state.get(f"{sim_key_prefix}_sim_leagues", preset_vals if preset_vals else default_sel),
            key=f"{sim_key_prefix}_sim_leagues",
        )

        if preset_vals_raw and not preset_vals:
            st.warning("Preset has leagues, but none match your allowed list/dataset.")
        elif preset_vals_raw:
            st.caption(f"Preset: {sim_preset} — {len(preset_vals)} league(s). You can add/prune below.")

        sim_min_minutes, sim_max_minutes = st.slider("Minutes played (candidates)", 0, 6000, (750, 6000), key=f"{sim_key_prefix}_sim_min")
        sim_min_age, sim_max_age = st.slider("Age (candidates)", 14, 50, (16, 50), key=f"{sim_key_prefix}_sim_age")

        use_strength_filter = st.toggle("Filter by league quality (0–101)", value=False, key=f"{sim_key_prefix}_sim_use_strength")
        if use_strength_filter:
            sim_min_strength, sim_max_strength = st.slider("League quality (strength)", 0, 101, (0, 101), key=f"{sim_key_prefix}_sim_strength")
        else:
            sim_min_strength, sim_max_strength = 0, 101

        percentile_weight = st.slider("Percentile weight", 0.0, 1.0, 0.7, 0.05, key=f"{sim_key_prefix}_sim_pw")

        apply_league_adjust = st.toggle("Apply league difficulty adjustment", value=True, key=f"{sim_key_prefix}_sim_apply_ladj")
        league_weight_sim = st.slider(
            "League weight (difficulty adj.)", 0.0, 1.0, 0.2, 0.05, key=f"{sim_key_prefix}_sim_lw",
            disabled=not apply_league_adjust
        )

        top_n_sim = st.number_input("Show top N", min_value=5, max_value=200, value=14, step=5, key=f"{sim_key_prefix}_sim_top")

    return {
        "LS_MAP": LS_MAP,
        "sim_leagues": sim_leagues,
        "sim_min_minutes": sim_min_minutes,
        "sim_max_minutes": sim_max_minutes,
        "sim_min_age": sim_min_age,
        "sim_max_age": sim_max_age,
        "use_strength_filter": use_strength_filter,
        "sim_min_strength": sim_min_strength,
        "sim_max_strength": sim_max_strength,
        "percentile_weight": percentile_weight,
        "apply_league_adjust": apply_league_adjust,
        "league_weight_sim": league_weight_sim,
        "top_n_sim": top_n_sim,
    }

def compute_similarity_candidates_from_pool(
    ranked_pool: pd.DataFrame,
    tmpl_src: pd.DataFrame,
    sim_features: List[str],
    weights_dict: dict,
    target_league: str,
    settings: dict,
) -> pd.DataFrame:
    if tmpl_src.empty or ranked_pool.empty:
        return pd.DataFrame()

    df_candidates = ranked_pool.copy()
    df_candidates = df_candidates[df_candidates["League"].isin(settings["sim_leagues"])].copy()

    LS_MAP = settings.get("LS_MAP") or {}
    if settings["use_strength_filter"] and LS_MAP:
        df_candidates["League strength"] = df_candidates["League"].map(LS_MAP).fillna(0.0)
        df_candidates = df_candidates[
            (df_candidates["League strength"] >= float(settings["sim_min_strength"])) &
            (df_candidates["League strength"] <= float(settings["sim_max_strength"]))
        ]

    df_candidates["Minutes played"] = pd.to_numeric(df_candidates.get("Minutes played"), errors="coerce")
    df_candidates["Age"] = pd.to_numeric(df_candidates.get("Age"), errors="coerce")
    df_candidates = df_candidates[
        df_candidates["Minutes played"].between(settings["sim_min_minutes"], settings["sim_max_minutes"]) &
        df_candidates["Age"].between(settings["sim_min_age"], settings["sim_max_age"])
    ]

    df_candidates = df_candidates[~((df_candidates["Team"].astype(str) == template_team) & (df_candidates["League"].astype(str) == template_league))].copy()

    sim_out = compute_similarity_from_template(
        tmpl_src=tmpl_src,
        pool=df_candidates,
        sim_features=sim_features,
        weights_dict=weights_dict,
        target_league=target_league,
        percentile_weight=float(settings["percentile_weight"]),
        apply_league_adjust=bool(settings["apply_league_adjust"]),
        league_weight_sim=float(settings["league_weight_sim"]),
    )
    return sim_out

# ========================= TAB WRAPPER =========================
def role_tab(role_title: str, compute_fn, sim_role_key: str):
    ranked, title, tmpl_src = compute_fn()

    # Handle empty returns gracefully (no st.stop())
    if ranked is None or (isinstance(ranked, pd.DataFrame) and ranked.empty):
        st.info(f"No candidates found for {title}.")
        return

    t1, t2, t3 = st.tabs(["Role Fit", "Similar players", "Template players"])

    with t1:
        render_tiles(ranked, title, score_col="Role Fit Score", badge_label="Match")

    with t2:
        weights_dict = SIM_WEIGHTS.get(sim_role_key, {})
        sim_features = [f for f in list(weights_dict.keys()) if f in df.columns]

        if not sim_features:
            st.info("No similarity features found in dataset for this role.")
        else:
            settings = similarity_settings_ui(sim_key_prefix=f"sim_{sim_role_key}", default_leagues=leagues_sel)

            sim_out = compute_similarity_candidates_from_pool(
                ranked_pool=ranked,
                tmpl_src=tmpl_src,
                sim_features=sim_features,
                weights_dict=weights_dict,
                target_league=template_league,
                settings=settings,
            )

            if sim_out.empty:
                st.info("No candidates after similarity filters.")
            else:
                sim_out_display = sim_out.head(int(settings["top_n_sim"])).copy()
                sim_tiles = sim_out_display.copy()
                sim_tiles["Adjusted Similarity"] = pd.to_numeric(sim_tiles["Adjusted Similarity"], errors="coerce").fillna(0.0)
                sim_tiles = sim_tiles.rename(columns={"Adjusted Similarity": "Role Fit Score"})

                # ✅ Use top_n_override parameter — NO globals mutation
                render_tiles(
                    sim_tiles,
                    title,
                    score_col="Role Fit Score",
                    badge_label="Similar",
                    top_n_override=int(settings["top_n_sim"]),
                )

                st.markdown("---")
                render_matches_table(sim_out_display, top_n_override=int(settings["top_n_sim"]), score_col="Adjusted Similarity")

    with t3:
        render_template_players_used(title, tmpl_src)


# ========================= MAIN TABS =========================
tabs = st.tabs(["Strikers", "Attackers", "Central Midfield", "Fullbacks", "Center Backs"])

with tabs[0]:
    role_tab("Strikers", compute_strikers, sim_role_key="ST")

with tabs[1]:
    att_choice = st.selectbox(
        "Attacker subgroup",
        ["All", "Right Wingers", "Left Wingers", "Attacking Midfielders"],
        index=0,
        key="att_role_choice",
    )
    role_tab(f"Attackers ({att_choice})", lambda: compute_attackers(att_choice), sim_role_key="ATT")

with tabs[2]:
    role_tab("Central Midfield", compute_central_mid, sim_role_key="CM")

with tabs[3]:
    fb_choice = st.selectbox("Fullback side", ["All", "Right Backs", "Left Backs"], index=0, key="fb_role_choice")
    role_tab(f"Fullbacks ({fb_choice})", lambda: compute_fullbacks(fb_choice), sim_role_key="FB")

with tabs[4]:
    role_tab("Center Backs", compute_center_backs, sim_role_key="CB")


# ============================== SECTION B (v2 — FULL, TITLES OFF, FULL LABELS) ==============================
st.markdown("---")
st.header("Section B — League Comparison Radar & Tables")

import numpy as _np
import pandas as _pd
import io as _io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def _safe_div(num, den):
    num = _pd.to_numeric(num, errors="coerce")
    den = _pd.to_numeric(den, errors="coerce").replace(0, _np.nan)
    return (num / den).fillna(0.0)

def _pct_from_rank(rank:int, total:int) -> int:
    if total <= 1:
        return 100
    return int(round((1 - (rank - 1) / (total - 1)) * 100))

def _polar_bars(metrics_labels, percentiles):
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

    for i in range(N):
        ax.bar(angles[i], 105, width=bar_width, color='#ffffff22', edgecolor=None, bottom=0, linewidth=0, zorder=0)

    for i in range(N):
        ax.bar(angles[i], percentiles[i], width=bar_width, color=bar_colors[i],
               edgecolor='white', linewidth=1.4, zorder=2)

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


def _cfg_v2(role_key: str):
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
        label_map={"Pass Volume":"Pass Volume","Pass Verticality":"Pass Verticality","Progression Volume":"Progression Volume","Defensive Volume":"Defensive Volume","Positional Demand":"Positional Demand","Aerial Volume":"Aerial Volume"}
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
            out["Attacking Contribution"]= (0.4*_pd.to_numeric(out['xA per 90'],errors="coerce")+0.2*_pd.to_numeric(out['Crosses per 90'],errors="coerce")+0.2*_pd.to_numeric(out['Touches in box per 90'],errors="coerce")+0.1*_pd.to_numeric(out['Shots per 90'],errors="coerce")+0.1*_pd.to_numeric(out['Passes to penalty area per 90'],errors="coerce"))
            out["Defensive Volume"]= (0.5*_pd.to_numeric(out['Defensive duels per 90'],errors="coerce")+0.3*_pd.to_numeric(out['PAdj Interceptions'],errors="coerce")+0.2*_pd.to_numeric(out['Aerial duels per 90'],errors="coerce"))
            out["Pass Volume"]= _pd.to_numeric(out['Passes per 90'],errors="coerce")
            out["Retention"]= _pd.to_numeric(out['Accurate passes, %'],errors="coerce")
            return out
        agg_cols=["Pass Volume","Pass Verticality","Progression Volume","Ball Carrying","Attacking Contribution","Defensive Volume","Retention"]
        label_map={"Pass Volume":"Pass Volume","Pass Verticality":"Pass Verticality","Progression Volume":"Progression Volume","Ball Carrying":"Ball Carrying","Attacking Contribution":"Attacking Contribution","Defensive Volume":"Defensive Volume","Retention":"Retention"}
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
        label_map={"Pass Volume":"Pass Volume","Pass Verticality":"Pass Verticality","Progression Volume":"Progression Volume","Defensive Volume":"Defensive Volume","Interception Volume":"Interception Volume","Retention":"Retention"}
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, label_map=label_map, title="Central Midfielders")

    if role_key == "attack":
        require = ['Accurate passes, %','xG per 90','Non-penalty goals per 90','Touches in box per 90','xA per 90',
            'Passes to penalty area per 90','Passes per 90','Progressive passes per 90','Passes to final third per 90',
            'Dribbles per 90','Progressive runs per 90']
        import re as _re2
        def pos_ok(s: str) -> bool:
            s = str(s).upper().strip()
            main = _re2.split(r'[/,]', s)[0].strip()
            main = main.split()[0]
            if main in ("RW", "LW"):
                return True
            prefixes = ("RWF", "LWF", "LAMF", "RAMF", "AMF")
            return main.startswith(prefixes)
        def compute(df):
            out = df.copy()
            out["Retention Style"] = _pd.to_numeric(out['Accurate passes, %'], errors="coerce")
            out["Goal Threat"] = (0.4*_pd.to_numeric(out['xG per 90'],errors="coerce")+0.4*_pd.to_numeric(out['Non-penalty goals per 90'],errors="coerce")+0.2*_pd.to_numeric(out['Touches in box per 90'],errors="coerce"))
            out["Creativity Threat"] = (0.65*_pd.to_numeric(out['xA per 90'],errors="coerce")+0.35*_pd.to_numeric(out['Passes to penalty area per 90'],errors="coerce"))
            out["Passing Volume"] = _pd.to_numeric(out['Passes per 90'], errors="coerce")
            out["Deeper Playmaking"] = (0.5*_pd.to_numeric(out['Progressive passes per 90'],errors="coerce")+0.5*_pd.to_numeric(out['Passes to final third per 90'],errors="coerce"))
            out["Ball Carrying"] = (0.6*_pd.to_numeric(out['Dribbles per 90'],errors="coerce")+0.4*_pd.to_numeric(out['Progressive runs per 90'],errors="coerce"))
            return out
        agg_cols=["Retention Style","Goal Threat","Creativity Threat","Passing Volume","Deeper Playmaking","Ball Carrying"]
        label_map={"Retention Style":"Retention Style","Goal Threat":"Goal Threat","Creativity Threat":"Creativity Threat","Passing Volume":"Passing Volume","Deeper Playmaking":"Deeper Playmaking","Ball Carrying":"Ball Carrying"}
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, label_map=label_map, title="Attackers")

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
        label_map={"Opportunities":"Opportunities","Ball Carrying":"Carrying Outlet","Aerial Requirement":"Aerial Volume","Passing Volume":"Passing Volume","Goal Output":"Goal Output","Retention":"Retention"}
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, label_map=label_map, title="Strikers")


def _sectionB_for_role(role_key: str):
    cfg=_cfg_v2(role_key)
    leagues = sorted([str(x) for x in df["League"].dropna().unique()])
    topA, topB, topC = st.columns([1.4, 1.6, 1.6])

    with topA:
        included_league = st.selectbox(f"League ({cfg['title']})", leagues, key=f"secB_league_{role_key}")

    league_df = df[df["League"].astype(str)==included_league].copy()
    missing = [c for c in cfg["require_cols"] if c not in league_df.columns]
    if missing:
        st.info(f"Missing columns for {cfg['title']}: {', '.join(missing)}")
        return

    for c in ("Minutes played","Age","Goals"):
        if c in league_df.columns: league_df[c]=_pd.to_numeric(league_df[c], errors="coerce")

    league_df = league_df[league_df["Position"].apply(cfg["pos_filter"])].dropna(subset=cfg["require_cols"])
    if league_df.empty:
        st.info(f"No {cfg['title']} in this league with required stats.")
        return

    fl1, fl2, fl3 = st.columns([1.6,1.6,1.6])
    with fl1:
        teams = sorted(league_df["Team"].dropna().astype(str).unique())
        teams_selected = st.multiselect(
            "Filter teams", teams, default=teams,
            key=f"secB_teams_{role_key}_{included_league}"
        )
        min_minutes, max_minutes = st.slider("Minutes played", 0, 6000, (750, 6000), key=f"secB_min_{role_key}")
        a_min = int(_np.nanmin(league_df["Age"])) if league_df["Age"].notna().any() else 16
        a_max = int(_np.nanmax(league_df["Age"])) if league_df["Age"].notna().any() else 50
        min_age_b, max_age_b = st.slider("Age", a_min, a_max, (16, 50), key=f"secB_age_{role_key}")
    with fl3:
        q = st.text_input("Quick player search (optional)", "", key=f"secB_q_{role_key}")

    pool = league_df[
        league_df["Team"].astype(str).isin(teams_selected)
        & league_df["Minutes played"].between(min_minutes, max_minutes)
        & league_df["Age"].between(min_age_b, max_age_b)
    ].copy()
    if q.strip():
        s=q.strip().lower()
        pool = pool[pool["Player"].astype(str).str.lower().str.contains(s, na=False)]

    if pool.empty:
        st.info("No players after filters.")
        return

    pool = cfg["compute_metrics"](pool).copy()

    left, right = st.columns([1.8, 2.2])
    with left:
        compare_mode = st.radio("Compare", ["Team average","Specific player"], horizontal=True, key=f"secB_mode_{role_key}")
        teams_pool = sorted(pool["Team"].dropna().astype(str).unique())
        target_team = st.selectbox("Target team", teams_pool, key=f"secB_team_{role_key}")

    agg = pool.groupby("Team")[cfg["agg_cols"]].mean().reset_index()

    if compare_mode == "Team average":
        if target_team not in agg["Team"].values:
            st.info("Target team has no eligible players in filtered set.")
            return
        target_vals = agg.set_index("Team").loc[target_team, cfg["agg_cols"]].to_dict()
        label_subject = f"{target_team} AVG"
        exclude_label = target_team
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
        exclude_label = str(prow["Team"].iloc[0])
        team_players_used = None

    with right:
        st.markdown("**Team role averages (post-filter league scope)**")
        sort_col = st.selectbox("Sort by metric", ["Team"] + cfg["agg_cols"], index=0, key=f"secB_sort_{role_key}")
        asc = st.checkbox("Ascending", False, key=f"secB_sort_asc_{role_key}")
        st.dataframe(agg.sort_values(sort_col, ascending=asc), use_container_width=True)
        _download_df_button(agg, f"{cfg['title'].replace(' ','_')}_team_averages.csv", "⬇️ Download team averages (CSV)")

    rows=[]
    for met in cfg["agg_cols"]:
        temp = agg[["Team",met]].copy()
        temp = temp[temp["Team"] != exclude_label].copy()
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

    st.markdown("### 📊 Per-metric league tables (descending)")
    for met_key in cfg["agg_cols"]:
        pretty = cfg["label_map"].get(met_key, met_key)
        tmp = agg[["Team", met_key]].copy().rename(columns={met_key: pretty})
        tmp = tmp.sort_values(by=pretty, ascending=False).reset_index(drop=True)
        tmp.insert(0, "Rank", _np.arange(1, len(tmp)+1))
        with st.expander(f"{pretty} — league table"):
            st.dataframe(tmp, use_container_width=True)

    who = f"{label_subject} ({cfg['title']}) vs league team averages"
    st.subheader(f"📈 {who}")
    st.dataframe(rank_df, use_container_width=True)
    _download_df_button(rank_df, f"{cfg['title'].replace(' ','_')}_rank_summary_{label_subject.replace(' ','_')}.csv",
                        "⬇️ Download ranking summary (CSV)")

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

    labels = [cfg["label_map"].get(m,m) for m in cfg["agg_cols"]]
    percentiles = [int(x) for x in rank_df["Percentile"].tolist()]
    fig = _polar_bars(labels, percentiles)
    st.pyplot(fig, use_container_width=True)
    buf=_io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor()); buf.seek(0)
    st.download_button("⬇️ Download radar", data=buf.getvalue(),
                       file_name=f"SectionB_{cfg['title'].replace(' ','_')}_{label_subject.replace(' ','_')}.png",
                       mime="image/png", key=f"dl_{role_key}_{label_subject}")
    plt.close(fig)

tab_cb, tab_fb, tab_cm, tab_att, tab_st = st.tabs(
    ["Center Backs", "Fullbacks", "Central Midfielders", "Attackers", "Strikers"]
)
with tab_cb:  _sectionB_for_role("cb")
with tab_fb:  _sectionB_for_role("fb")
with tab_cm:  _sectionB_for_role("cm")
with tab_att: _sectionB_for_role("attack")
with tab_st:  _sectionB_for_role("cf")


# ======================== SECTION B (v4.6 — ONE CHART, EVEN SPLIT, MANUAL TEAM RANKS + AUTO ROLE) ========================
st.markdown("---")
st.header("Section B — League Comparison (Single Split Radar)")

import re as _re

def _keyify(s: str) -> str:
    s = str(s or "").strip().lower()
    s = _re.sub(r"[^a-z0-9_]+", "_", s)
    return s[:80] if s else "x"

TEAM_STYLE_LABELS = {
    "cb":     ["Possession", "Passes", "Direct Speed", "xGA", "Goals vs"],
    "fb":     ["Possession", "Passes", "Pressing", "Direct Speed", "xGA"],
    "cm":     ["Possession", "Passes", "Pressing", "Direct Speed", "Passes to Final 3rd"],
    "attack": ["Possession", "Passes", "Pressing", "Direct Speed", "xG"],
    "cf":     ["Possession", "Passes", "Pressing", "Long Balls", "xG"],
}

def _single_split_polar(team_labels, team_pcts, role_labels, role_pcts):
    value_colors = ["#be2a3e", "#e25f48", "#f88f4d", "#f4d166", "#90b960", "#4b9b5f", "#22763f"]
    cmap = LinearSegmentedColormap.from_list("pct_scale", value_colors)

    TEAM_TRACK = "#2b3646"
    ROLE_TRACK = "#362b46"

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
    for r, alpha_v, lw in [(25, 0.08, 0.8), (50, 0.22, 1.8), (75, 0.08, 0.8)]:
        ax.plot(theta_ring, _np.full_like(theta_ring, r),
                color="white", alpha=alpha_v, linewidth=lw, zorder=0)

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

def _cfg_v46(role_key: str):
    role_key = str(role_key).lower().strip()

    if role_key == "cb":
        require = ["Aerial duels per 90","Defensive duels per 90","Passes per 90","Forward passes per 90",
                   "Progressive passes per 90","Progressive runs per 90","PAdj Interceptions","Shots blocked per 90"]
        def pos_ok(s): s=str(s).upper().strip(); return s.startswith(("CB","RCB","LCB"))
        def compute(df):
            out = df.copy()
            out["Pass Verticality"] = _safe_div(out["Forward passes per 90"], out["Passes per 90"])
            out["Passing Volume"]   = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Defensive Volume"] = _pd.to_numeric(out["Defensive duels per 90"], errors="coerce")
            out["Progression Volume"] = (_pd.to_numeric(out["Progressive passes per 90"], errors="coerce")+_pd.to_numeric(out["Progressive runs per 90"], errors="coerce"))
            out["Aerial Volume"] = _pd.to_numeric(out["Aerial duels per 90"], errors="coerce")
            return out
        agg_cols = ["Defensive Volume","Aerial Volume","Passing Volume","Progression Volume","Pass Verticality"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Center Backs")

    if role_key == "fb":
        require = ["Passes per 90","Forward passes per 90","Progressive passes per 90","Progressive runs per 90",
                   "Defensive duels per 90","PAdj Interceptions","Aerial duels per 90","xA per 90","Crosses per 90",
                   "Touches in box per 90","Shots per 90","Passes to penalty area per 90","Accurate passes, %","Dribbles per 90"]
        def pos_ok(s): s=str(s).upper().strip(); return s.startswith(("LB","LWB","RB","RWB"))
        def compute(df):
            out = df.copy()
            out["Progression Volume"] = (_pd.to_numeric(out["Progressive passes per 90"], errors="coerce")+_pd.to_numeric(out["Progressive runs per 90"], errors="coerce"))
            out["Attacking Contribution"] = (0.4*_pd.to_numeric(out["xA per 90"], errors="coerce")+0.2*_pd.to_numeric(out["Crosses per 90"], errors="coerce")+0.2*_pd.to_numeric(out["Touches in box per 90"], errors="coerce")+0.1*_pd.to_numeric(out["Shots per 90"], errors="coerce")+0.1*_pd.to_numeric(out["Passes to penalty area per 90"], errors="coerce"))
            out["Defensive Volume"] = (0.5*_pd.to_numeric(out["Defensive duels per 90"], errors="coerce")+0.3*_pd.to_numeric(out["PAdj Interceptions"], errors="coerce")+0.2*_pd.to_numeric(out["Aerial duels per 90"], errors="coerce"))
            out["Pass Volume"] = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Retention"] = _pd.to_numeric(out["Accurate passes, %"], errors="coerce")
            return out
        agg_cols = ["Defensive Volume","Pass Volume","Attacking Contribution","Retention","Progression Volume"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Fullbacks")

    if role_key == "cm":
        require = ["Passes per 90","Forward passes per 90","Progressive passes per 90","Progressive runs per 90",
                   "Defensive duels per 90","PAdj Interceptions","Accurate passes, %"]
        def pos_ok(s): s=str(s).upper().strip(); return s.startswith(("DMF","LDMF","RDMF","LCMF","RCMF","CMF"))
        def compute(df):
            out = df.copy()
            out["Pass Verticality"] = _safe_div(out["Forward passes per 90"], out["Passes per 90"])
            out["Pass Volume"] = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Progressive Volume"] = (_pd.to_numeric(out["Progressive passes per 90"], errors="coerce")+_pd.to_numeric(out["Progressive runs per 90"], errors="coerce"))
            out["Defensive Volume"] = _pd.to_numeric(out["Defensive duels per 90"], errors="coerce")
            out["Retention"] = _pd.to_numeric(out["Accurate passes, %"], errors="coerce")
            return out
        agg_cols = ["Defensive Volume","Pass Volume","Progressive Volume","Retention","Pass Verticality"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Central Midfielders")

    if role_key == "attack":
        require = ["Accurate passes, %","xG per 90","Non-penalty goals per 90","Touches in box per 90","xA per 90",
                   "Passes to penalty area per 90","Passes per 90","Progressive passes per 90","Passes to final third per 90",
                   "Dribbles per 90","Progressive runs per 90"]
        def pos_ok(s: str) -> bool:
            s = str(s).upper().strip()
            main = _re.split(r"[/,]", s)[0].strip().split()[0]
            if main in ("RW", "LW"): return True
            return main.startswith(("RWF","LWF","LAMF","RAMF","AMF"))
        def compute(df):
            out = df.copy()
            out["Goal Threat"] = (0.4*_pd.to_numeric(out["xG per 90"], errors="coerce")+0.4*_pd.to_numeric(out["Non-penalty goals per 90"], errors="coerce")+0.2*_pd.to_numeric(out["Touches in box per 90"], errors="coerce"))
            out["Creative Threat"] = (0.65*_pd.to_numeric(out["xA per 90"], errors="coerce")+0.35*_pd.to_numeric(out["Passes to penalty area per 90"], errors="coerce"))
            out["Pass Volume"] = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Deep Playmaking"] = (0.5*_pd.to_numeric(out["Progressive passes per 90"], errors="coerce")+0.5*_pd.to_numeric(out["Passes to final third per 90"], errors="coerce"))
            out["Ball Carrying"] = (0.6*_pd.to_numeric(out["Dribbles per 90"], errors="coerce")+0.4*_pd.to_numeric(out["Progressive runs per 90"], errors="coerce"))
            return out
        agg_cols = ["Pass Volume","Deep Playmaking","Ball Carrying","Goal Threat","Creative Threat"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Attackers")

    if role_key == "cf":
        require = ["Touches in box per 90","xG per 90","Dribbles per 90","Progressive runs per 90",
                   "Aerial duels per 90","Aerial duels won, %","Passes per 90","Accurate passes, %"]
        def pos_ok(s): return str(s).upper().strip().startswith("CF")
        def compute(df):
            out = df.copy()
            out["Opportunities"] = (0.7*_pd.to_numeric(out["Touches in box per 90"], errors="coerce")+0.3*_pd.to_numeric(out["xG per 90"], errors="coerce"))
            out["Carrying Outlet"] = (0.65*_pd.to_numeric(out["Dribbles per 90"], errors="coerce")+0.35*_pd.to_numeric(out["Progressive runs per 90"], errors="coerce"))
            out["Aerial Volume"] = (_pd.to_numeric(out["Aerial duels per 90"], errors="coerce")*_pd.to_numeric(out["Aerial duels won, %"], errors="coerce")/100.0)
            out["Pass Volume"] = _pd.to_numeric(out["Passes per 90"], errors="coerce")
            out["Retention"] = _pd.to_numeric(out["Accurate passes, %"], errors="coerce")
            return out
        agg_cols = ["Aerial Volume","Pass Volume","Opportunities","Retention","Carrying Outlet"]
        return dict(pos_filter=pos_ok, require_cols=require, compute_metrics=compute, agg_cols=agg_cols, title="Strikers")

    return None

def _sectionB_singlechart_manualteam_for_role(role_key: str, kbase: str = "secB_SINGLE_v46"):
    cfg = _cfg_v46(role_key)
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

    league_df = league_df[league_df["Position"].apply(cfg["pos_filter"])].dropna(subset=cfg["require_cols"])
    if league_df.empty:
        st.info(f"No {cfg['title']} in this league with required stats.")
        return

    f1, f2 = st.columns([2.0, 1.4])
    with f1:
        q = st.text_input("Quick player search (optional)", "", key=f"{K}_q")
    with f2:
        min_minutes_sc, max_minutes_sc = st.slider("Minutes played", 0, 6000, (750, 6000), key=f"{K}_mins")
        a_min = int(_np.nanmin(league_df["Age"])) if league_df["Age"].notna().any() else 16
        a_max = int(_np.nanmax(league_df["Age"])) if league_df["Age"].notna().any() else 50
        min_age_sc, max_age_sc = st.slider("Age", a_min, a_max, (16, 50), key=f"{K}_age")

    pool = league_df[
        league_df["Minutes played"].between(min_minutes_sc, max_minutes_sc)
        & league_df["Age"].between(min_age_sc, max_age_sc)
    ].copy()

    if q.strip():
        s = q.strip().lower()
        pool = pool[pool["Player"].astype(str).str.lower().str.contains(s, na=False)]

    if pool.empty:
        st.info("No players after filters.")
        return

    pool = cfg["compute_metrics"](pool).copy()

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

    N_TEAMS = int(rank_role_df["Total teams"].max()) if not rank_role_df.empty else int(agg_role["Team"].nunique())
    N_TEAMS = max(1, N_TEAMS)

    st.markdown("### ✍️ Team Style — enter ranks (1 = best, N = worst)")
    st.caption(f"League benchmark size used for ranks: **N = {N_TEAMS} teams** (post minutes/age/search filters)")

    team_labels = TEAM_STYLE_LABELS.get(str(role_key).lower().strip(), ["Metric1","Metric2","Metric3","Metric4","Metric5"])[:5]
    cols_sc = st.columns(5)
    team_ranks = []
    for col, lab in zip(cols_sc, team_labels):
        with col:
            r = st.number_input(
                f"{lab} (rank 1–{N_TEAMS})",
                min_value=1, max_value=int(N_TEAMS),
                value=1, step=1,
                key=f"{K}_teamrank_{_keyify(lab)}"
            )
            team_ranks.append(int(r))
    team_pcts = [_pct_from_rank(r, N_TEAMS) for r in team_ranks]

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

tab_cb2, tab_fb2, tab_cm2, tab_att2, tab_st2 = st.tabs(
    ["Center Backs", "Fullbacks", "Central Midfielders", "Attackers", "Strikers"]
)
with tab_cb2:  _sectionB_singlechart_manualteam_for_role("cb", kbase="secB_SINGLE_v46")
with tab_fb2:  _sectionB_singlechart_manualteam_for_role("fb", kbase="secB_SINGLE_v46")
with tab_cm2:  _sectionB_singlechart_manualteam_for_role("cm", kbase="secB_SINGLE_v46")
with tab_att2: _sectionB_singlechart_manualteam_for_role("attack", kbase="secB_SINGLE_v46")
with tab_st2:  _sectionB_singlechart_manualteam_for_role("cf", kbase="secB_SINGLE_v46")


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

CONTRACT_COL = "Contract expires"

try:
    from adjustText import adjust_text
    HAVE_ADJUSTTEXT = True
except ImportError:
    HAVE_ADJUSTTEXT = False

with st.expander("Squad Profile settings", expanded=False):

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

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    min_minutes_s, max_minutes_s = st.slider(
        "Minutes range (for axis & filter)",
        0, 5000, (0, 4000), step=250, key="sq_min",
    )

    min_age_s, max_age_s = st.slider(
        "Age range (for axis & filter)",
        14, 45, (16, 40), key="sq_age",
    )

    st.markdown("**Minutes bands (horizontal dashed lines)**")
    important_line = st.slider("Important Player line (minutes)", 0, 5000, 500, step=250, key="sq_line_important")
    crucial_line = st.slider("Crucial Player line (minutes)", 0, 5000, 1000, step=250, key="sq_line_crucial")

    band_lines = sorted(
        [("Important Player", important_line), ("Crucial Player", crucial_line)],
        key=lambda x: x[1],
    )

    auto_contract_red = st.checkbox(
        "Highlight players with contract ≤ 2026 in red", value=True, key="sq_auto_contract"
    )

    team_players_all = sorted(df[df["Team"] == squad_team]["Player"].dropna().unique().tolist())
    custom_red_players = st.multiselect(
        "Force-highlight specific players in red",
        options=team_players_all, default=[], key="sq_custom_red",
    )

    show_labels = st.toggle("Show labels", value=True, key="sq_show_labels")
    label_size = st.slider("Label size", 8, 22, 15, 1, key="sq_lblsize")
    point_size = st.slider("Point size", 24, 300, 300, 2, key="sq_pts")
    point_alpha = st.slider("Point opacity", 0.2, 1.0, 0.92, 0.02, key="sq_alpha")

    PAGE_BG = "#0a0f1c"
    PLOT_BG = "#0a0f1c"
    GRID_MAJ = "#3a4050"
    txt_col = "#f1f5f9"

    canvas_preset = st.selectbox(
        "Canvas size", ["1280×720", "1600×900", "1920×820", "1920×1080"],
        index=1, key="sq_canvas",
    )
    w_px, h_px = map(int, canvas_preset.replace("×", "x").split("x"))

    top_gap_px = st.slider("Top gap (px)", 0, 240, 80, 5, key="sq_gap")
    render_exact = st.checkbox("Render exact pixels (PNG)", value=True, key="sq_exact")

squad = df[df["Team"] == squad_team].copy()
if squad.empty:
    st.info("No players found for this squad.")
else:
    squad["Minutes played"] = pd.to_numeric(squad["Minutes played"], errors="coerce")
    squad["Age"] = pd.to_numeric(squad["Age"], errors="coerce")

    squad = squad[
        squad["Minutes played"].between(min_minutes_s, max_minutes_s)
        & squad["Age"].between(min_age_s, max_age_s)
    ]

    if squad.empty:
        st.info("No players after applying filters.")
    else:
        if auto_contract_red and CONTRACT_COL in squad.columns:
            contract_year = (
                squad[CONTRACT_COL].astype(str).str.extract(r"(\d{4})")[0].astype(float)
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

        line_col = "#FFFFFF"
        AGE_BAND_LABELS = ["YOUTH", "ASCENT", "PRIME", "EXPERIENCED", "OLD"]
        AGE_BAND_EDGES = [16, 21, 25, 29, 33, 45]

        for al in [21, 25, 29, 33]:
            if min_age_s <= al <= max_age_s:
                ax.axvline(al, color=line_col, linestyle=(0, (4, 4)), lw=1.5)

        for i, label in enumerate(AGE_BAND_LABELS):
            band_start = AGE_BAND_EDGES[i]
            band_end = AGE_BAND_EDGES[i + 1]
            visible_start = max(band_start, min_age_s)
            visible_end = min(band_end, max_age_s)
            if visible_start >= visible_end or max_age_s == min_age_s:
                continue
            center = (visible_start + visible_end) / 2.0
            x_frac = (center - min_age_s) / float(max_age_s - min_age_s)
            ax.text(x_frac, 1.01, label, transform=ax.transAxes,
                    fontsize=20, fontweight="bold", color=txt_col, ha="center", va="bottom")

        for name, y_val in band_lines:
            if min_minutes_s <= y_val <= max_minutes_s:
                ax.axhline(y_val, color=line_col, linestyle=(0, (4, 4)), lw=1.5)
                ax.text(min_age_s + 0.2, y_val + (max_minutes_s - min_minutes_s) * 0.01, name,
                        fontsize=14, fontweight="bold", color="#020617",
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="#e5e7eb", edgecolor="none", alpha=0.95),
                        va="bottom")

        effective_point_size = point_size * 1.1
        for is_red, grp in squad.groupby("IsRed"):
            ax.scatter(grp["Age"], grp["Minutes played"],
                       s=effective_point_size,
                       c="#ef4444" if is_red else "#e5e7eb",
                       alpha=point_alpha, edgecolors="none", linewidth=0,
                       zorder=3 if is_red else 2)

        if show_labels:
            label_df = squad.copy()
            axis_height = max_minutes_s - min_minutes_s
            top_margin = axis_height * 0.04
            bottom_margin = axis_height * 0.03

            if HAVE_ADJUSTTEXT:
                texts = []
                xs = label_df["Age"].values
                ys = label_df["Minutes played"].values
                for x, y, name, is_red in zip(xs, ys, label_df["Player"], label_df["IsRed"]):
                    t = ax.text(x, y, name, fontsize=label_size, color=txt_col,
                                weight="semibold", ha="center", va="bottom",
                                zorder=6 if is_red else 5)
                    t.set_path_effects([pe.withStroke(linewidth=2, foreground="#020617", alpha=0.9)])
                    texts.append(t)
                adjust_text(texts, x=xs, y=ys, ax=ax, autoalign="y",
                            only_move={"points": "y", "text": "xy"},
                            force_points=0.7, force_text=0.7,
                            expand_points=(1.1, 1.5), expand_text=(1.1, 1.5),
                            arrowprops=dict(arrowstyle="-", lw=0.6, color=txt_col, alpha=0.6))
                for t in texts:
                    x_lab, y_lab = t.get_position()
                    y_lab = max(min_minutes_s + bottom_margin, min(y_lab, max_minutes_s - top_margin))
                    t.set_position((x_lab, y_lab))
            else:
                axis_height = max_minutes_s - min_minutes_s
                base_offset = axis_height * 0.015
                min_y_delta = axis_height * 0.05
                age_tol = 0.7
                x_jitter = 0.25

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

                for _, r in label_df.iterrows():
                    x = float(r["Age"])
                    y = float(r["Minutes played"])
                    x_lab, y_lab = positions.get(r["Player"], (x, y + base_offset))
                    if abs(x_lab - x) > 0.05 or abs(y_lab - (y + base_offset)) > 0.05:
                        ax.plot([x, x_lab], [y, y_lab], linestyle="-", linewidth=0.5,
                                color=txt_col, alpha=0.5, zorder=5)
                    z = 6 if r["IsRed"] else 5
                    t = ax.annotate(r["Player"], xy=(x_lab, y_lab), textcoords="data",
                                    fontsize=label_size, color=txt_col, weight="semibold",
                                    ha="center", va="bottom", zorder=z)
                    t.set_path_effects([pe.withStroke(linewidth=2, foreground="#020617", alpha=0.9)])

        fig.subplots_adjust(
            left=0.06, right=0.98, bottom=0.11,
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
# ============================== END FEATURE R ==============================

































































