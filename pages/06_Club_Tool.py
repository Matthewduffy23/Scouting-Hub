# streamlit_app.py
# ✅ Club Scouting — CLEAN Pro Tiles (FotMob photos + crests only)
# - Dataset picker (WORLD*.csv or upload)
# - Pool filters + scoring controls
# - Template team selector
# - Tabs: Strikers / Attackers / Central Mid / Fullbacks / Center Backs
# - Per role: Tabs = Tiles + Template players
# - Tiles: click/expand → role scores (pills) + full metrics (values + percentiles)
# - FotMob-only: squad API for player images + crest URLs
# - Optional per-player photo override URL stored in session only

import io, math, re, time, unicodedata
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from numpy.linalg import norm
from numpy import exp
import requests
from difflib import SequenceMatcher

# ========================= PAGE =========================
st.set_page_config(page_title="Club Scouting — Pro Tiles (FotMob)", layout="wide")
st.title("🔎 Advanced Club Scouting — Pro Tiles (FotMob)")
st.caption("Template team → role fit matching → tiles. Match% = Role Fit Score (distance + optional league blend).")

# ========================= DATASET PICKER =========================
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def _candidate_dirs() -> List[Path]:
    """Directories to search for WORLD*.csv files."""
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
    """Find WORLD*.csv across candidate dirs."""
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
    for k in [
        "photo_map", "crest_map",
        "st_tmpl_pick", "att_tmpl_pick", "cm_tmpl_pick", "fb_tmpl_pick", "cb_tmpl_pick",
        "att_role_choice", "fb_role_choice",
    ]:
        st.session_state.pop(k, None)
    st.session_state["_active_dataset_name"] = DATASET_NAME

st.session_state.setdefault("photo_map", {})  # player overrides
st.session_state.setdefault("crest_map", {})  # team overrides

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
'Albania 1.':51.96,'Bosnia 1.':51.76,'Japan 2.':50.98,'England 4.':50.78,
'Ireland 1.':50.59,'Russia 1.':62.41,'Kazakhstan 1.':50.39,'France 3.':49.61,
'Tunisia 1.':49.22,'Venezuela 1.':48.63,'Belgium 2.':48.43,'Finland 1.':48.43,'Armenia 1.':47.84,
'Georgia 1.':47.65,'Switzerland 2.':46.47,'Qatar 1.':46.27,'Uzbekistan 1.':46.27,'Poland 2.':46.27,
'Iceland 1.':46.08,'Norway 2.':45.88,'Sweden 2.':45.69,'North Macedonia 1.':44.71,'Turkey 2.':44.51,
'Czech 2.':43.33,'Brazil 3.':43.14,'Lithuania 1.':42.35,'Netherlands 2.':42.16,
'Malta 1.':41.96,'Italy 3.':45,'Denmark 2.':40.39,'Moldova 1.':40.39,'USA 2.':40.00,
'Latvia 1.':40.00,'Scotland 2.':38.63,'Canada 1.':38.24,'Austria 2.':38.24,
'Israel 2.':38.04,'England 7.':37.25,'Germany 4.':35.29,'Portugal 3.':35.29,'England 5.':33.33,
'Estonia 1.':40,'England 9.':31.37,'Northern Ireland 1.':30.98,'Serbia 2.':30.39,
'Slovakia 2.':28.24,'Wales 1.':26.67,'Scotland 3.':20.00,'England 6.':16.08,'England 8.':15.69,'England 10.':3.92,
'Estonia 2.':3
}

# ========================= TOP BAR: FILTERS + SCORING =========================
st.markdown("---")
st.header("⚙️ Adjustments & Candidate Pool")

cA, cB, cC, cD = st.columns([1.2, 1.6, 1.2, 1.2])
with cA:
    use_top5 = st.checkbox("Top-5 preset", False)
    use_top20 = st.checkbox("Top-20 preset", False)
    use_efl = st.checkbox("EFL preset", False)

seed = set()
if use_top5: seed |= PRESET_LEAGUES["Top 5 Europe"]
if use_top20: seed |= PRESET_LEAGUES["Top 20 Europe"]
if use_efl: seed |= PRESET_LEAGUES["EFL (England 2–4)"]

leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df["League"].dropna().unique()))
default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES

with cB:
    leagues_sel = st.multiselect("Leagues in candidate pool", leagues_avail, default=default_leagues)

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
    decay_rate = st.slider("Exp. decay (↑=stricter)", 0.5, 10.0, 5.0, 0.5)
with cF:
    use_league_weighting = st.checkbox("Blend league strength (β)", value=True)
    beta = st.slider("β (0–1)", 0.0, 1.0, 0.40, 0.05)
with cG:
    use_league_mismatch = st.checkbox("League mismatch penalty (α,p)", value=True)
    alpha = st.slider("α", 0.0, 5.0, 1.20, 0.05)
    p_exp = st.slider("p", 1.0, 3.0, 1.50, 0.10)
with cH:
    penalty_mode = st.selectbox("Penalty combine", ["Additive (stronger)", "Quadrature (gentler)"], index=0)
    min_strength, max_strength = st.slider("League strength (pool)", 0, 101, (0, 101))
    top_n = st.number_input("Top N", 5, 200, 20, 5)

DEBUG_PHOTOS = st.checkbox("Debug photos", False)

# ========================= TEMPLATE TEAM =========================
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

# ========================= HELPERS (SCOUTING MATH) =========================
def build_base_pool() -> pd.DataFrame:
    """
    Base candidate pool:
      - league filter
      - minutes, age, market value filters
      - league strength filter
    """
    p = df.copy()
    p = p[p["League"].isin(leagues_sel)]

    for c in ["Minutes played", "Age", "Market value", "Goals"]:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")

    p = p[p["Minutes played"].between(min_minutes, max_minutes)]
    p = p[p["Age"].between(min_age, max_age)]
    p = p[p["Market value"].between(pool_min_value, pool_max_value)]

    p["League Strength"] = p["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
    p = p[(p["League Strength"] >= float(min_strength)) & (p["League Strength"] <= float(max_strength))]
    return p

def _template_rows_for_role(pos_predicate) -> pd.DataFrame:
    """Template players from selected team+league filtered by role position predicate + minutes."""
    src = df[
        (df["League"].astype(str) == template_league)
        & (df["Team"].astype(str) == template_team)
        & (df["Position"].apply(lambda p: pos_predicate(str(p))))
    ].copy()
    src["Minutes played"] = pd.to_numeric(src["Minutes played"], errors="coerce")
    src = src[src["Minutes played"] >= min_minutes_template]
    return src

def _score_block(df_with_baseDist: pd.DataFrame) -> pd.DataFrame:
    """
    Converts BaseDist into Role Fit Score:
      1) optional league mismatch penalty inside distance
      2) exp-decay distance -> 0..100 base score
      3) optional blend with league strength (beta)
    """
    if df_with_baseDist.empty:
        return df_with_baseDist

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
    """Forward passes / passes with safe zero handling."""
    f = pd.to_numeric(forward_per90, errors="coerce")
    p = pd.to_numeric(passes_per90, errors="coerce").replace(0, np.nan)
    return (f / p).fillna(0.0)

# ========================= ROLE SCORE (PILLS) HELPERS =========================
def pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank 0..100 within a series."""
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True) * 100.0

def ensure_percentiles(df_role: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Adds '<metric> Percentile' columns for all metrics in df_role."""
    out = df_role.copy()
    for m in metrics:
        if m in out.columns:
            out[m] = pd.to_numeric(out[m], errors="coerce")
            out[m + " Percentile"] = pct_rank(out[m])
    return out

def compute_role_pills(df_role: pd.DataFrame, roles: Dict[str, Dict], prefix: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates columns like f"{prefix}{role_name} Score" (0-99 int).
    Each role score is a weighted average of metric percentiles.
    """
    out = df_role.copy()
    created_cols = []
    for rname, rdef in roles.items():
        weights = (rdef.get("metrics") or {})
        use_metrics = [m for m in weights.keys() if m in out.columns]
        if not use_metrics:
            continue

        for m in use_metrics:
            pc = m + " Percentile"
            if pc not in out.columns:
                out[pc] = pct_rank(out[m])

        wsum = float(sum(weights[m] for m in use_metrics)) or 1.0
        score = 0.0
        for m in use_metrics:
            score = score + (pd.to_numeric(out[m + " Percentile"], errors="coerce").fillna(0.0) * float(weights[m]))
        score = (score / wsum)  # 0..100
        score99 = (score * 0.99).clip(0, 99).round(0).astype(int)
        col = f"{prefix}{rname} Score"
        out[col] = score99
        created_cols.append(col)

    return out, created_cols

def pretty_pill_name(col: str) -> str:
    return col.replace(" Score", "").strip()

# ========================= ROLE DEFINITIONS (PILLS) =========================
ATT_ROLES = {
    "Playmaker": {"metrics": {
        "Passes per 90": 2, "xA per 90": 3, "Key passes per 90": 1,
        "Deep completions per 90": 1.5, "Smart passes per 90": 1.5, "Passes to penalty area per 90": 2,
    }},
    "Goal Threat": {"metrics": {
        "xG per 90": 3, "Non-penalty goals per 90": 3, "Shots per 90": 2, "Touches in box per 90": 2,
    }},
    "Ball Carrier": {"metrics": {
        "Dribbles per 90": 4, "Successful dribbles, %": 2, "Progressive runs per 90": 3, "Accelerations per 90": 3,
    }},
}

ST_ROLES = {
    "Target Man": {"metrics": {"Aerial duels per 90": 3, "Aerial duels won, %": 5}},
    "Goal Threat": {"metrics": {"Non-penalty goals per 90": 3, "Shots per 90": 1.5, "xG per 90": 3, "Touches in box per 90": 1, "Shots on target, %": 0.5}},
    "Link-Up": {"metrics": {
        "Passes per 90": 2, "Passes to penalty area per 90": 1.5, "Deep completions per 90": 1, "Smart passes per 90": 1.5,
        "Accurate passes, %": 1.5, "Key passes per 90": 1, "Dribbles per 90": 2, "Successful dribbles, %": 1,
        "Progressive runs per 90": 2, "xA per 90": 3,
    }},
}

CM_ROLES = {
    "Deep Playmaker": {"metrics": {"Passes per 90": 2.5, "Accurate passes, %": 2, "Forward passes per 90": 2, "Progressive passes per 90": 2, "Key passes per 90": 1, "xA per 90": 1}},
    "Ball Winner": {"metrics": {"Defensive duels per 90": 2.5, "Defensive duels won, %": 2.5, "PAdj Interceptions": 3, "Aerial duels per 90": 1}},
    "Box to Box": {"metrics": {"Progressive runs per 90": 3, "Shots per 90": 1.5, "Touches in box per 90": 2, "Progressive passes per 90": 1.5, "Defensive duels per 90": 1.5}},
}

FB_ROLES = {
    "Progressor": {"metrics": {"Progressive passes per 90": 2.5, "Progressive runs per 90": 2.5, "Forward passes per 90": 1.5, "Passes per 90": 1}},
    "Creator": {"metrics": {"xA per 90": 3, "Crosses per 90": 2, "Passes to penalty area per 90": 2, "Touches in box per 90": 1}},
    "Defender": {"metrics": {"Defensive duels per 90": 2.5, "PAdj Interceptions": 2.5, "Aerial duels per 90": 1.5, "Defensive duels won, %": 1.5}},
}

CB_ROLES = {
    "Stopper": {"metrics": {"Defensive duels per 90": 3, "Aerial duels per 90": 2.5, "Aerial duels won, %": 2.5, "Shots blocked per 90": 1.5}},
    "Ball Player": {"metrics": {"Passes per 90": 2.5, "Accurate passes, %": 2, "Forward passes per 90": 2, "Progressive passes per 90": 2, "Deep completions per 90": 1}},
    "Reader": {"metrics": {"PAdj Interceptions": 3, "Shots blocked per 90": 2, "Defensive duels won, %": 2}},
}

# ========================= ROLE METRIC SECTIONS (DISPLAY) =========================
DISPLAY_SECTIONS = {
    "ATTACKING": [
        ("Non-pen goals", "Non-penalty goals per 90"),
        ("xG", "xG per 90"),
        ("Shots", "Shots per 90"),
        ("Shots on target %", "Shots on target, %"),
        ("Touches in box", "Touches in box per 90"),
        ("xA", "xA per 90"),
    ],
    "POSSESSION": [
        ("Passes", "Passes per 90"),
        ("Accurate passes %", "Accurate passes, %"),
        ("Forward passes", "Forward passes per 90"),
        ("Prog passes", "Progressive passes per 90"),
        ("Prog runs", "Progressive runs per 90"),
        ("Dribbles", "Dribbles per 90"),
        ("Dribble success %", "Successful dribbles, %"),
        ("Key passes", "Key passes per 90"),
        ("Deep completions", "Deep completions per 90"),
        ("Smart passes", "Smart passes per 90"),
        ("Passes to PA", "Passes to penalty area per 90"),
    ],
    "DEFENSIVE": [
        ("Def duels", "Defensive duels per 90"),
        ("Def duels won %", "Defensive duels won, %"),
        ("PAdj Int", "PAdj Interceptions"),
        ("Aerial duels", "Aerial duels per 90"),
        ("Aerial won %", "Aerial duels won, %"),
        ("Shots blocked", "Shots blocked per 90"),
    ],
}

def _metric_value(row: pd.Series, met: str):
    v = row.get(met, np.nan)
    try:
        v = float(v)
        if np.isnan(v):
            return "—"
        return f"{v:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(v) if pd.notna(v) else "—"

def _metric_pct(row: pd.Series, met: str):
    col = met + " Percentile"
    v = row.get(col, np.nan)
    try:
        v = float(v)
        if np.isnan(v):
            return 0
        return int(round(max(0, min(100, v))))
    except Exception:
        return 0

def build_metrics_html(row: pd.Series) -> str:
    """HTML block: 3 columns sections with label/value + percentile badge."""
    def sec(title, pairs):
        rows = []
        for lab, met in pairs:
            if met not in row.index:
                continue
            val = _metric_value(row, met)
            pct = _metric_pct(row, met)
            rows.append(
                f"<div class='m-row'><div class='m-label'>{lab}</div>"
                f"<div class='m-right'><div class='m-val'>{val}</div><div class='m-badge'>{pct:02d}</div></div></div>"
            )
        if not rows:
            rows.append("<div class='m-row'><div class='m-label'>—</div><div class='m-right'><div class='m-val'>—</div><div class='m-badge'>00</div></div></div>")
        return f"<div class='m-sec'><div class='m-title'>{title}</div>{''.join(rows)}</div>"

    return (
        "<div class='metrics-grid'>"
        + sec("ATTACKING", DISPLAY_SECTIONS["ATTACKING"])
        + sec("POSSESSION", DISPLAY_SECTIONS["POSSESSION"])
        + sec("DEFENSIVE", DISPLAY_SECTIONS["DEFENSIVE"])
        + "</div>"
    )

# ========================= FOTMOB HELPERS =========================
def _fotmob_team_id_from_url(team_url: str) -> str:
    m = re.search(r"/teams/(\d+)/", str(team_url or ""))
    return m.group(1) if m else ""

def _fotmob_crest_url(team_url: str) -> str:
    tid = _fotmob_team_id_from_url(team_url)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png" if tid else ""

def _fotmob_team_squad(team_id: str) -> List[dict]:
    """FotMob teams API -> cached squad list."""
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
    repl = {"ø":"o","œ":"oe","æ":"ae","å":"a","ä":"a","ö":"o","ü":"u","ß":"ss","ł":"l","đ":"d","ð":"d","þ":"th","ç":"c","ş":"s","ğ":"g","ı":"i"}
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
    Crest priority:
      1) session override (crest_map)
      2) FotMob teamlogo/{team_id}.png derived from team_fotmob_urls.py
      3) empty string
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


# ========================= UI STYLES =========================
st.markdown(
    """
<style>
:root{
  --bg:#0f1115; --card:#161a22; --stroke:#252b3a; --muted:#a8b3cf; --soft:#202633;
  --pill:#0b0d12; --pillStroke:#2a3145;
}
.block-container{ padding-top:.8rem; }
.tiles{ display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:14px; }
.tile{
  position:relative; background:var(--card); border:1px solid var(--stroke);
  border-radius:16px; padding:14px; overflow:hidden;
  box-shadow: 0 2px 12px rgba(0,0,0,.22);
}
.row{ display:flex; gap:12px; align-items:flex-start; }
.avatar{ width:74px; height:74px; border-radius:14px; object-fit:cover; background:#0b0d12; border:1px solid var(--pillStroke); }
.name{ font-weight:900; font-size:18px; color:#e8ecff; line-height:1.1; }
.teamline{ margin-top:6px; color:#cbd5f5; font-size:13px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.crest{ width:20px; height:20px; object-fit:contain; border-radius:4px; background:#0b0d12; border:1px solid var(--pillStroke); }
.meta{ margin-top:8px; color:var(--muted); font-size:12px; display:flex; gap:8px; flex-wrap:wrap; }
.chip{ background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:2px 8px; border-radius:10px; }
.match{ position:absolute; top:10px; right:10px; background:var(--pill); border:1px solid var(--pillStroke); color:#e8ecff; border-radius:12px; padding:6px 10px; font-weight:900; }
.match small{ display:block; font-size:10px; color:var(--muted); font-weight:700; margin-top:1px; text-align:right; }

.pills{ display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 10px 0; }
.pill{ background:var(--pill); border:1px solid var(--pillStroke); color:#e8ecff; border-radius:999px; padding:6px 10px; font-size:12px; font-weight:800; }
.pill span{ color:var(--muted); font-weight:700; margin-left:6px; }

.metrics-grid{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; }
.m-sec{ background:#0f1219; border:1px solid #22293a; border-radius:14px; padding:10px; }
.m-title{ color:#cbd5f5; font-weight:900; font-size:12px; letter-spacing:.06em; margin-bottom:6px; }
.m-row{ display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid rgba(255,255,255,.06); }
.m-row:last-child{ border-bottom:none; }
.m-label{ color:#cbd5f5; font-size:12px; max-width:65%; }
.m-right{ display:flex; align-items:center; gap:10px; }
.m-val{ color:#e8ecff; font-weight:800; font-size:12px; min-width:44px; text-align:right; }
.m-badge{ width:28px; height:22px; border-radius:8px; background:#0b0d12; border:1px solid #2a3145; color:#e8ecff; display:flex; align-items:center; justify-content:center; font-weight:900; font-size:11px; }
@media (max-width: 1100px){ .metrics-grid{ grid-template-columns:1fr; } }
</style>
""",
    unsafe_allow_html=True,
)

def _fmt_int(v):
    try:
        if pd.isna(v):
            return "—"
        return f"{int(float(v))}"
    except Exception:
        return str(v) if pd.notna(v) else "—"

def _fmt_money(v):
    try:
        if pd.isna(v):
            return "—"
        x = float(v)
        if x >= 1_000_000:
            return f"{x/1_000_000:.1f}M".rstrip("0").rstrip(".") + "€"
        return f"{int(x)}€"
    except Exception:
        return str(v) if pd.notna(v) else "—"


# ========================= ENRICHMENT: percentiles + pills per role =========================
def _role_bundle(role_key: str):
    """
    Map role_key -> (role_defs, pill_prefix)
    role_key values used below: 'ST', 'ATT', 'CM', 'FB', 'CB'
    """
    if role_key == "ST":
        return ST_ROLES, ""
    if role_key == "ATT":
        return ATT_ROLES, ""
    if role_key == "CM":
        return CM_ROLES, ""
    if role_key == "FB":
        return FB_ROLES, ""
    if role_key == "CB":
        return CB_ROLES, ""
    return {}, ""

def enrich_for_display(ranked: pd.DataFrame, role_key: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Adds:
      - percentiles for display metrics + any role-pill metrics
      - role pill score columns
    Returns:
      df_enriched, pill_cols, metrics_used_for_percentiles
    """
    df_role = ranked.copy()

    roles, prefix = _role_bundle(role_key)
    pill_metrics = []
    for r in roles.values():
        pill_metrics.extend(list((r.get("metrics") or {}).keys()))
    pill_metrics = sorted(set([m for m in pill_metrics if isinstance(m, str)]))

    display_metrics = []
    for _, pairs in DISPLAY_SECTIONS.items():
        for _, met in pairs:
            display_metrics.append(met)
    display_metrics = sorted(set(display_metrics))

    needed = sorted(set(pill_metrics + display_metrics))
    df_role = ensure_percentiles(df_role, needed)
    df_role, pill_cols = compute_role_pills(df_role, roles, prefix)

    return df_role, pill_cols, needed

def top_role_pills(row: pd.Series, pill_cols: List[str], topk: int = 3) -> List[Tuple[str, int]]:
    vals = []
    for c in pill_cols:
        try:
            vals.append((pretty_pill_name(c), int(row.get(c, 0))))
        except Exception:
            vals.append((pretty_pill_name(c), 0))
    vals.sort(key=lambda x: x[1], reverse=True)
    return vals[:topk]


# ========================= PRO CARDS (tiles + expand) =========================
def render_pro_card(row: pd.Series, pill_cols: List[str], role_key: str, card_key: str, show_override: bool):
    player = str(row.get("Player", ""))
    team = str(row.get("Team", ""))
    league = str(row.get("League", ""))
    pos = str(row.get("Position", ""))
    age = row.get("Age", np.nan)
    minutes = row.get("Minutes played", np.nan)
    mv = row.get("Market value", np.nan)

    score = float(pd.to_numeric(row.get("Role Fit Score", 0.0), errors="coerce") or 0.0)
    match_pct = max(0, min(100, int(round(score))))

    avatar = resolve_player_photo(player, team, league)
    crest = resolve_team_crest(team, league)

    crest_html = f"<img class='crest' src='{crest}' />" if crest else ""

    st.markdown(
        f"""
<div class="tile">
  <div class="match">{match_pct}%<small>Match</small></div>
  <div class="row">
    <img class="avatar" src="{avatar}" />
    <div>
      <div class="name">{player}</div>
      <div class="teamline">{crest_html}<span>{team} · {league}</span></div>
      <div class="meta">
        <span class="chip">{pos}</span>
        <span class="chip">Age {_fmt_int(age)}</span>
        <span class="chip">{_fmt_int(minutes)} min</span>
        <span class="chip">MV {_fmt_money(mv)}</span>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Streamlit-safe “click”: expander per card
    with st.expander("View details", expanded=False):
        # role pills
        pills = top_role_pills(row, pill_cols, topk=3)
        if pills:
            pills_html = "<div class='pills'>" + "".join(
                [f"<div class='pill'>{n}<span>{v:02d}</span></div>" for n, v in pills]
            ) + "</div>"
            st.markdown(pills_html, unsafe_allow_html=True)

        # full metrics block
        st.markdown(build_metrics_html(row), unsafe_allow_html=True)

        if show_override:
            st.markdown("---")
            st.caption("Optional photo override stored in session only.")
            key_id = f"{player}|||{team}|||{league}"
            default_url = st.session_state.get("photo_map", {}).get(key_id, "")
            url = st.text_input(
                "FotMob player image URL (…/playerimages/{id}.png)",
                value=default_url,
                key=f"url_{card_key}",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save override", key=f"save_{card_key}"):
                    val = (url or "").strip()
                    if not val:
                        st.error("Paste a URL first.")
                    elif not (val.startswith("http://") or val.startswith("https://")):
                        st.error("URL must start with http:// or https://")
                    else:
                        st.session_state["photo_map"][key_id] = val
                        st.success("Saved.")
                        st.rerun()
            with c2:
                if st.button("Clear override", key=f"clear_{card_key}"):
                    st.session_state["photo_map"].pop(key_id, None)
                    st.info("Cleared.")
                    st.rerun()


def render_tiles_tab(ranked: pd.DataFrame, role_key: str, role_title: str):
    df_enriched, pill_cols, _ = enrich_for_display(ranked, role_key)
    df_view = df_enriched.head(int(top_n)).copy()
    if df_view.empty:
        st.info("No matches.")
        return

    # grid wrapper
    st.markdown("<div class='tiles'>", unsafe_allow_html=True)
    for i, (_, row) in enumerate(df_view.iterrows()):
        render_pro_card(
            row=row,
            pill_cols=pill_cols,
            role_key=role_key,
            card_key=f"{role_key}_{role_title}_{i}",
            show_override=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_template_players_tab(tmpl_src: pd.DataFrame, role_key: str, role_title: str):
    if tmpl_src is None or tmpl_src.empty:
        st.info("No eligible template players (check minutes threshold / team selection).")
        return

    # Build percentiles & pills within template set (so their cards have pills too)
    tmpl_src2 = tmpl_src.copy()

    # For template view, we might not have Role Fit Score; create a neutral placeholder
    if "Role Fit Score" not in tmpl_src2.columns:
        tmpl_src2["Role Fit Score"] = 0.0

    tmpl_enriched, pill_cols, _ = enrich_for_display(tmpl_src2, role_key)
    tmpl_enriched["Minutes played"] = pd.to_numeric(tmpl_enriched.get("Minutes played"), errors="coerce")
    tmpl_enriched = tmpl_enriched.sort_values("Minutes played", ascending=False)

    st.caption("Template players are sorted by minutes played.")
    st.markdown("<div class='tiles'>", unsafe_allow_html=True)
    for i, (_, row) in enumerate(tmpl_enriched.iterrows()):
        render_pro_card(
            row=row,
            pill_cols=pill_cols,
            role_key=role_key,
            card_key=f"tmpl_{role_key}_{role_title}_{i}",
            show_override=False,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ========================= ROLE TAB WRAPPER (ONLY 2 SUBTABS) =========================
def role_view(role_key: str, compute_fn, role_title: str):
    """
    compute_fn must return: (ranked_df, title_str, template_players_df)
    - ranked_df must include Role Fit Score already (from your _score_block)
    """
    ranked, title, tmpl_src = compute_fn()

    t1, t2 = st.tabs(["Tiles", "Template players"])
    with t1:
        render_tiles_tab(ranked, role_key, title)
    with t2:
        render_template_players_tab(tmpl_src, role_key, title)


# ========================= MAIN TABS =========================
tabs = st.tabs(["Strikers", "Attackers", "Central Midfield", "Fullbacks", "Center Backs"])

with tabs[0]:
    role_view("ST", compute_strikers, "Strikers")

with tabs[1]:
    att_choice = st.selectbox(
        "Attacker subgroup",
        ["All", "Right Wingers", "Left Wingers", "Attacking Midfielders"],
        index=0,
        key="att_role_choice",
    )
    role_view("ATT", lambda: compute_attackers(att_choice), f"Attackers ({att_choice})")

with tabs[2]:
    role_view("CM", compute_central_mid, "Central Midfield")

with tabs[3]:
    fb_choice = st.selectbox("Fullback side", ["All", "Right Backs", "Left Backs"], index=0, key="fb_role_choice")
    role_view("FB", lambda: compute_fullbacks(fb_choice), f"Fullbacks ({fb_choice})")

with tabs[4]:
    role_view("CB", compute_center_backs, "Center Backs")

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
        teams_selected = st.multiselect("Filter teams", teams, default=teams, key=f"secB_teams_{role_key}")
    with fl2:
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


































































