# --- PART 1 (DROP-IN REPLACEMENT WITH DATASET PICKER) ---

import io, math, uuid, re, time
from pathlib import Path
from urllib.parse import quote
from typing import List, Tuple  # Py3.8+ friendly

import numpy as np
import pandas as pd
import streamlit as st
from numpy.linalg import norm
from numpy import exp

# ---------- Page ----------
st.set_page_config(page_title="Club Scouting — Tiles", layout="wide")
st.title("🔎 Advanced Club Scouting — Tiles View")
st.caption(
    "Club Selection → Role Template matching with glossy tiles per role. "
    "Each tab computes its own Fit %. Dropdown per tile lets you paste a custom image URL to override the photo."
)

# ======================== DATA LOADER (picker + upload) ========================
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def _candidate_dirs() -> List[Path]:
    """Directories to search for WORLD*.csv files."""
    dirs: List[Path] = [Path.cwd()]
    # common project layouts (e.g., running from /pages)
    try:
        dirs.append(Path.cwd().parent)
    except Exception:
        pass
    # include script directory and its parent if __file__ exists
    try:
        here = Path(__file__).resolve().parent
        dirs.extend([here, here.parent])
    except Exception:
        pass
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for d in dirs:
        rp = d.resolve()
        if rp not in seen:
            seen.add(rp); uniq.append(rp)
    return uniq

def _find_world_csvs() -> List[Path]:
    files: List[Path] = []
    for base in _candidate_dirs():
        files.extend(sorted(base.glob("WORLD*.csv")))
    # unique by resolved path, preserve order
    seen, uniq = set(), []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp); uniq.append(rp)
    return uniq

def _label_for(p: Path) -> str:
    """Human-friendly label that disambiguates duplicates by parent folder."""
    parent_hint = p.parent.name or str(p.parent)
    return f"{p.name} — {parent_hint}/"

def pick_or_upload_world_csv() -> Tuple[pd.DataFrame, str]:
    """
    UI: choose one of the detected WORLD*.csv files or upload your own.
    Returns (df, dataset_name_for_state)
    """
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

    # Map label back to the concrete Path (robust if duplicate names exist)
    idx = labels.index(sel) - 1  # shift for "Upload a CSV…"
    chosen_path = found[idx]
    df_disk = _read_csv_from_path(str(chosen_path))
    # Use resolved name for stable session checks
    return df_disk, chosen_path.resolve().name

# ---- Load dataset now (df used by the rest of the app) ----
df, DATASET_NAME = pick_or_upload_world_csv()

# Optional: reset any dataset-scoped widgets/state on change
if st.session_state.get("_active_dataset_name") != DATASET_NAME:
    for key in [
        # add any widget keys you want cleared when switching datasets
        "att_role_choice", "fb_role_choice",
        "st_tmpl_pick", "att_tmpl_pick", "cm_tmpl_pick", "fb_tmpl_pick", "cb_tmpl_pick",
        "photo_map",
    ]:
        st.session_state.pop(key, None)
    st.session_state["_active_dataset_name"] = DATASET_NAME

# ======================== leagues & strengths ========================
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
'Iceland 1.':46.08,'Norway 2.':45.88,'Sweden 2.':45.69,'North Macedonia 1.':44.71,'Turkey 2.':44.51,
'Korea 2.':43.53,'Czech 2.':43.33,'Brazil 3.':43.14,'Lithuania 1.':42.35,'Netherlands 2.':42.16,
'Malta 1.':41.96,'Italy 3.':45,'Denmark 2.':40.39,'Moldova 1.':40.39,'USA 2.':40.00,
'Latvia 1.':40.00,'Montenegro 1.':39.80,'Scotland 2.':38.63,'Canada 1.':38.24,'Austria 2.':38.24,
'Israel 2.':38.04,'England 7.':37.25,'Germany 4.':35.29,'Portugal 3.':35.29,'England 5.':33.33,
'Estonia 1.':40,'England 9.':31.37,'Northern Ireland 1.':30.98,'Serbia 2.':30.39,'Denmark 3.':29.41,
'Sweden 3.':29.41,'Slovenia 2.':28.82,'Slovakia 2.':28.24,'Greece 2.':27.06,'Wales 1.':26.67,
'USA 3.':22.55,'Scotland 3.':20.00,'England 6.':16.08,'England 8.':15.69,'England 10.':3.92,
'Estonia 2.':3

}

# ======================== sidebar: candidate pool filters (generic) ========================
with st.sidebar:
    st.header("Candidate Pool Filters (affect MATCHES)")
    c1, c2, c3 = st.columns(3)
    use_top5 = c1.checkbox("Top-5", False)
    use_top20 = c2.checkbox("Top-20", False)
    use_efl = c3.checkbox("EFL", False)

    seed = set()
    if use_top5: seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top20: seed |= PRESET_LEAGUES["Top 20 Europe"]
    if use_efl: seed |= PRESET_LEAGUES["EFL (England 2–4)"]

    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df["League"].dropna().unique()))
    default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES
    leagues_sel = st.multiselect("Leagues in candidate pool", leagues_avail, default=default_leagues)

    # NOTE: per-role position filters are inside each tab
    min_minutes, max_minutes = st.slider("Minutes played (pool)", 0, 6000, (750, 6000))

    age_min_data = int(np.nanmin(pd.to_numeric(df["Age"], errors="coerce"))) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(pd.to_numeric(df["Age"], errors="coerce"))) if df["Age"].notna().any() else 50
    min_age, max_age = st.slider("Age (pool)", age_min_data, age_max_data, (16, 50))

    st.markdown("**Market value (€)**")
    mv_col = pd.to_numeric(df["Market value"], errors="coerce")
    mv_max_raw = int(np.nanmax(mv_col)) if mv_col.notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = max(1, mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, min(max_m, 10)))
        pool_min_value, pool_max_value = mv_min_m*1_000_000, mv_max_m*1_000_000
    else:
        pool_min_value, pool_max_value = st.slider("Range (€)", 0, mv_cap, (0, min(mv_cap, 10_000_000)), step=100_000)

    min_strength, max_strength = st.slider(
        "League quality (strength)", 0, 101, (0, 101),
        help="Filter candidates by league strength (0–100)."
    )

    # Role score settings
    st.subheader("Role Score")
    decay_rate = st.slider("Exp. decay (↑=stricter)", 0.5, 10.0, 5.0, 0.5)
    use_league_weighting = st.checkbox("Blend in league strength (β)", value=True)
    beta = st.slider("β (0–1)", 0.0, 1.0, 0.40, 0.05, help="0=distance only, 1=league strength only")

    use_league_mismatch = st.checkbox(
        "Penalise league mismatch inside distance (α, p)", value=True,
        help="Adds a distance-penalty based on |candidate league − template league|."
    )
    alpha = st.slider("League mismatch weight α", 0.0, 5.0, 1.20, 0.05)
    p_exp = st.slider("League mismatch exponent p", 1.0, 3.0, 1.50, 0.10)
    penalty_mode = st.selectbox("Penalty combine mode", ["Additive (stronger)", "Quadrature (gentler)"], index=0)

    top_n = st.number_input("How many tiles (Top N)", 5, 200, 20, 5)
    DEBUG_PHOTOS = st.checkbox("Debug player photos", False)

# ======================== helpers: build base candidate pool (no position) ========================
def build_base_pool():
    p = df.copy()
    p = p[p["League"].isin(leagues_sel)]

    # numeric coercions
    for c in ["Minutes played","Age","Market value","Goals"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")

    p = p[p["Minutes played"].between(min_minutes, max_minutes)]
    p = p[p["Age"].between(min_age, max_age)]
    p = p[p["Market value"].between(pool_min_value, pool_max_value)]

    p["League Strength"] = p["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
    p = p[(p["League Strength"] >= float(min_strength)) & (p["League Strength"] <= float(max_strength))]
    return p

# ======================== Club Selection (re-used by all tabs) ========================
st.markdown("---")
st.header("🎯 Club Selection (template source)")

template_league_list = sorted([str(x) for x in df["League"].dropna().unique()])
template_league = st.selectbox("Template league (scopes team list)", template_league_list)
templ_teams_all = sorted(df.loc[df["League"].astype(str) == template_league, "Team"].dropna().astype(str).unique())
search = st.text_input("Search team (filters list)", "")
templ_teams = [t for t in templ_teams_all if search.lower() in t.lower()] or templ_teams_all
template_team = st.selectbox("Template team", templ_teams)

min_minutes_template = st.slider("Minimum minutes for template players", 0, 6000, 1000, 100)
use_single_template_player = st.checkbox("Use single player only (else avg of role at team)", False)
template_strength = float(LEAGUE_STRENGTHS.get(template_league, 0.0))

# ======================== shared: pick template rows by role predicate ========================
def _template_rows_for_role(pos_predicate):
    src = df[
        (df["League"].astype(str) == template_league)
        & (df["Team"].astype(str) == template_team)
        & (df["Position"].apply(lambda p: pos_predicate(str(p))))
    ].copy()
    src["Minutes played"] = pd.to_numeric(src["Minutes played"], errors="coerce")
    src = src[src["Minutes played"] >= min_minutes_template]
    return src

# ======================== scoring helper used by all roles ========================
def _score_block(df_with_baseDist: pd.DataFrame) -> pd.DataFrame:
    """Given df with ['BaseDist','League'] columns, compute Role Fit Score with options."""

    # league mismatch inside distance
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

    # exp-decay base score
    dmin, dmax = float(df_with_baseDist["Role Fit Distance"].min()), float(df_with_baseDist["Role Fit Distance"].max())
    rng = dmax - dmin
    if rng <= 1e-12:
        base_score = pd.Series(100.0, index=df_with_baseDist.index)
    else:
        base_score = 100.0 * exp(-decay_rate * ((df_with_baseDist["Role Fit Distance"] - dmin) / rng))

    league_part = df_with_baseDist["League"].map(LEAGUE_STRENGTHS).fillna(0.0) if use_league_weighting else 0.0
    df_with_baseDist["Role Fit Score"] = (1.0 - beta) * base_score + beta * league_part
    return df_with_baseDist.sort_values("Role Fit Score", ascending=False).reset_index(drop=True)

# ======================== small util: safe verticality ========================
def _safe_verticality(forward_per90, passes_per90):
    f = pd.to_numeric(forward_per90, errors="coerce")
    p = pd.to_numeric(passes_per90, errors="coerce")
    p = p.replace(0, np.nan)
    out = f / p
    return out.fillna(0.0)

# ======================== ROLE CALCULATORS (updated) ========================
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
        st.stop()

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

    # hard caps
    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 26)
                & (pd.to_numeric(pool["Market value"], errors="coerce") <= 10_000_000)
                & (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 1000)]

    for c in feats: pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Opportunities"]      = 0.7*pool['Touches in box per 90'] + 0.3*pool['xG per 90']
    pool["Ball Carrying"]      = 0.65*pool['Dribbles per 90'] + 0.35*pool['Progressive runs per 90']
    pool["Aerial Requirement"] = pool['Aerial duels per 90'] * pool['Aerial duels won, %'] / 100.0
    pool["Passing Volume"]     = pool['Passes per 90']
    pool["Goal Output"]        = pool['Non-penalty goals per 90']
    pool["Retention"]          = pool['Accurate passes, %']

    cols = ["Opportunities","Ball Carrying","Aerial Requirement","Passing Volume","Goal Output","Retention"]
    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    return ranked, pool, "Strikers (CF)", tmpl_src

def compute_attackers(role_choice: str):
    feats = [
        'Accurate passes, %','xG per 90','Non-penalty goals per 90','Touches in box per 90',
        'xA per 90','Passes to penalty area per 90','Passes per 90',
        'Progressive passes per 90','Passes to final third per 90',
        'Dribbles per 90','Progressive runs per 90'
    ]

    # ----- Position filter with ROLE_CHOICE inside the tab -----
    def pos_ok(p):
        s = str(p).upper().strip()
        tokens = [t for t in re.split(r"[,/;]\s*|\s+", s) if t]
        if not tokens: return False
        t0 = tokens[0]
        if role_choice == "All":
            allowed = {"RW","RWF","RAMF","LW","LWF","LAMF","AMF"}
            return t0 in allowed
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

    # caps for attackers
    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 23)
                & (pd.to_numeric(pool["Market value"], errors="coerce") <= 5_000_000)
                & (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 900)]

    for c in feats: pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Retention Style"]   = pool['Accurate passes, %']
    pool["Goal Threat"]       = 0.4*pool['xG per 90'] + 0.4*pool['Non-penalty goals per 90'] + 0.2*pool['Touches in box per 90']
    pool["Creativity Threat"] = 0.65*pool['xA per 90'] + 0.35*pool['Passes to penalty area per 90']
    pool["Passing Volume"]    = pool['Passes per 90']
    pool["Deeper Playmaking"] = 0.5*pool['Progressive passes per 90'] + 0.5*pool['Passes to final third per 90']
    pool["Ball Carrying"]     = 0.6*pool['Dribbles per 90'] + 0.4*pool['Progressive runs per 90']

    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    return ranked, pool, "Attackers (Wingers/AM)", tmpl_src, pos_ok

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
        chosen = st.selectbox("Template player (Central Midfield)", ["— Select —"] + players, index=0, key="cm_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]

    if tmpl_src.empty:
        st.error("No central midfielders found for template conditions.")
        st.stop()

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

    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 32)
                & (pd.to_numeric(pool["Market value"], errors="coerce") <= 5_000_000)
                & (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 1000)]

    for c in feats: pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Pass Verticality"]    = _safe_verticality(pool['Forward passes per 90'], pool['Passes per 90'])
    pool["Progression Volume"]  = pool['Progressive passes per 90'] + pool['Progressive runs per 90']
    pool["Attacking Contribution"] = pool['Touches in box per 90'] + pool['Shots per 90']
    pool["Defensive Volume"]    = pool['Defensive duels per 90']
    pool["Interception Volume"] = pool['PAdj Interceptions']
    pool["Retention"]           = pool['Accurate passes, %']

    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    return ranked, pool, "Central Midfield", tmpl_src

def compute_fullbacks(role_choice: str):
    feats = [
        'Passes per 90','Forward passes per 90',
        'Progressive passes per 90','Progressive runs per 90',
        'Defensive duels per 90','PAdj Interceptions','Aerial duels per 90',
        'xA per 90','Crosses per 90','Touches in box per 90',
        'Shots per 90','Passes to penalty area per 90',
        'Accurate passes, %'
    ]

    # ----- Position filter with ROLE_CHOICE inside the tab -----
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
        chosen = st.selectbox("Template player (Fullbacks)", ["— Select —"] + players, index=0, key="fb_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]

    if tmpl_src.empty:
        st.error("No fullbacks found for template conditions.")
        st.stop()

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

    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 30)
                & (pd.to_numeric(pool["Market value"], errors="coerce") <= 10_000_000)
                & (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 1000)]

    for c in feats: pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Pass Verticality"]     = _safe_verticality(pool['Forward passes per 90'], pool['Passes per 90'])
    pool["Progression Volume"]   = pool['Progressive passes per 90'] + pool['Progressive runs per 90']
    pool["Attacking Contribution"]= 0.4*pool['xA per 90'] + 0.2*pool['Crosses per 90'] + 0.2*pool['Touches in box per 90'] + 0.1*pool['Shots per 90'] + 0.1*pool['Passes to penalty area per 90']
    pool["Defensive Volume"]     = 0.5*pool['Defensive duels per 90'] + 0.3*pool['PAdj Interceptions'] + 0.2*pool['Aerial duels per 90']
    pool["Retention"]            = pool['Accurate passes, %']

    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    return ranked, pool, "Fullbacks", tmpl_src, pos_ok

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
        chosen = st.selectbox("Template player (Center Backs)", ["— Select —"] + players, index=0, key="cb_tmpl_pick")
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

    cols = ["Aerial duels per 90","Defensive duels per 90","Positional Demand","Passing Volume","Passing Verticality","Progression Volume"]
    tmpl_vec = f[cols].mean()

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]

    # CB caps
    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 22)
                & (pd.to_numeric(pool["Market value"], errors="coerce") <= 10_000_000)
                & (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 500)]

    for c in feats: pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Passing Verticality"] = _safe_verticality(pool['Forward passes per 90'], pool['Passes per 90'])
    pool["Passing Volume"]      = pool['Passes per 90']
    pool["Positional Demand"]   = pool['PAdj Interceptions'] + pool['Shots blocked per 90']
    pool["Progression Volume"]  = pool['Progressive passes per 90'] + pool['Progressive runs per 90']

    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    return ranked, pool, "Center Backs", tmpl_src

    # --- PART 2 ---

# ---------- Style for tiles ----------
st.markdown(
    """
<style>
:root { --bg:#0f1115; --card:#161a22; --muted:#a8b3cf; --soft:#202633; }
.block-container { padding-top:.8rem; }
body{ background:var(--bg); font-family: system-ui,-apple-system,'Segoe UI','Segoe UI Emoji',Roboto,Helvetica,Arial,sans-serif;}
.wrap{ display:flex; justify-content:center; }
.player-card{ width:min(980px,96%); display:grid; grid-template-columns:112px 1fr 100px; gap:14px; align-items:start; background:var(--card); border:1px solid #252b3a; border-radius:18px; padding:16px; box-shadow: 0 2px 14px rgba(0,0,0,.25); }
.avatar{ width:112px; height:112px; border-radius:12px; background-color:#0b0d12; background-size:cover; background-position:center; border:1px solid #2a3145; }
.leftcol{ display:flex; flex-direction:column; align-items:center; gap:8px; }
.name{ font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
.sub{ color:#a8b3cf; font-size:15px; }
.pill{ padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
.row{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:4px 0; }
.chip{ background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:10px; font-size:13px; line-height:18px; }
.pos{ color:#eaf0ff; font-weight:700; padding:4px 10px; border-radius:10px; font-size:12px; border:1px solid rgba(255,255,255,.08); }
.teamline{ color:#e6ebff; font-size:15px; font-weight:400; margin-top:2px; }
.fit{ color:#94f0c8; font-weight:900; font-size:28px; text-align:right; }
.fit small{ display:block; color:#9fb3c6; font-weight:600; font-size:12px; margin-top:4px; }
.divider{ height:12px; }
.metric-section{ background:#121621; border:1px solid #242b3b; border-radius:14px; padding:10px 12px; }
.m-title{ color:#e8ecff; font-weight:800; letter-spacing:.02em; margin:4px 0 10px 0; font-size:20px; text-transform:uppercase; }
.m-row{ display:flex; justify-content:space-between; align-items:center; padding:8px 8px; border-radius:10px; }
.m-row + .m-row{ margin-top:6px; }
.m-label{ color:#c9d3f2; font-size:16px; }
.m-right{ display:flex; align-items:center; gap:8px; }
.m-badge{ min-width:40px; text-align:center; padding:2px 10px; border-radius:8px; font-weight:800; font-size:18px; color:#0b0d12; border:1px solid rgba(0,0,0,.15); }
.metrics-grid{ display:grid; grid-template-columns:1fr; gap:12px; }
@media (min-width: 980px){ .metrics-grid{ grid-template-columns:repeat(3, 1fr); } }
</style>
""",
    unsafe_allow_html=True,
)

PALETTE=[(0,(208,2,27)),(50,(245,166,35)),(65,(248,231,28)),(75,(126,211,33)),(85,(65,117,5)),(100,(40,90,4))]
def _lerp(a,b,t): return tuple(int(round(a[i]+(b[i]-a[i])*t)) for i in range(3))
def rating_color(v:float)->str:
    # HTML/CSS rgb(...)
    v=max(0.0,min(100.0,float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v<=x1:
            t=0 if x1==x0 else (v-x0)/(x1-x0); r,g,b=_lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b=PALETTE[-1][1]; return f"rgb({r},{g},{b})"

def rating_color_hex(v: float) -> str:
    # Matplotlib-friendly HEX
    v = max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v<=x1:
            t=0 if x1==x0 else (v-x0)/(x1-x0)
            r,g,b=_lerp(c0,c1,t)
            return f"#{r:02x}{g:02x}{b:02x}"
    r,g,b=PALETTE[-1][1]
    return f"#{r:02x}{g:02x}{b:02x}"

# ====================== PlaymakerStats image resolver ======================
_PS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.playmakerstats.com/",
}

@st.cache_data(show_spinner=False, ttl=24*3600)
def _http_get_text(url: str, retries: int = 1, timeout: int = 12) -> str:
    import requests
    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=_PS_HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.6); continue
        except Exception:
            time.sleep(0.25); continue
    return ""

def _extract_og_image(html: str):
    m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.I)
    return m.group(1) if m else None

@st.cache_data(show_spinner=False, ttl=24*3600)
def playmakerstats_image_by_name_team(name: str, team: str):
    q = f"{name} {team}".strip()
    # PlaymakerStats (EN)
    search_url = f"https://www.playmakerstats.com/search.php?search={quote(q)}"
    html = _http_get_text(search_url, retries=1)
    if html:
        links = re.findall(r'href=["\'](/(?:player|jogador)\.php\?id=\d+[^"\']*)', html, flags=re.I)
        if links:
            p_html = _http_get_text("https://www.playmakerstats.com" + links[0], retries=1)
            if p_html:
                img = _extract_og_image(p_html)
                if img: return img
    # zerozero.pt fallback
    search_url2 = f"https://www.zerozero.pt/procura.php?search={quote(q)}"
    html2 = _http_get_text(search_url2, retries=1)
    if html2:
        links2 = re.findall(r'href=["\'](/jogador\.php\?id=\d+[^"\']*)', html2, flags=re.I)
        if links2:
            p_html2 = _http_get_text("https://www.zerozero.pt" + links2[0], retries=1)
            if p_html2:
                img2 = _extract_og_image(p_html2)
                if img2: return img2
    return None

PLACEHOLDER_IMG = "https://i.redd.it/43axcjdu59nd1.jpeg"
if "photo_map" not in st.session_state:
    st.session_state["photo_map"] = {}

# ---------- small helper: table of template players used (fixes issue #1) ----------
def render_template_players_used(role_name: str, tmpl_src: pd.DataFrame):
    showcols = [c for c in ["Player","Minutes played","Position","League","Team"] if c in tmpl_src.columns]
    st.subheader(f"🧩 Players used for {role_name} Role Template")
    if tmpl_src.empty or not showcols:
        st.info("No eligible template players for the selected team/filters.")
        return
    st.dataframe(
        tmpl_src[showcols].sort_values("Minutes played", ascending=False),
        use_container_width=True
    )

# ---------- shared tile+FeatureZ renderer ----------
def render_tiles_and_featureZ(ranked: pd.DataFrame, df_pool_role: pd.DataFrame, role_title: str):
    st.markdown("---")
    st.header(f"🏅 Top Role Matches — Tiles · {role_title}")
    st.caption(f"Showing Top N = **{int(top_n)}**")

    # Helpers for dropdown metrics (percentiles vs pool)
    def pct_series_for_player(player_row: pd.Series, col: str, within_df: pd.DataFrame) -> float:
        vals = pd.to_numeric(within_df[col], errors="coerce").dropna()
        if vals.empty: return 0.0
        v = pd.to_numeric(player_row.get(col), errors="coerce")
        if pd.isna(v): return 0.0
        return float((vals <= v).mean() * 100.0)

    st.write("")  # tiny spacer

    for idx, row in ranked.head(int(top_n)).iterrows():
        name   = str(row.get("Player",""))
        team   = str(row.get("Team",""))
        league = str(row.get("League",""))
        pos    = str(row.get("Position",""))
        age    = int(row.get("Age",0)) if not pd.isna(row.get("Age",np.nan)) else 0
        minutes= int(row.get("Minutes played",0)) if not pd.isna(row.get("Minutes played",np.nan)) else 0
        fit    = float(row.get("Role Fit Score",0.0))
        fit_pct= max(0, min(100, int(round(fit))))

        key_id = f"{name}|||{team}|||{league}"
        avatar_url = playmakerstats_image_by_name_team(name, team) or PLACEHOLDER_IMG
        override_url = st.session_state.get("photo_map", {}).get(key_id, "")
        if override_url:
            avatar_url = override_url + f"?t={int(time.time())}"

        if DEBUG_PHOTOS:
            st.write(f"PHOTO DEBUG → '{name}' / '{team}' → {avatar_url}")

        ov_style = f"background:{rating_color(fit_pct)};"
        codes = [c for c in re.split(r"[,/; ]+", pos.strip().upper()) if c]
        chips_html = " ".join(f"<span class='pos'>{c}</span>" for c in dict.fromkeys(codes))

        st.markdown(
            f"""
<div class='wrap'>
  <div class='player-card'>
    <div class='leftcol'>
      <div class='avatar' style="background-image:url('{avatar_url}');"></div>
      <div class='row'><span class='chip'>{age}y</span><span class='chip'>{minutes}m</span></div>
    </div>
    <div>
      <div class='name'>{name}</div>
      <div class='row' style='align-items:center;'>
        <span class='pill' style='{ov_style}'>{fit_pct}</span>
        <span class='sub'>Overall Fit</span>
      </div>
      <div class='row'>{chips_html}</div>
      <div class='teamline'>{team} · {league}</div>
    </div>
    <div class='fit'>{fit_pct}%<small>Fit</small></div>
  </div>
</div>
<div class='divider'></div>
            """,
            unsafe_allow_html=True
        )

        # === dropdown: metrics + image URL override ===
        with st.expander("▼ Show individual metrics / Set photo override"):
            ATTACKING = []
            for lab, met in [
                ("Goals: Non-Penalty","Non-penalty goals per 90"),
                ("xG","xG per 90"),
                ("Shots","Shots per 90"),
                ("Header Goals","Head goals per 90"),
                ("Expected Assists","xA per 90"),
                ("Progressive Runs","Progressive runs per 90"),
                ("Touches in Opposition Box","Touches in box per 90"),
            ]:
                if met in df_pool_role.columns:
                    ATTACKING.append((lab, pct_series_for_player(row, met, df_pool_role)))

            DEFENSIVE = []
            for lab, met in [
                ("Aerial Duels","Aerial duels per 90"),
                ("Aerial Duel Success %","Aerial duels won, %"),
                ("PAdj. Interceptions","PAdj Interceptions"),
                ("Defensive Duels","Defensive duels per 90"),
                ("Defensive Duel Success %","Defensive duels won, %"),
            ]:
                if met in df_pool_role.columns:
                    DEFENSIVE.append((lab, pct_series_for_player(row, met, df_pool_role)))

            POSSESSION = []
            for lab, met in [
                ("Dribbles","Dribbles per 90"),
                ("Dribbling Success %","Successful dribbles, %"),
                ("Key Passes","Key passes per 90"),
                ("Passes","Passes per 90"),
                ("Passing Accuracy %","Accurate passes, %"),
                ("Passes to Penalty Area","Passes to penalty area per 90"),
                ("Passes to Penalty Area %","Accurate passes to penalty area, %"),
                ("Deep Completions","Deep completions per 90"),
                ("Smart Passes","Smart passes per 90"),
            ]:
                if met in df_pool_role.columns:
                    POSSESSION.append((lab, pct_series_for_player(row, met, df_pool_role)))

            def section_html(title: str, items):
                rows=[]
                for lab, pct in items:
                    pct_i = int(round(max(0.0, min(100.0, float(pct)))))
                    rows.append(
                        f"<div class='m-row'><div class='m-label'>{lab}</div>"
                        f"<div class='m-right'><span class='m-badge' style='background:{rating_color(pct_i)}'>{pct_i}</span></div></div>"
                    )
                return f"<div class='metric-section'><div class='m-title'>{title}</div>{''.join(rows)}</div>"

            col_html = (
                "<div class='metrics-grid'>"
                + section_html('ATTACKING', ATTACKING)
                + section_html('DEFENSIVE', DEFENSIVE)
                + section_html('POSSESSION', POSSESSION)
                + "</div>"
            )
            st.markdown(col_html, unsafe_allow_html=True)

            # --- Custom image URL override ---
            img_key = f"imgurl_{key_id}"
            default_url = st.session_state.get("photo_map", {}).get(key_id, "")
            _ = st.text_input(
                "Custom image URL (override avatar — e.g., https://images.fotmob.com/image_resources/playerimages/1199383.png)",
                value=default_url, key=img_key
            )
            col_a, col_b = st.columns([1, 3])
            with col_a:
                if st.button("Apply to this player", key=f"apply_{key_id}"):
                    val = (st.session_state.get(img_key, "") or "").strip()
                    if not val:
                        st.error("Please paste an image URL.")
                    elif not (val.startswith("http://") or val.startswith("https://")):
                        st.error("Image URL must start with http:// or https://")
                    else:
                        st.session_state.setdefault("photo_map", {})[key_id] = val
                        st.success("Saved!")
                        try: st.rerun()
                        except Exception: st.experimental_rerun()
            with col_b:
                if st.button("Clear override", key=f"clear_{key_id}"):
                    st.session_state["photo_map"].pop(key_id, None)
                    st.info("Cleared.")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()

    # ======================== Feature Z (stability + Matplotlib color fixes) ========================
    st.markdown("---")
    st.header(f"Advanced Individual Player Analysis (Feature Z) · {role_title}")

    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation
    from matplotlib.font_manager import FontProperties
    from PIL import Image  # noqa: F401

    left, right = st.columns([2,2])

    options_ranked = ranked["Player"].astype(str).head(int(top_n)).tolist()
    any_pool = left.checkbox("Pick from entire candidate pool (not just Top N)", value=False, key=f"fz_pool_toggle_{role_title}")
    options = sorted(df_pool_role["Player"].dropna().astype(str).unique()) if any_pool else options_ranked
    if not options:
        st.info("No players available for Feature Z. Adjust filters.")
        return
    player_sel = left.selectbox("Choose player for Feature Z", options, index=0, key=f"fz_pick_{role_title}")

    show_height = right.checkbox("Show height in info row", value=True, key=f"fz_height_{role_title}")
    foot_override_on = right.checkbox("Edit foot value", value=False, key=f"fz_foot_on_{role_title}")
    foot_override_text = right.text_input("Foot (e.g., Left)", value="", disabled=not foot_override_on, key=f"fz_foot_txt_{role_title}")
    name_override_on = right.checkbox("Edit display name", value=False, key=f"fz_name_on_{role_title}")
    name_override = right.text_input("Display name", "", disabled=not name_override_on, key=f"fz_name_txt_{role_title}")
    footer_caption_text = right.text_input("Footer caption", "Percentile Rank", key=f"fz_footer_{role_title}")

    player_row = df_pool_role[df_pool_role["Player"].astype(str) == str(player_sel)].head(1)
    if player_row.empty:
        st.info("Pick a player above."); return

    def _safe_get(sr, key, default="—"):
        try:
            v = sr.iloc[0].get(key, default)
            s = "" if v is None else str(v)
            return default if s.strip() == "" else s
        except Exception:
            return default

    def pct_series(col: str) -> float:
        vals = pd.to_numeric(df_pool_role[col], errors="coerce").dropna()
        if vals.empty: return np.nan
        v = pd.to_numeric(player_row.iloc[0][col], errors="coerce")
        if pd.isna(v): return np.nan
        return float((vals <= v).mean() * 100.0)

    def val_of(col: str):
        v = player_row.iloc[0].get(col)
        if pd.isna(v): return (np.nan, "—")
        if isinstance(v, (int,float,np.floating)):
            return (float(v), f"{float(v):.0f}%" if "%" in col else f"{float(v):.2f}")
        return (v, str(v))

    pos = _safe_get(player_row, "Position", "—")
    name_ = _safe_get(player_row, "Player", _safe_get(player_row, "Name", ""))
    if name_override_on and name_override.strip():
        name_ = name_override.strip()
    team = _safe_get(player_row, "Team", "")
    age_raw = _safe_get(player_row, "Age", "")
    try: age = f"{float(age_raw):.0f}"
    except Exception: age = age_raw
    games   = _safe_get(player_row, "Matches played", _safe_get(player_row, "Games", _safe_get(player_row, "Apps", "—")))
    minutes = _safe_get(player_row, "Minutes", _safe_get(player_row, "Minutes played", "—"))
    goals   = _safe_get(player_row, "Goals", "—")
    assists = _safe_get(player_row, "Assists", "—")
    foot    = _safe_get(player_row, "Foot", _safe_get(player_row, "Preferred Foot", "—"))
    foot_display = (foot_override_text.strip() if (foot_override_on and foot_override_text and foot_override_text.strip()) else foot)
    height_text = ""
    for col in ["Height","Height (ft)","Height ft","Height (cm)"]:
        v = _safe_get(player_row, col, "")
        if v and v != "—":
            height_text = str(v).strip(); break

    ATTACKING, DEFENSIVE, POSSESSION = [], [], []
    for lab, met in [
        ("Goals: Non-Penalty","Non-penalty goals per 90"),
        ("xG","xG per 90"),
        ("Shots","Shots per 90"),
        ("Header Goals","Head goals per 90"),
        ("Expected Assists","xA per 90"),
        ("Progressive Runs","Progressive runs per 90"),
        ("Touches in Opp. Box","Touches in box per 90"),
    ]:
        if met in df_pool_role.columns:
            ATTACKING.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

    for lab, met in [
        ("Aerial Duels","Aerial duels per 90"),
        ("Aerial Duel Success %","Aerial duels won, %"),
        ("PAdj. Interceptions","PAdj Interceptions"),
        ("Defensive Duels","Defensive duels per 90"),
        ("Defensive Duel Success %","Defensive duels won, %"),
    ]:
        if met in df_pool_role.columns:
            DEFENSIVE.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

    for lab, met in [
        ("Dribbles","Dribbles per 90"),
        ("Dribbling Success %","Successful dribbles, %"),
        ("Key Passes","Key passes per 90"),
        ("Passes","Passes per 90"),
        ("Passing Accuracy %","Accurate passes, %"),
        ("Passes to Penalty Area","Passes to penalty area per 90"),
        ("Passes to Penalty Area %","Accurate passes to penalty area, %"),
        ("Deep Completions","Deep completions per 90"),
        ("Smart Passes","Smart passes per 90"),
    ]:
        if met in df_pool_role.columns:
            POSSESSION.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

    sections = [("Attacking",ATTACKING),("Defensive",DEFENSIVE),("Possession",POSSESSION)]
    sections = [(t,lst) for t,lst in sections if lst]

    import matplotlib.pyplot as _plt_cleanup

    def _font_name_or_fallback(pref, fallback="DejaVu Sans"):
        from matplotlib import font_manager as fm
        installed = {f.name for f in fm.fontManager.ttflist}
        for n in pref:
            if n in installed:
                return n
        return fallback

    FONT_TITLE_FAMILY = _font_name_or_fallback(["Tableau Bold","Tableau Sans Bold","Tableau"])
    FONT_BOOK_FAMILY  = _font_name_or_fallback(["Tableau Book","Tableau Sans","Tableau"])

    TITLE_FP = FontProperties(family=FONT_TITLE_FAMILY, weight='bold', size=24)
    H2_FP    = FontProperties(family=FONT_TITLE_FAMILY, weight='semibold', size=20)
    LABEL_FP = FontProperties(family=FONT_BOOK_FAMILY, weight='medium', size=10)
    INFO_LABEL_FP= FontProperties(family=FONT_BOOK_FAMILY, weight='bold', size=10)
    INFO_VALUE_FP= FontProperties(family=FONT_BOOK_FAMILY, weight='regular', size=10)
    BAR_VALUE_FP = FontProperties(family=FONT_BOOK_FAMILY, weight='regular', size=8)
    TICK_FP  = FontProperties(family=FONT_BOOK_FAMILY, weight='medium', size=10)
    FOOTER_FP= FontProperties(family=FONT_BOOK_FAMILY, weight='medium', size=10)

    PAGE_BG = "#ebebeb"; AX_BG = "#f3f3f3"; TRACK="#d6d6d6"
    TITLE_C="#111111"; LABEL_C="#222222"; DIVIDER="#000000"
    ticks = np.arange(0,101,10)
    LEFT, RIGHT, TOP, BOT = 0.055, 0.030, 0.035, 0.07
    header_h, GAP = 0.045, 0.020
    gutter = 0.215
    BAR_FRAC = 0.92

    if not sections:
        st.info("No comparable metrics available for this role/player.")
        return

    fig_size = (11.8, 9.6); dpi = 120
    title_row_h = 0.125; header_block_h = title_row_h + 0.055
    fig = plt.figure(figsize=fig_size, dpi=dpi); fig.patch.set_facecolor(PAGE_BG)
    fig.text(LEFT, 1 - TOP - 0.010, f"{name_}\u2009|\u2009{team}", ha="left", va="top", color=TITLE_C, fontproperties=TITLE_FP)

    def draw_pairs_line(pairs_line, y):
        x = LEFT; renderer = fig.canvas.get_renderer()
        for i,(lab,val) in enumerate(pairs_line):
            t1 = fig.text(x, y, lab, ha="left", va="top", color=LABEL_C, fontproperties=INFO_LABEL_FP); fig.canvas.draw(); x += t1.get_window_extent(renderer).width / fig.bbox.width
            t2 = fig.text(x, y, str(val), ha="left", va="top", color=LABEL_C, fontproperties=INFO_VALUE_FP); fig.canvas.draw(); x += t2.get_window_extent(renderer).width / fig.bbox.width
            if i != len(pairs_line)-1:
                t3 = fig.text(x, y, " | ", ha="left", va="top", color="#555555", fontproperties=INFO_VALUE_FP); fig.canvas.draw(); x += t3.get_window_extent(renderer).width / fig.bbox.width

    row1 = [("Position: ",pos), ("Age: ",age), ("Height: ", height_text if (show_height and height_text) else "—")]
    row2 = [("Games: ",games), ("Goals: ",goals), ("Assists: ",assists)]
    row3 = [("Minutes: ",minutes), ("Foot: ",foot_display)]
    title_y = 1 - TOP - 0.010
    y1 = title_y - 0.055; y2 = y1 - 0.039; y3 = y2 - 0.039
    draw_pairs_line(row1, y1); draw_pairs_line(row2, y2); draw_pairs_line(row3, y3)

    fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT],[1 - TOP - (title_row_h+0.055) + 0.004]*2, transform=fig.transFigure, color=DIVIDER, lw=0.8, alpha=0.35))

    def draw_panel(panel_top, title, tuples, *, show_xticks=False, draw_bottom_divider=True):
        n = len(tuples)
        if n == 0: return panel_top

        total_rows = sum(len(lst) for _, lst in sections)
        rows_space_total = max(0.15, 1 - (TOP + BOT) - (title_row_h+0.055) - header_h*len(sections) - GAP*(len(sections)-1))
        row_slot = rows_space_total / max(total_rows,1)

        fig.text(LEFT, panel_top - 0.012, title, ha="left", va="top", color=TITLE_C, fontproperties=H2_FP)
        ax = fig.add_axes([LEFT + gutter, panel_top - header_h - n*row_slot, 1 - LEFT - RIGHT - gutter, n*row_slot])
        ax.set_facecolor(AX_BG); ax.set_xlim(0,100); ax.set_ylim(-0.5,n-0.5)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False, length=0)
        ax.tick_params(axis="y", left=False, labelleft=False, length=0)
        ax.set_yticks([]); ax.get_yaxis().set_visible(False)

        for i in range(n):
            ax.add_patch(plt.Rectangle((0, i-(BAR_FRAC/2)), 100, BAR_FRAC, facecolor=TRACK, edgecolor="none", linewidth=0, zorder=0.5))
        for gx in ticks:
            ax.vlines(gx, -0.5, n-0.5, colors=(0,0,0,0.16), linewidth=0.8, zorder=0.75)

        for i,(lab,pct,val_str) in enumerate(tuples[::-1]):
            y = i; bar_w = float(np.clip(pct,0,100))
            if bar_w > 0:
                ax.add_patch(plt.Rectangle(
                    (0, y-(BAR_FRAC/2)), bar_w, BAR_FRAC,
                    facecolor=rating_color_hex(bar_w), edgecolor="none", linewidth=0, zorder=1.0
                ))
            x_text = 1.0 if bar_w >= 3 else min(100.0, bar_w + 0.8)
            ax.text(x_text, y, val_str, ha="left", va="center", color="#0B0B0B", fontproperties=BAR_VALUE_FP, zorder=2.0, clip_on=False)

        for i, (lab, _, _) in enumerate(tuples[::-1]):
            y_fig = (panel_top - header_h - n * row_slot) + ((i + 0.5) * row_slot)
            fig.text(LEFT, y_fig, lab, ha="left", va="center", color=LABEL_C, fontproperties=LABEL_FP)

        if show_xticks:
            trans = ax.get_xaxis_transform()
            offset_inner = ScaledTranslation(7/72, 0, fig.dpi_scale_trans)
            offset_pct_0 = ScaledTranslation(4/72, 0, fig.dpi_scale_trans)
            offset_pct_100 = ScaledTranslation(10/72, 0, fig.dpi_scale_trans)
            y_label = -0.075
            for gx in ticks:
                ax.plot([gx, gx], [-0.03, 0.0], transform=trans, color=(0, 0, 0, 0.6), lw=1.1, clip_on=False, zorder=4)
                if gx == 0:
                    ax.text(gx, y_label, "0", transform=trans, ha="center", va="top", color="#000", fontproperties=TICK_FP, zorder=4, clip_on=False)
                    ax.text(gx, y_label, "%", transform=trans + offset_pct_0, ha="left", va="top", color="#000", fontproperties=TICK_FP)
                elif gx == 100:
                    ax.text(gx, y_label, "100", transform=trans, ha="center", va="top", color="#000", fontproperties=TICK_FP, zorder=4, clip_on=False)
                    ax.text(gx, y_label, "%", transform=trans + offset_pct_100, ha="left", va="top", color="#000", fontproperties=TICK_FP)
                else:
                    ax.text(gx, y_label, f"{int(gx)}", transform=trans, ha="center", va="top", color="#000", fontproperties=TICK_FP, zorder=4, clip_on=False)
                    ax.text(gx, y_label, "%", transform=trans + offset_inner, ha="left", va="top", color="#000", fontproperties=TICK_FP)

        if draw_bottom_divider:
            y0 = (panel_top - header_h - n * row_slot) - 0.008
            fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT], [y0, y0], transform=fig.transFigure, color=DIVIDER, lw=1.2, alpha=0.35))

        return (panel_top - header_h - n * row_slot) - GAP

    y_top = 1 - TOP - (title_row_h+0.055)
    for sec_idx, (sec_title, sec_data) in enumerate(sections):
        last = (sec_idx == len(sections) - 1)
        y_top = draw_panel(y_top, sec_title, sec_data, show_xticks=last, draw_bottom_divider=not last)

    fig.text((LEFT + 0.215 + (1 - RIGHT)) / 2.0, BOT * 0.1, footer_caption_text if 'footer_caption_text' in locals() else "Percentile Rank",
             ha="center", va="center", color="#222222", fontproperties=FOOTER_FP)

    st.pyplot(fig, use_container_width=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.download_button(
        "⬇️ Download Feature Z (PNG)",
        data=buf.getvalue(),
        file_name=f"{str(name_).replace(' ','_')}_featureZ.png",
        mime="image/png",
        key=f"download_feature_z_{uuid.uuid4().hex}"
    )
    _plt_cleanup.close(fig)

# ======================== per-tab compute + render (with role pickers & players-used tables) ========================
tab_st, tab_att, tab_cm, tab_fb, tab_cb = st.tabs(
    ["Strikers", "Attackers", "Central Midfield", "Fullbacks", "Center Backs"]
)

with tab_st:
    ranked, pool, tag, tmpl_src = compute_strikers()
    render_template_players_used("Striker", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_att:
    st.markdown("### Position filter")
    ROLE_CHOICE_ATT = st.radio(
        "Choose attacker sub-role", ["All","Left Wingers","Right Wingers","Attacking Midfielders"],
        horizontal=True, key="att_role_choice"
    )
    ranked, pool, tag, tmpl_src, _pos_ok = compute_attackers(ROLE_CHOICE_ATT)
    render_template_players_used("Attacker", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_cm:
    ranked, pool, tag, tmpl_src = compute_central_mid()
    render_template_players_used("CM", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_fb:
    st.markdown("### Position filter")
    ROLE_CHOICE_FB = st.radio(
        "Choose fullback side", ["All","Left Backs","Right Backs"],
        horizontal=True, key="fb_role_choice"
    )
    ranked, pool, tag, tmpl_src, _pos_ok = compute_fullbacks(ROLE_CHOICE_FB)
    render_template_players_used("Fullback", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_cb:
    ranked, pool, tag, tmpl_src = compute_center_backs()
    render_template_players_used("CB", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)

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
        require=['Accurate passes, %','xG per 90','Non-penalty goals per 90','Touches in box per 90','xA per 90',
                 'Passes to penalty area per 90','Passes per 90','Progressive passes per 90','Passes to final third per 90',
                 'Dribbles per 90','Progressive runs per 90']
        prefixes=('RWF','LWF','LAMF','RAMF','AMF')
        def pos_ok(s):
            s=str(s).upper().strip()
            if s in ('RW','LW'): return True
            return s.startswith(prefixes)
        def compute(df):
            out=df.copy()
            out["Retention Style"]= _pd.to_numeric(out['Accurate passes, %'],errors="coerce")
            out["Goal Threat"]= 0.4*_pd.to_numeric(out['xG per 90'],errors="coerce")+0.4*_pd.to_numeric(out['Non-penalty goals per 90'],errors="coerce")+0.2*_pd.to_numeric(out['Touches in box per 90'],errors="coerce")
            out["Creativity Threat"]= 0.65*_pd.to_numeric(out['xA per 90'],errors="coerce")+0.35*_pd.to_numeric(out['Passes to penalty area per 90'],errors="coerce")
            out["Passing Volume"]= _pd.to_numeric(out['Passes per 90'],errors="coerce")
            out["Deeper Playmaking"]= 0.5*_pd.to_numeric(out['Progressive passes per 90'],errors="coerce")+0.5*_pd.to_numeric(out['Passes to final third per 90'],errors="coerce")
            out["Ball Carrying"]= 0.6*_pd.to_numeric(out['Dribbles per 90'],errors="coerce")+0.4*_pd.to_numeric(out['Progressive runs per 90'],errors="coerce")
            return out
        agg_cols=["Retention Style","Goal Threat","Creativity Threat","Passing Volume","Deeper Playmaking","Ball Carrying"]
        label_map={
            "Retention Style":"Retention Style",
            "Goal Threat":"Goal Threat",
            "Creativity Threat":"Creativity Threat",
            "Passing Volume":"Passing Volume",
            "Deeper Playmaking":"Deeper Playmaking",
            "Ball Carrying":"Ball Carrying"
        }
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
# Match your actual column name
CONTRACT_COL = "Contract expires"   # <= 2026 -> default red highlight

# --------------------------------------------------------------------------------------
# SETTINGS PANEL
# --------------------------------------------------------------------------------------
with st.expander("Squad Profile settings", expanded=False):

    # --- Squad selection ---
    teams_available = sorted(df["Team"].dropna().unique())

    # Safely try to use player_row if it exists, otherwise fall back to None
    default_team = None
    selected_player_name = None

    player_row_obj = globals().get("player_row", pd.DataFrame())
    if isinstance(player_row_obj, pd.DataFrame) and not player_row_obj.empty:
        default_team = player_row_obj.iloc[0].get("Team", None)
        selected_player_name = player_row_obj.iloc[0].get("Player", None)

    default_idx = 0
    if default_team in teams_available:
        default_idx = teams_available.index(default_team)

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
        0, 5000, 1500, step=250, key="sq_line_important",
    )
    crucial_line = st.slider(
        "Crucial Player line (minutes)",
        0, 5000, 2500, step=250, key="sq_line_crucial",
    )

    band_lines = sorted(
        [("Important Player", important_line),
         ("Crucial Player", crucial_line)],
        key=lambda x: x[1],
    )

    # --- Contract highlight toggle & custom highlight players ---
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

    # --- Labels & points (defaults +2) ---
    show_labels = st.toggle("Show labels", value=True, key="sq_show_labels")
    label_size = st.slider("Label size", 8, 22, 14, 1, key="sq_lblsize")  # +2 vs before
    point_size = st.slider("Point size", 24, 300, 227, 2, key="sq_pts")   # +2 vs before
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
    # Extract year like 2026 from strings such as '2026-06-30' or '2026'
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

# Selected player flag (only if name is known)
squad["Selected"] = False
if selected_player_name:
    squad["Selected"] = squad["Player"] == selected_player_name

# Custom multiple-player red highlight
squad["CustomRed"] = squad["Player"].isin(custom_red_players)

# Final red flag: contract <= 2026 OR selected player OR custom red
squad["IsRed"] = squad["AutoRed"] | squad["Selected"] | squad["CustomRed"]

# --------------------------------------------------------------------------------------
# SCATTER: X = Age, Y = Minutes played
# --------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
fig.patch.set_facecolor(PAGE_BG)
ax.set_facecolor(PLOT_BG)

# Axis limits
ax.set_xlim(min_age_s, max_age_s)
ax.set_ylim(min_minutes_s, max_minutes_s)

ax.set_xlabel("Age", fontsize=16, fontweight="semibold", color=txt_col)
ax.xaxis.labelpad = 14
ax.set_ylabel("Minutes Played", fontsize=16, fontweight="semibold", color=txt_col)

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(250))  # every 250 minutes

for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("semibold")
    tick.set_color(txt_col)
    tick.set_fontsize(14)

# Grid & spines
ax.grid(True, color=GRID_MAJ, linewidth=0.6)
for s in ax.spines.values():
    s.set_color("#e5e7eb")
    s.set_linewidth(1.1)

# --------------------------------------------------------------------------------------
# AGE BANDS (vertical dashed lines + centered titles)
# --------------------------------------------------------------------------------------
line_col = "#FFFFFF"
age_lines = [21, 25, 29, 33]  # section boundaries
for al in age_lines:
    if min_age_s <= al <= max_age_s:
        ax.axvline(al, color=line_col, linestyle=(0, (4, 4)), lw=1.5)

# Compute centers of each age segment and draw titles slightly lower & centered
age_band_labels = ["YOUTH", "ASCENT", "PRIME", "EXPERIENCED", "OLD"]
age_edges = [min_age_s] + age_lines + [max_age_s]
age_centers = [(age_edges[i] + age_edges[i + 1]) / 2 for i in range(len(age_edges) - 1)]

for center, label in zip(age_centers, age_band_labels):
    x_frac = (center - min_age_s) / float(max_age_s - min_age_s) if max_age_s > min_age_s else 0.5
    ax.text(
        x_frac,
        1.01,  # slightly down from the very top
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        color=txt_col,
        ha="center",
        va="bottom",
    )

# --------------------------------------------------------------------------------------
# MINUTES BANDS (horizontal dashed lines + labels)
# --------------------------------------------------------------------------------------
for name, y_val in band_lines:
    if min_minutes_s <= y_val <= max_minutes_s:
        ax.axhline(y_val, color=line_col, linestyle=(0, (4, 4)), lw=1.5)
        # Label near the left edge of the plot
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
# LABELS – simple collision-aware placement
# --------------------------------------------------------------------------------------
if show_labels:
    # Sort by minutes so lower-minute players place first
    squad_sorted = squad.sort_values("Minutes played")
    placed = []  # list of (x, y_label)

    # minimal vertical separation between labels (in minutes)
    min_y_delta = (max_minutes_s - min_minutes_s) * 0.03  # ~3% of axis
    age_tol = 0.7  # labels treated as colliding if ages closer than this

    for _, r in squad_sorted.iterrows():
        x = r["Age"]
        y = r["Minutes played"]
        y_label = y

        # push label up until it doesn't collide with previous labels near same age
        while True:
            collision = False
            for (px, py) in placed:
                if abs(x - px) < age_tol and abs(y_label - py) < min_y_delta:
                    collision = True
                    break
            if not collision:
                break
            y_label += min_y_delta
            if y_label > max_minutes_s:
                # if we run past top, drop it back down a bit
                y_label = max_minutes_s - min_y_delta
                break

        placed.append((x, y_label))

        z = 6 if r["IsRed"] else 5
        # Draw a subtle leader line from point to label if it's moved
        if y_label != y:
            ax.plot([x, x], [y, y_label], linestyle="-", linewidth=0.5, color=txt_col, alpha=0.5, zorder=z)

        t = ax.annotate(
            r["Player"],
            xy=(x, y_label),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=label_size,
            color=txt_col,
            weight="semibold",
            ha="center",
            va="bottom",
            zorder=z + 0.1,
        )
        t.set_path_effects([
            pe.withStroke(linewidth=2, foreground="#020617", alpha=0.9)
        ])

# --------------------------------------------------------------------------------------
# LAYOUT
# --------------------------------------------------------------------------------------
fig.subplots_adjust(
    left=0.06,
    right=0.98,
    bottom=0.11,
    top=1.02 - top_gap_px / float(h_px),
)

# --------------------------------------------------------------------------------------
# RENDER
# --------------------------------------------------------------------------------------
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






























































