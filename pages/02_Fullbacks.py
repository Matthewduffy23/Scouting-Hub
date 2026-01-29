# app.py — Advanced Fullback Scouting System (dataset-switch safe)

import os
import io
import math
from pathlib import Path
import re

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Wedge

# ---- Optional sklearn (fallback provided) ----
try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    class StandardScaler:  # minimal drop-in
        def __init__(self): self.mean_ = None; self.scale_ = None
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = X.mean(axis=0)
            std = X.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            self.scale_ = std
            return self
        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.mean_) / self.scale_
        def fit_transform(self, X):
            self.fit(X); return self.transform(X)

# ✅ --- Tiny CSV loaders (cached) ---
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str) -> pd.DataFrame:
    candidates = [
        Path.cwd() / csv_name,
        Path(__file__).resolve().parent.parent / csv_name,
        Path(__file__).resolve().parent / csv_name,
    ]
    for p in candidates:
        if p.exists():
            return _read_csv_from_path(str(p))
    up = st.file_uploader(f"Upload {csv_name}", type=["csv"])
    if up is None:
        st.stop()
    return _read_csv_from_bytes(up.getvalue())

# 🔍 Detect all CSVs starting with WORLD*
csv_files = [f.name for f in Path.cwd().glob("WORLD*.csv")]
if not csv_files:
    st.error("No WORLD*.csv files found in the project folder.")
    st.stop()

# ----------------- PAGE -----------------
st.set_page_config(page_title="Advanced Fullback Scouting System", layout="wide")
st.title("🔎 Advanced Fullback Scouting System")
st.caption("Use the sidebar to shape your pool. Each section explains what you’re seeing and why.")

selected_file = st.selectbox("Select dataset to load:", csv_files, key="fb_dataset_select")
df = load_df(selected_file)

# If dataset changes, clear dataset-scoped widget state
if st.session_state.get("_active_dataset_fb") != selected_file:
    for k in [
        "fb_leagues_sel",
        # add more fb-specific keys here if you later create them elsewhere
    ]:
        st.session_state.pop(k, None)
    st.session_state["_active_dataset_fb"] = selected_file

# ----------------- CONFIG -----------------
INCLUDED_LEAGUES = [
'England 1.','Spain 1.','Germany 1.','Italy 1.','France 1.',
'England 2.','Belgium 1.','Brazil 1.','Portugal 1.','Argentina 1.',
'USA 1.','Denmark 1.','Poland 1.','Turkey 1.','Netherlands 1.',
'Croatia 1.','Germany 2.','Japan 1.','Switzerland 1.','Spain 2.',
'Norway 1.','Mexico 1.','Sweden 1.','Colombia 1.','Cyprus 1.',
'Czech 1.','Ecuador 1.','Greece 1.','Saudi 1.','Italy 2.',
'Hungary 1.','Austria 1.','Morocco 1.','Korea 1.','Paraguay 1.',
'France 2.','England 3.','Romania 1.','Scotland 1.','Algeria 1.',
'Uruguay 1.','Chile 1.','Egypt 1.','Israel 1.','Brazil 2.',
'Slovenia 1.','Bolivia 1.','Slovakia 1.','Azerbaijan 1.','South Africa 1.',
'UAE 1.','Costa Rica 1.','Peru 1.','Germany 3.','Ukraine 1.',
'Spain 3.','Portugal 2.','Bulgaria 1.','Australia 1.','Serbia 1.',
'Albania 1.','Bosnia 1.','Kosovo 1.','Japan 2.','England 4.',
'Ireland 1.','Russia 1.','Kazakhstan 1.','Nigeria 1.','France 3.',
'Tunisia 1.','Venezuela 1.','Belgium 2.','Finland 1.','Armenia 1.',
'Georgia 1.','Switzerland 2.','Qatar 1.','Uzbekistan 1.','Poland 2.',
'Iceland 1.','Norway 2.','Sweden 2.','North Macedonia 1.','Turkey 2.',
'Korea 2.','Czech 2.','Brazil 3.','Lithuania 1.','Netherlands 2.',
'Malta 1.','Italy 3.','Denmark 2.','Moldova 1.','USA 2.',
'Latvia 1.','Montenegro 1.','Scotland 2.','Canada 1.','Austria 2.',
'Israel 2.','England 7.','Germany 4.','Portugal 3.','England 5.',
'Estonia 1.','England 9.','Northern Ireland 1.','Serbia 2.','Denmark 3.',
'Sweden 3.','Slovenia 2.','Slovakia 2.','Greece 2.','Wales 1.',
'USA 3.','Scotland 3.','England 6.','England 8.','England 10.',
'Estonia 2.', 'China 1.', 'Ireland 2.',
]

PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.', 'France 1.', 'Germany 1.', 'Italy 1.', 'Spain 1.'},
    "Top 20 Europe": {
        'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.',
        'England 2.','Portugal 1.','Belgium 1.','Turkey 1.','Germany 2.','Spain 2.','France 2.',
        'Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'
    },
    "EFL (England 2–4)": {'England 2.','England 3.','England 4.'}
}

FEATURES = [ 'Defensive duels per 90', 'Defensive duels won, %',
    'Aerial duels per 90', 'Aerial duels won, %', 'Shots blocked per 90',
    'PAdj Interceptions', 'Non-penalty goals per 90', 'xG per 90',
    'Shots per 90', 'Shots on target, %', 'Crosses per 90',
    'Accurate crosses, %', 'Dribbles per 90', 'Successful dribbles, %',
    'Offensive duels per 90', 'Offensive duels won, %', 'Touches in box per 90',
    'Progressive runs per 90', 'Accelerations per 90', 'Passes per 90',
    'Accurate passes, %', 'Forward passes per 90', 'Accurate forward passes, %',
    'Long passes per 90', 'Accurate long passes, %', 'xA per 90',
    'Smart passes per 90', 'Key passes per 90', 'Passes to final third per 90',
    'Accurate passes to final third, %', 'Passes to penalty area per 90',
    'Accurate passes to penalty area, %', 'Deep completions per 90',
    'Progressive passes per 90', 'Accurate progressive passes, %'
]

POLAR_METRICS = [
    "Defensive duels per 90","Defensive duels won, %","PAdj Interceptions","Aerial duels won, %",
    "Passes per 90","Accurate passes, %","Progressive passes per 90",
    "Progressive runs per 90","Dribbles per 90","xA per 90","Passes to penalty area per 90"
]

# -------- Position filter (fullbacks) --------
# Uses the sidebar's ROLE_CHOICE: "All", "Left Backs", "Right Backs"
def position_filter(pos) -> bool:
    s = str(pos).strip().upper()
    if ROLE_CHOICE == "Right Backs":
        prefixes = ("RB", "RWB")
    elif ROLE_CHOICE == "Left Backs":
        prefixes = ("LB", "LWB")
    else:  # All
        prefixes = ("RB", "RWB", "LB", "LWB")
    return any(s.startswith(p) for p in prefixes)
# ---------------------------------------------



# Role buckets
ROLES = {
    'Build Up FB': {
        'metrics': {
            'Passes per 90': 2, 'Accurate passes, %': 1.5, 'Forward passes per 90': 2,
            'Accurate forward passes, %': 2, 'Progressive passes per 90': 2.5, 'Progressive runs per 90': 2,
            'Dribbles per 90': 2, 'Passes to final third per 90': 2,
            'xA per 90': 1,
        }
    },
    'Attacking FB': {
        'metrics': {
            'Crosses per 90': 2, 'Dribbles per 90': 3.5,
            'Accelerations per 90': 1, 'Successful dribbles, %': 1,
            'Touches in box per 90': 2, 'Progressive runs per 90': 3,
            'Passes to penalty area per 90': 2, 'xA per 90': 3,
        }
    },
    'Defensive FB': {
        'metrics': {
            'Aerial duels per 90': 1, 'Aerial duels won, %': 1.5, 'Defensive duels per 90': 2,
            'PAdj Interceptions': 3, 'Shots blocked per 90': 1, 'Defensive duels won, %': 3.5,
        }
    },

    # ===== NEW ROLES (inserted here) =====
    'Wide Creator': {
        'metrics': {
            'xA per 90': 3,
            'Crosses per 90': 3,
            'Accurate crosses, %': 1,
        }
    },
    'Wide Carrier': {
        'metrics': {
            'Dribbles per 90': 3,
            'Successful dribbles, %': 1,
            'Progressive runs per 90': 3,
            'Accelerations per 90': 1,
        }
    },
    'PL Profile': {
        'metrics': {
            'Defensive duels per 90': 2,
            'Defensive duels won, %': 2,
            'Aerial duels won, %': 1,
            'Progressive runs per 90': 2,
            'xA per 90': 1,
            'Dribbles per 90': 2,
            'Pass Ratio': 2,   # ✅ NEW — progressive / total passing tendency
        }
    },
    # ===== END NEW ROLES =====

    'All In': {
        'metrics': {
            'Progressive passes per 90': 2, 'xA per 90': 2, 'PAdj Interceptions': 2,
            'Progressive runs per 90': 2, 'Defensive duels won, %': 2, 'Dribbles per 90': 1.5,
        }
    }
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
'Iceland 1.':46.08,'Norway 2.':45.88,'Sweden 2.':45.69,'North Macedonia 1.':44.71,'China 1.':44.7,'Turkey 2.':44.51,
'Korea 2.':43.53,'Czech 2.':43.33,'Brazil 3.':43.14,'Lithuania 1.':42.35,'Netherlands 2.':42.16,
'Malta 1.':41.96,'Italy 3.':45,'Denmark 2.':40.39,'Moldova 1.':40.39,'USA 2.':40.00,
'Latvia 1.':40.00,'Montenegro 1.':39.80,'Scotland 2.':38.63,'Canada 1.':38.24,'Austria 2.':38.24,
'Israel 2.':38.04,'England 7.':37.25,'Germany 4.':35.29,'Portugal 3.':35.29,'England 5.':33.33,
'Estonia 1.':40,'England 9.':31.37,'Northern Ireland 1.':30.98,'Serbia 2.':30.39,'Denmark 3.':29.41,
'Sweden 3.':29.41,'Slovenia 2.':28.82,'Slovakia 2.':28.24,'Greece 2.':27.06,'Wales 1.':26.67,
'USA 3.':22.55,'Scotland 3.':20.00,'England 6.':16.08,'England 8.':15.69,'England 10.':3.92,
'Estonia 2.':3, 'Ireland 2.':10,
}

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
    # You can keep England 2./3. here as well if you want, but they’re already Band 1 above.

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
    """
    Map 'Country N.' league name to custom GBE band 1–6.
    Unlisted leagues default to Band 6.
    """
    league_name = str(league_name).strip()
    return int(GBE_LEAGUE_BANDS.get(league_name, 6))


import re

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
    "Northern Ireland": "Europe", "Wales": "Europe", "Russia": "Europe", "Kazakhstan": "Europe",
    "Lithuania": "Europe", "Malta": "Europe", "Moldova": "Europe", "Israel": "Europe",

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

    # Asia (incl. some UEFA/dual countries – choose what fits your model best)
    "Japan": "Asia", "Korea": "Asia", "Saudi": "Asia",
    "UAE": "Asia", "Qatar": "Asia", "Uzbekistan": "Asia", 
    "Turkey": "Asia", "Azerbaijan": "Asia", "China": "Asia",

    # Oceania / other (you can change this to "Other" if you prefer)
    "Australia": "Asia",
}

def extract_country(league_name: str) -> str:
    """
    'South Africa 1.'  -> 'South Africa'
    'England 2.'       -> 'England'
    'Costa Rica 1.'    -> 'Costa Rica'
    """
    s = str(league_name).strip()
    s = s.rstrip(".")                      # 'South Africa 1.' -> 'South Africa 1'
    s = re.sub(r"\s+\d+$", "", s)         # drop trailing ' 1', ' 2', etc.
    return s

def league_region(league_name: str) -> str:
    """
    Map league string to one of the high-level regions.
    Falls back to 'Other' if unknown.
    """
    country = extract_country(league_name)
    return COUNTRY_TO_REGION.get(country, "Other")


REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}

# ----------------- WIDGET SAFETY -----------------
def multiselect_safe(label, *, options, default=None, key=None, **kwargs):
    """Clamp defaults to current options to avoid Streamlit crashes after dataset switches."""
    options = list(options)
    default = [x for x in (default or []) if x in options]
    return st.multiselect(label, options=options, default=default, key=key, **kwargs)

# ----------------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.header("Filters")

    # --- Region filter (default = all regions) ---
    all_regions = ["Europe", "Africa", "Asia", "North America", "South America"]
    regions_key = f"fb_regions_{selected_file}"
    if regions_key not in st.session_state:
        st.session_state[regions_key] = all_regions

    regions_sel = st.multiselect(
        "Regions",
        options=all_regions,
        default=st.session_state[regions_key],
        key=regions_key,
    )

    # --- Youth leagues toggle (excluded by default) ---
    include_youth_key = f"fb_include_youth_{selected_file}"
    include_youth = st.checkbox(
        "Include youth leagues",
        value=False,
        key=include_youth_key,
    )

    st.markdown("---")

    # --- League presets ---
    c1, c2, c3 = st.columns([1, 1, 1])
    use_top5  = c1.checkbox("Top-5 EU", value=False, key=f"fb_top5_{selected_file}")
    use_top20 = c2.checkbox("Top-20 EU", value=False, key=f"fb_top20_{selected_file}")
    use_efl   = c3.checkbox("EFL",      value=False, key=f"fb_efl_{selected_file}")

    # Preset → seed leagues
    seed = set()
    if use_top5:
        seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top20:
        seed |= PRESET_LEAGUES["Top 20 Europe"]
    if use_efl:
        seed |= PRESET_LEAGUES["EFL (England 2–4)"]

    # --- GBE Band presets (1–6) ---
    st.markdown("### GBE Bands")

    b1, b2, b3 = st.columns(3)
    use_band1 = b1.checkbox("Band 1", value=True,  key=f"fb_band1_{selected_file}")
    use_band2 = b2.checkbox("Band 2", value=True,  key=f"fb_band2_{selected_file}")
    use_band3 = b3.checkbox("Band 3", value=True,  key=f"fb_band3_{selected_file}")

    c4, c5, c6 = st.columns(3)
    use_band4 = c4.checkbox("Band 4", value=True,  key=f"fb_band4_{selected_file}")
    use_band5 = c5.checkbox("Band 5", value=True,  key=f"fb_band5_{selected_file}")
    use_band6 = c6.checkbox("Band 6", value=True,  key=f"fb_band6_{selected_file}")

    selected_bands: list[int] = []
    if use_band1: selected_bands.append(1)
    if use_band2: selected_bands.append(2)
    if use_band3: selected_bands.append(3)
    if use_band4: selected_bands.append(4)
    if use_band5: selected_bands.append(5)
    if use_band6: selected_bands.append(6)

    st.session_state[f"fb_gbe_bands_{selected_file}"] = selected_bands


    # Leagues present in this dataset
    leagues_in_df = sorted(
        pd.Series(df.get("League", pd.Series(dtype=object)))
          .dropna()
          .unique()
          .tolist()
    )

    # Apply region filter
    if regions_sel:
        leagues_in_df = [
            lg for lg in leagues_in_df
            if league_region(lg) in regions_sel
        ]

    # Apply youth-league filter
    if include_youth:
        leagues_avail = leagues_in_df
    else:
        leagues_avail = [lg for lg in leagues_in_df if lg not in YOUTH_LEAGUES]

    # Clamp presets to currently available leagues
    seed = {x for x in seed if x in leagues_avail}
    default_leagues = sorted(seed) if seed else leagues_avail

    # Multiselect with preset + region + youth sensitivity
    ms_key = f"fb_leagues_sel_{selected_file}"
    preset_sig = (
        tuple(sorted(regions_sel)),
        include_youth,
        use_top5,
        use_top20,
        use_efl,
        selected_file,
    )

    # Seed multiselect defaults (dataset-safe)
    if ms_key not in st.session_state:
        st.session_state[ms_key] = default_leagues
    if st.session_state.get("fb_preset_sig") != preset_sig:
        st.session_state["fb_preset_sig"] = preset_sig
        st.session_state[ms_key] = default_leagues

    leagues_sel = multiselect_safe(
        "Leagues (add or prune the presets)",
        options=leagues_avail,
        default=st.session_state[ms_key],
        key=ms_key,
    )
    st.session_state["fb_leagues_sel"] = leagues_sel

    # Numeric coercions
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    min_minutes, max_minutes = st.slider(
        "Minutes played", 0, 5000, (500, 5000), key=f"fb_minmax_minutes_{selected_file}"
    )

    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    def_age_lo = max(16, age_min_data)
    def_age_hi = min(40, age_max_data)
    if def_age_lo > def_age_hi:  # guard tiny datasets
        def_age_lo, def_age_hi = age_min_data, age_max_data

    min_age, max_age = st.slider(
        "Age", age_min_data, age_max_data, (def_age_lo, def_age_hi), key=f"fb_minmax_age_{selected_file}"
    )

    # Role toggle (drives position_filter)
    st.subheader("Fullback roles to include")
    role_choice = st.radio(
        " ",  # no label
        ["All", "Left Backs", "Right Backs"],
        index=0,
        key=f"fb_role_choice_{selected_file}",
    )
    # Keep in session for other blocks if needed
    st.session_state["fb_role_choice"] = role_choice

    # Contract filter
    apply_contract = st.checkbox("Filter by contract expiry", value=False, key=f"fb_apply_contract_{selected_file}")
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026, key=f"fb_cutoff_{selected_file}")

    # League strength and weighting
    min_strength, max_strength = st.slider(
        "League quality (strength)", 0, 101, (0, 101), key=f"fb_minmax_strength_{selected_file}"
    )
    use_league_weighting = st.checkbox(
        "Use league weighting in role score", value=False, key=f"fb_use_lw_{selected_file}"
    )
    beta = st.slider(
        "League weighting beta", 0.0, 1.0, 0.40, 0.05,
        help="0 = ignore league strength; 1 = only league strength",
        key=f"fb_beta_{selected_file}"
    )

    # Market value
    df["Market value"] = pd.to_numeric(df["Market value"], errors="coerce")
    mv_col = "Market value"
    mv_max_raw = int(np.nanmax(df[mv_col])) if df[mv_col].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    st.markdown("**Market value (€)**")
    use_m = st.checkbox("Adjust in millions", True, key=f"fb_use_m_{selected_file}")
    if use_m:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, max_m), key=f"fb_mv_range_m_{selected_file}")
        min_value = mv_min_m * 1_000_000
        max_value = mv_max_m * 1_000_000
    else:
        min_value, max_value = st.slider(
            "Range (€)", 0, mv_cap, (0, mv_cap), step=100_000, key=f"fb_mv_range_{selected_file}"
        )
    value_band_max = st.number_input(
        "Value band (tab 4 max €)", min_value=0,
        value=min_value if min_value > 0 else 5_000_000, step=250_000, key=f"fb_value_band_{selected_file}"
    )

    # Thresholds
    st.subheader("Minimum performance thresholds")
    enable_min_perf = st.checkbox(
        "Require minimum percentile on selected metrics", value=False, key=f"fb_enable_min_perf_{selected_file}"
    )
    sel_metrics = st.multiselect(
        "Metrics to threshold", FEATURES[:],
        default=(['Non-penalty goals per 90', 'xG per 90'] if enable_min_perf else []),
        key=f"fb_sel_metrics_{selected_file}"
    )
    min_pct = st.slider("Minimum percentile (0–100)", 0, 100, 60, key=f"fb_min_pct_{selected_file}")

    # Table display
    top_n = st.number_input("Top N per table", 5, 200, 50, 5, key=f"fb_topn_{selected_file}")
    round_to = st.selectbox("Round output percentiles to", [0, 1], index=0, key=f"fb_round_to_{selected_file}")

# Make ROLE_CHOICE available to position_filter()
ROLE_CHOICE = st.session_state.get("fb_role_choice", "All")

# ----------------- VALIDATION -----------------
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required base columns: {missing}")
    st.stop()
missing_feats = [c for c in FEATURES if c not in df.columns]
if missing_feats:
    st.error(f"Dataset missing required feature columns: {missing_feats}")
    st.stop()

# ----------------- FILTER POOL -----------------
df_f = df.copy()

# leagues
if st.session_state.get("fb_leagues_sel"):
    df_f = df_f[df_f["League"].isin(st.session_state["fb_leagues_sel"])]

# --- GBE band filter (NO auto-pass) ---
df_f["GBE Band"] = df_f["League"].apply(gbe_league_band).astype(int)

bands_sel = st.session_state.get(f"fb_gbe_bands_{selected_file}", [1, 2, 3, 4, 5, 6])
if bands_sel:
    df_f = df_f[df_f["GBE Band"].isin(bands_sel)]


# positions (fullbacks)
df_f = df_f[df_f["Position"].astype(str).apply(position_filter)]

# numerics
df_f["Minutes played"] = pd.to_numeric(df_f["Minutes played"], errors="coerce")
df_f["Age"] = pd.to_numeric(df_f["Age"], errors="coerce")
min_minutes, max_minutes = st.session_state[f"fb_minmax_minutes_{selected_file}"]
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
min_age, max_age = st.session_state[f"fb_minmax_age_{selected_file}"]
df_f = df_f[df_f["Age"].between(min_age, max_age)]

# contract (optional)
df_f["Contract expires"] = pd.to_datetime(df_f["Contract expires"], errors="coerce")
if st.session_state.get(f"fb_apply_contract_{selected_file}", False):
    cutoff_year = st.session_state[f"fb_cutoff_{selected_file}"]
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]

# league strength — default to 50.0 for unknown leagues
df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(50.0)
min_strength, max_strength = st.session_state[f"fb_minmax_strength_{selected_file}"]
df_f = df_f[(df_f["League Strength"] >= float(min_strength)) & (df_f["League Strength"] <= float(max_strength))]

# market value
df_f["Market value"] = pd.to_numeric(df_f["Market value"], errors="coerce")
if st.session_state.get(f"fb_use_m_{selected_file}", True):
    mv_min_m, mv_max_m = st.session_state[f"fb_mv_range_m_{selected_file}"]
    min_value = mv_min_m * 1_000_000
    max_value = mv_max_m * 1_000_000
else:
    min_value, max_value = st.session_state[f"fb_mv_range_{selected_file}"]
df_f = df_f[(df_f["Market value"] >= min_value) & (df_f["Market value"] <= max_value)]

# features numeric + dropna
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)

if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# ---------- DERIVED METRIC: Pass Ratio (compute BEFORE percentiles) ----------
# Progressive passing tendency = Progressive passes per 90 / Passes per 90
pp90 = pd.to_numeric(df_f.get("Progressive passes per 90"), errors="coerce")
p90  = pd.to_numeric(df_f.get("Passes per 90"), errors="coerce")

with np.errstate(divide="ignore", invalid="ignore"):
    pass_ratio = np.where(p90 > 0, pp90 / p90, np.nan)

# Clip to 0–1 so it behaves like a rate
df_f["Pass Ratio"] = np.clip(pass_ratio, 0, 1.0)
# ---------------------------------------------------------------------------


# ----------------- PERCENTILES FOR TABLES (per league) -----------------
for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# Percentiles for Pass Ratio so roles can weight it like any other metric
df_f["Pass Ratio Percentile"] = (
    df_f.groupby("League")["Pass Ratio"]
        .transform(lambda x: x.rank(pct=True) * 100.0)
)

if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# ===================================================================
#  FB IMPACT FEATURE BLOCK – METRICS + CRESTS + CIES-STYLE IMAGE
#  (Pool defined by sidebar; display filters do NOT change pool)
# ===================================================================

import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ---------------------------------------------------------
# 0) REMOTE PNG LOADER (logos + special flags)
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def fb_load_remote_png(url: str):
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        return plt.imread(io.BytesIO(r.content))
    except Exception:
        return None


def fb_scale_0_100(s: pd.Series, default: float = 50.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        return 100.0 * (s - lo) / (hi - lo)
    return pd.Series(default, index=s.index, dtype=float)


# ---------------------------------------------------------
# 1) ENSURE IMPACT / BUCKET SCORES EXIST (+ COMPLETE SCORE)
# ---------------------------------------------------------

def fb_pct(m: str) -> str:
    return "%s Percentile" % m


def ensure_fb_impact_metrics(df_f: pd.DataFrame, selected_file: str) -> pd.DataFrame:
    required_cols = [
        "Impact Score", "Impact Score (no league)",
        "Aerial Score", "Ground Score", "Retention Score",
        "Carrying Score", "Playmaking Score", "Chance Creation Score",
        "League Factor",
        "Raw Impact Score", "Raw Impact No League"
    ]

    has_all_old = all(c in df_f.columns for c in required_cols)
    df_f = df_f.copy()

    if not has_all_old:
        # ----- Sub-scores -----
        df_f["Aerial Score"] = (
            0.30 * df_f[fb_pct("Aerial duels per 90")] +
            0.70 * df_f[fb_pct("Aerial duels won, %")]
        )

        df_f["Ground Score"] = (
            0.30 * df_f[fb_pct("Defensive duels per 90")] +
            0.70 * df_f[fb_pct("Defensive duels won, %")]
        )

        df_f["Retention Score"] = (
            0.25 * df_f[fb_pct("Accurate passes, %")] +
            0.25 * df_f[fb_pct("Accurate forward passes, %")] +
            0.25 * df_f[fb_pct("Accurate progressive passes, %")] +
            0.25 * df_f[fb_pct("Accurate long passes, %")]
        )

        df_f["Carrying Score"] = (
            0.40 * df_f[fb_pct("Dribbles per 90")] +
            0.20 * df_f[fb_pct("Successful dribbles, %")] +
            0.40 * df_f[fb_pct("Progressive runs per 90")]
        )

        df_f["Playmaking Score"] = (
            0.50 * df_f[fb_pct("Progressive passes per 90")] +
            0.25 * df_f[fb_pct("Forward passes per 90")] +
            0.25 * df_f[fb_pct("Passes to final third per 90")]
        )

        df_f["Chance Creation Score"] = (
            0.60 * df_f[fb_pct("xA per 90")] +
            0.20 * df_f[fb_pct("Passes to penalty area per 90")] +
            0.10 * df_f[fb_pct("Smart passes per 90")] +
            0.10 * df_f[fb_pct("Crosses per 90")]
        )

        sub_scores = [
            "Aerial Score", "Ground Score", "Retention Score",
            "Carrying Score", "Playmaking Score", "Chance Creation Score",
        ]
        df_f["Base FB Score"] = df_f[sub_scores].mean(axis=1)

        # ----- Minutes factor -----
        minutes_pct = df_f.groupby("League")["Minutes played"].rank(pct=True)
        df_f["Minutes Factor"] = 0.90 + 0.20 * minutes_pct

        # ----- Team context factor -----
        league_avg = df_f.groupby("League")["Base FB Score"].transform("mean")
        team_avg   = df_f.groupby(["League", "Team"])["Base FB Score"].transform("mean")

        with np.errstate(divide="ignore", invalid="ignore"):
            strength_ratio = team_avg / league_avg.replace(0, np.nan)

        raw_team_factor = np.where(strength_ratio > 0, 1.0 / strength_ratio, 1.0)
        df_f["Team Context Factor"] = np.clip(raw_team_factor, 0.90, 1.10)
        df_f["Team Context Factor"] = df_f["Team Context Factor"].fillna(1.0)

        # ----- Raw impact (no league) -----
        df_f["Raw Impact No League"] = (
            df_f["Base FB Score"] *
            df_f["Minutes Factor"] *
            df_f["Team Context Factor"]
        )

        # ----- League factor -----
        # Ensure League Strength exists
        if "League Strength" not in df_f.columns:
            df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).astype(float)

        ls_norm = df_f["League Strength"].fillna(50.0).astype(float) / 100.0
        ls_norm = np.clip(ls_norm, 0.30, 1.00)

        beta_league = float(st.session_state.get("fb_beta_%s" % selected_file, 0.40))
        gamma = 1.0 + 1.5 * beta_league

        df_f["League Factor"] = np.power(ls_norm, gamma)

        # ----- Raw impact with league -----
        df_f["Raw Impact Score"] = df_f["Raw Impact No League"] * df_f["League Factor"]

        # ----- 0–100 scaling of impact (global) -----
        df_f["Impact Score"]             = fb_scale_0_100(df_f["Raw Impact Score"]).astype(float)
        df_f["Impact Score (no league)"] = fb_scale_0_100(df_f["Raw Impact No League"]).astype(float)

    # ----- Complete Score -----
    if "Complete Score" not in df_f.columns:
        df_f["Complete Score"] = (
            0.10 * df_f[fb_pct("PAdj Interceptions")] +
            0.10 * df_f[fb_pct("Defensive duels won, %")] +
            0.10 * df_f[fb_pct("Accurate passes, %")] +
            0.05 * df_f[fb_pct("Defensive duels per 90")] +
            0.10 * df_f[fb_pct("Dribbles per 90")] +
            0.10 * df_f[fb_pct("Progressive runs per 90")] +
            0.10 * df_f[fb_pct("Progressive passes per 90")] +
            0.10 * df_f[fb_pct("Passes to final third per 90")] +
            0.10 * df_f[fb_pct("xA per 90")] +
            0.10 * df_f[fb_pct("Passes to penalty area per 90")] +
            0.05 * df_f[fb_pct("Smart passes per 90")]
        )

    return df_f


df_f = ensure_fb_impact_metrics(df_f, selected_file)

# ---------------------------------------------------------
# 2) RANKING / DISPLAY CONTROLS
# ---------------------------------------------------------

rank_mode = st.radio(
    "Ranking mode (FB)",
    ["Composite (FB scores)", "Raw metric (any numeric column)"],
    index=0,
    horizontal=True,
    key="fb_rank_mode_%s" % selected_file,
)

RANK_OPTIONS = {
    "Impact Score": "Impact Score",
    "Aerial Score": "Aerial Score",
    "Ground Score": "Ground Score",
    "Retention Score": "Retention Score",
    "Carrying Score": "Carrying Score",
    "Playmaking Score": "Playmaking Score",
    "Chance Creation Score": "Chance Creation Score",
    "Complete Score": "Complete Score",
}

BASE_COMPOSITE_COLS = [
    "Aerial Score",
    "Ground Score",
    "Retention Score",
    "Carrying Score",
    "Playmaking Score",
    "Chance Creation Score",
]


def fb_raw_metric_candidates(df: pd.DataFrame):
    bad = {
        "Impact Score", "Impact Score (no league)",
        "Aerial Score", "Ground Score", "Retention Score", "Carrying Score",
        "Playmaking Score", "Chance Creation Score",
        "Base FB Score", "Raw Impact Score", "Raw Impact No League",
        "Minutes Factor", "Team Context Factor", "League Factor",
        "Complete Score",
        "Custom Combo Raw", "_MetricForBars",
    }
    numeric_cols = []
    for c in df.columns:
        if c in bad:
            continue
        if df[c].dtype.kind in ("i", "u", "f"):
            numeric_cols.append(c)
    return sorted(numeric_cols)


raw_metric_list = fb_raw_metric_candidates(df_f)

use_custom_combo = False
custom_combo_components = []

if rank_mode == "Composite (FB scores)":
    use_custom_combo = st.checkbox(
        "Use custom combination of base FB scores (equal weights)",
        value=False,
        key="fb_use_custom_combo_%s" % selected_file,
        help="Combine any of Aerial/Ground/Retention/Carrying/Playmaking/Chance Creation into a single equal-weight score.",
    )
    if use_custom_combo:
        custom_combo_components = st.multiselect(
            "Base scores to include in custom combo (FB)",
            BASE_COMPOSITE_COLS,
            default=["Aerial Score", "Ground Score", "Chance Creation Score"],
            key="fb_custom_combo_components_%s" % selected_file,
        )
        rank_label = "Custom Combo"
    else:
        rank_label = st.selectbox(
            "Ranking metric (FB composite)",
            list(RANK_OPTIONS.keys()),
            index=0,
            key="fb_rank_metric_%s" % selected_file,
        )
else:
    default_raw = (
        "Progressive passes per 90"
        if "Progressive passes per 90" in raw_metric_list
        else (raw_metric_list[0] if raw_metric_list else None)
    )
    rank_label = st.selectbox(
        "Ranking metric (raw column, FB)",
        raw_metric_list,
        index=(raw_metric_list.index(default_raw) if default_raw in raw_metric_list else 0),
        key="fb_rank_raw_metric_%s" % selected_file,
        help="Bars/ranks are scaled vs pool; printed value is also scaled 0–100.",
    )

display_with_league_strength = st.checkbox(
    "Display league-strength adjusted (0–100) – FB",
    value=False,
    key="fb_display_ls_%s" % selected_file,
)

all_leagues_in_pool = sorted([x for x in df_f["League"].dropna().unique()])
selected_display_league = st.selectbox(
    "Display league (does not change pool, FB)",
    ["All leagues"] + all_leagues_in_pool,
    index=0,
    key="fb_display_league_dd_%s" % selected_file,
)

selected_display_team = "All teams"
if selected_display_league != "All leagues":
    teams_in_league = sorted(
        df_f.loc[df_f["League"] == selected_display_league, "Team"].dropna().unique().tolist()
    )
    selected_display_team = st.selectbox(
        "Display team (does not change pool, FB)",
        ["All teams"] + teams_in_league,
        index=0,
        key="fb_display_team_dd_%s" % selected_file,
    )

display_ls_min, display_ls_max = st.slider(
    "Display league strength range (does not change pool, FB)",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=1,
    key="fb_display_ls_range_%s" % selected_file,
)

max_rank_age = st.number_input(
    "Max age in displayed list/image (does not change pool, FB)",
    min_value=16, max_value=40, value=23, step=1,
    key="fb_display_age_%s" % selected_file,
)

show_age_in_image = st.checkbox(
    "Show age in ranking image (FB)",
    value=False,
    key="fb_show_age_img_%s" % selected_file,
)

show_league_strength_col = st.checkbox(
    "Show League Strength column in table (FB)",
    value=True,
    key="fb_show_ls_col_%s" % selected_file,
)

image_theme = st.selectbox(
    "Image theme (FB)",
    ["Light", "Dark"],
    index=0,
    key="fb_img_theme_%s" % selected_file,
)

enable_highlight_players = st.checkbox(
    "Highlight players in the ranking image (FB)",
    value=False,
    key="fb_enable_highlight_%s" % selected_file,
)
highlight_player_names = []
if enable_highlight_players:
    player_opts = sorted(df_f["Player"].dropna().astype(str).unique().tolist())
    highlight_player_names = st.multiselect(
        "Players to highlight (FB)",
        options=player_opts,
        default=[],
        key="fb_highlight_players_%s" % selected_file,
    )


# ---------------------------------------------------------
# 3) BUILD DISPLAY METRIC COLUMN – ONE NORMALISED COLUMN
# ---------------------------------------------------------

df_pool = df_f.copy()

# base_for_display_raw → underlying metric BEFORE scaling to 0–100
if rank_mode == "Composite (FB scores)":
    if use_custom_combo:
        if custom_combo_components:
            valid_components = [c for c in custom_combo_components if c in df_pool.columns]
        else:
            valid_components = []
        if valid_components:
            df_pool["Custom Combo Raw"] = df_pool[valid_components].mean(axis=1)
        else:
            df_pool["Custom Combo Raw"] = df_pool["Base FB Score"]

        base_for_display_raw = df_pool["Custom Combo Raw"]

        if display_with_league_strength:
            base_for_display_raw = base_for_display_raw * df_pool["League Factor"]

        metric_label_for_image = "Custom Combo"

    else:
        if rank_label == "Impact Score":
            # Use raw impact so LS on/off behaves exactly like for other metrics
            if display_with_league_strength:
                base_for_display_raw = df_pool["Raw Impact Score"]
            else:
                base_for_display_raw = df_pool["Raw Impact No League"]
        else:
            base_col = RANK_OPTIONS[rank_label]
            base_for_display_raw = df_pool[base_col]
            if display_with_league_strength:
                base_for_display_raw = base_for_display_raw * df_pool["League Factor"]

        metric_label_for_image = rank_label

else:
    raw_col = rank_label
    base_for_display_raw = df_pool[raw_col]
    if display_with_league_strength:
        base_for_display_raw = base_for_display_raw * df_pool["League Factor"]
    metric_label_for_image = raw_col

# Single 0–100 column used for:
#   - sorting
#   - bar lengths
df_pool["_MetricForBars"] = fb_scale_0_100(base_for_display_raw)
display_metric_col = "_MetricForBars"

# ✅ Raw metric mode prints actual value (not 0–100)
if rank_mode == "Raw metric (any numeric column)":
    value_label_col = rank_label
else:
    value_label_col = "_MetricForBars"

def fb_format_market_value(v) -> str:
    """Format MV: 1750000 -> £1.75m, 30000000 -> £30m, etc."""
    if v is None:
        return "—"
    v = pd.to_numeric(v, errors="coerce")
    if not np.isfinite(v):
        return "—"

    v = float(v)
    if v >= 1_000_000:
        s = f"{v/1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"£{s}m"
    if v >= 1_000:
        return f"£{int(round(v/1_000))}k"
    return f"£{int(v):,}"

# ----------------- DISPLAY-ONLY MARKET VALUE FILTER (FB) -----------------
display_mv_mode = st.selectbox(
    "Display Market Value filter (does not change pool) – FB",
    ["Off", "Max only", "Range"],
    index=1,  # default Max only (set 0 if you want Off)
    key=f"fb_display_mv_mode_{selected_file}",
)

mv_all = pd.to_numeric(df_f["Market value"], errors="coerce")
mv_hi = float(np.nanmax(mv_all)) if mv_all.notna().any() else 50_000_000.0

# slider range in MILLIONS
mv_cap_m = int(math.ceil(mv_hi / 1_000_000.0))
mv_cap_m = max(1, mv_cap_m)

display_mv_min = None
display_mv_max = None

if display_mv_mode == "Max only":
    mv_max_m = st.slider(
        "Max market value to display (M£) – FB",
        0, mv_cap_m,
        min(10, mv_cap_m),
        step=1,
        key=f"fb_display_mv_max_m_{selected_file}",
    )
    display_mv_max = mv_max_m * 1_000_000
    st.caption(f"Max MV: **{fb_format_market_value(display_mv_max)}**")

elif display_mv_mode == "Range":
    mv_min_m, mv_max_m = st.slider(
        "Market value range to display (M£) – FB",
        0, mv_cap_m,
        (0, min(10, mv_cap_m)),
        step=1,
        key=f"fb_display_mv_range_m_{selected_file}",
    )
    display_mv_min = mv_min_m * 1_000_000
    display_mv_max = mv_max_m * 1_000_000
    st.caption(
        f"MV range: **{fb_format_market_value(display_mv_min)} → {fb_format_market_value(display_mv_max)}**"
    )




# ---------------------------------------------------------
# 4) DISPLAY FILTERS (do not change pool scaling)
# ---------------------------------------------------------

df_display = df_pool.copy()
df_display = df_display[df_display["Age"] <= max_rank_age]
df_display = df_display[df_display["League Strength"].between(display_ls_min, display_ls_max)]

# ✅ FB display-only market value filter
if display_mv_max is not None:
    df_display = df_display[
        pd.to_numeric(df_display["Market value"], errors="coerce") <= float(display_mv_max)
    ]

if display_mv_min is not None:
    df_display = df_display[
        pd.to_numeric(df_display["Market value"], errors="coerce") >= float(display_mv_min)
    ]


if selected_display_league != "All leagues":
    df_display = df_display[df_display["League"] == selected_display_league]
    if selected_display_team != "All teams":
        df_display = df_display[df_display["Team"] == selected_display_team]

df_display = df_display.dropna(subset=[display_metric_col]).sort_values(display_metric_col, ascending=False).copy()

top_n_fb = 10
top_n_fb = 10
df_table = df_display.copy()
df_table["Market value"] = pd.to_numeric(df_table["Market value"], errors="coerce").apply(fb_format_market_value)
st.dataframe(df_table.head(top_n_fb), use_container_width=True)



# ---------------------------------------------------------
# 5) FLAGS (Twemoji) + UK HOME NATIONS + RestCountries fallback
# ---------------------------------------------------------

_FB_CC_MAP = {
    "england": "FLAG_ENG",
    "scotland": "FLAG_SCT",
    "wales": "FLAG_WLS",
    "northern ireland": "gb",
    "north ireland": "gb",
}

_FB_TWEMOJI_SPECIAL = {
    "FLAG_ENG": "1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",
    "FLAG_SCT": "1f3f4-e0067-e0062-e0073-e0063-e0074-e007f",
    "FLAG_WLS": "1f3f4-e0067-e0062-e0077-e006c-e0073-e007f",
}

_FB_SPECIAL_FLAG_URLS = {
    "FLAG_NIR": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Ulster_banner.svg/200px-Ulster_banner.svg.png"
}

# NOTE: keys are fb-normalised (ASCII, lowercased) versions of your Birth country values
_FB_COUNTRY_OVERRIDES = {
    "netherlands": "nl",
    "spain": "es",
    "serbia": "rs",
    "germany": "de",
    "republic of ireland": "ie",
    "france": "fr",
    "slovakia": "sk",
    "italy": "it",
    "switzerland": "ch",
    "belgium": "be",
    "hungary": "hu",
    "cote d'ivoire": "ci",
    "portugal": "pt",
    "brazil": "br",
    "argentina": "ar",
    "denmark": "dk",
    "sierra leone": "sl",
    "united states": "us",
    "norway": "no",
    "sweden": "se",
    "united states": "us",
    "gambia": "gm",
    "ukraine": "ua",
    "ghana": "gh",
    "paraguay": "py",
    "senegal": "sn",
    "uruguay": "uy",
    "czech republic": "cz",
    "guernsey": "gg",
    "croatia": "hr",
    "nigeria": "ng",
    "ecuador": "ec",
    "colombia": "co",
    "mexico": "mx",
    "egypt": "eg",
    "angola": "ao",
    "french guiana": "gf",
    "japan": "jp",
    "burkina faso": "bf",
    "mozambique": "mz",
    "greece": "gr",
    "slovenia": "si",
    "guinea-bissau": "gw",
    "zimbabwe": "zw",
    "cameroon": "cm",
    "south africa": "za",
    "korea republic": "kr",
    "chad": "td",
    "congo dr": "cd",
    "austria": "at",
    "bulgaria": "bg",
    "turkiye": "tr",
    "new zealand": "nz",
    "georgia": "ge",
    "uzbekistan": "uz",
    "morocco": "ma",
    "bosnia and herzegovina": "ba",
    "poland": "pl",
    "australia": "au",
    "saudi arabia": "sa",
    "chile": "cl",
    "mali": "ml",
    "tanzania": "tz",
    "canada": "ca",
    "montenegro": "me",
    "zambia": "zm",
    "panama": "pa",
    "jersey": "je",
    "iceland": "is",
    "algeria": "dz",
    "curacao": "cw",
    "finland": "fi",
    "bermuda": "bm",
    "barbados": "bb",
    "congo": "cg",
    "grenada": "gd",
    "montserrat": "ms",
    "liberia": "lr",
    "jamaica": "jm",
    "lithuania": "lt",
    "afghanistan": "af",
    "malawi": "mw",
    "belize": "bz",
    "guadeloupe": "gp",
    "albania": "al",
    "somalia": "so",
    "guyana": "gy",
    "british virgin islands": "vg",
    "suriname": "sr",
    "gibraltar": "gi",
    "honduras": "hn",
    "mauritius": "mu",
    "great britain": "gb",
    "russia": "ru",
    "cyprus": "cy",
    "fiji": "fj",
    "thailand": "th",
    "hong kong": "hk",
    "latvia": "lv",
    "trinidad and tobago": "tt",
    "eritrea": "er",
    "north macedonia": "mk",
    "kosovo": "xk",
    "azerbaijan": "az",
    "luxembourg": "lu",
    "venezuela": "ve",
    "peru": "pe",
    "israel": "il",
    "moldova": "md",
    "estonia": "ee",
    "costa rica": "cr",
    "armenia": "am",
    "guinea": "gn",
    "comoros": "km",
    "kenya": "ke",
    "vanuatu": "vu",
    "malta": "mt",
    "iraq": "iq",
    "dominica": "dm",
    "reunion": "re",
    "cape verde islands": "cv",
    "romania": "ro",
    "liechtenstein": "li",
    "kazakhstan": "kz",
    "belarus": "by",
    "benin": "bj",
    "rwanda": "rw",
    "dominican republic": "do",
    "iran": "ir",
    "niger": "ne",
    "singapore": "sg",
    "burundi": "bi",
    "madagascar": "mg",
    "togo": "tg",
    "central african republic": "cf",
    "bolivia": "bo",
    "tajikistan": "tj",
    "martinique": "mq",
    "cuba": "cu",
    "china pr": "cn",
    "equatorial guinea": "gq",
    "gabon": "ga",
    "chinese taipei": "tw",
    "guatemala": "gt",
    "tunisia": "tn",
    "lebanon": "lb",
    "bahrain": "bh",
    "uganda": "ug",
    "oman": "om",
    "faroe islands": "fo",
    "jordan": "jo",
    "haiti": "ht",
    "africa": None,
    "syria": "sy",
    "st. lucia": "lc",
    "indonesia": "id",
    "ethiopia": "et",
    "philippines": "ph",
    "mauritania": "mr",
    "palestine": "ps",
    "libya": "ly",
    "malaysia": "my",
    "korea dpr": "kp",
    "nicaragua": "ni",
    "south sudan": "ss",
    "bonaire": "bq",
    "sao tome e principe": "st",
    "st. kitts and nevis": "kn",
    "el salvador": "sv",
    "new caledonia": "nc",
    "kyrgyzstan": "kg",
    "isle of man": "im",
    "lesotho": "ls",
    "united arab emirates": "ae",
    "andorra": "ad",
    "mongolia": "mn",
    "namibia": "na",
    "eswatini": "sz",
    "pakistan": "pk",
    "djibouti": "dj",
    "antigua and barbuda": "ag",
    "puerto rico": "pr",
    "cayman islands": "ky",
    "st. vincent and the grenadines": "vc",
}

def _fb_norm_country(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").strip().lower()


def _fb_twemoji_code_from_cc(cc: str) -> str:
    a, b = cc.upper()
    cp1 = 0x1F1E6 + (ord(a) - ord("A"))
    cp2 = 0x1F1E6 + (ord(b) - ord("A"))
    return "%x-%x" % (cp1, cp2)


@st.cache_data(show_spinner=False)
def fb_load_twemoji_png_by_code(code: str):
    try:
        url = "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/72x72/%s.png" % code
        r = requests.get(url, timeout=4)
        r.raise_for_status()
        return plt.imread(io.BytesIO(r.content))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fb_country_to_iso2_soft(name: str):
    n = _fb_norm_country(name)
    if not n:
        return None

    if n in _FB_COUNTRY_OVERRIDES:
        return _FB_COUNTRY_OVERRIDES[n]

    try:
        r = requests.get(
            "https://restcountries.com/v3.1/name/" + requests.utils.quote(n),
            params={"fields": "cca2,name"},
            timeout=4,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        cca2 = data[0].get("cca2")
        if isinstance(cca2, str) and len(cca2) == 2:
            return cca2.lower()
    except Exception:
        return None
    return None


def fb_birth_country_flag_image(birth_country):
    if not birth_country:
        return None
    norm = _fb_norm_country(birth_country)

    key = _FB_CC_MAP.get(norm)

    if key is None and norm in _FB_COUNTRY_OVERRIDES:
        cc = _FB_COUNTRY_OVERRIDES[norm]
        if cc is None:
            return None
        key = cc

    if key is None:
        iso2 = fb_country_to_iso2_soft(birth_country)
        if iso2:
            key = iso2

    if key is None:
        return None

    if key in _FB_TWEMOJI_SPECIAL:
        return fb_load_twemoji_png_by_code(_FB_TWEMOJI_SPECIAL[key])
    if key in _FB_SPECIAL_FLAG_URLS:
        return fb_load_remote_png(_FB_SPECIAL_FLAG_URLS[key])
    if isinstance(key, str) and len(key) == 2:
        return fb_load_twemoji_png_by_code(_fb_twemoji_code_from_cc(key))
    return None


# ---------------------------------------------------------
# 6) CREST / BADGE PIPELINE
# ---------------------------------------------------------

FB_BADGE_DIRS = [
    Path(__file__).resolve().parent / "badges",
    Path(__file__).resolve().parent / "crests",
]
for d in FB_BADGE_DIRS:
    d.mkdir(exist_ok=True)


def _fb_clean_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").lower()).strip("_")


@st.cache_data(show_spinner=False)
def fb_load_local_badge(team: str):
    key = _fb_clean_filename(team)
    if not key:
        return None
    for folder in FB_BADGE_DIRS:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = folder / ("%s%s" % (key, ext))
            if p.exists():
                try:
                    return plt.imread(str(p))
                except Exception:
                    continue
    return None


def fb_get_team_badge(row: pd.Series):
    team = str(row.get("Team", "")).strip()

    # 1) Local badge first
    img = fb_load_local_badge(team)
    if img is not None:
        return img

    # 2) FotMob crest fallback (team badge)
    crest = fb_load_fotmob_crest(team)
    if crest is not None:
        return crest

    # 3) Optional final fallback: birth-country flag (keep if you want)
    birth = row.get("Birth country") or row.get("Birth Country") or row.get("Nationality")
    return fb_birth_country_flag_image(birth)


# ---------------------------------------------------------
# 6B) FotMob crest fallback (same idea as your pro layout)
# ---------------------------------------------------------

try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS as _FB_FOTMOB_TEAM_URLS
except Exception:
    _FB_FOTMOB_TEAM_URLS = {}

def fb_get_fotmob_url(team: str) -> str:
    return (_FB_FOTMOB_TEAM_URLS.get(team) or "").strip()

def fb_fotmob_team_id_from_url(team_url: str) -> str:
    try:
        m = re.search(r"/teams/(\d+)/", str(team_url or ""))
        return m.group(1) if m else ""
    except Exception:
        return ""

def fb_fotmob_crest_url(team: str) -> str:
    team_url = fb_get_fotmob_url(team)
    tid = fb_fotmob_team_id_from_url(team_url)
    return "https://images.fotmob.com/image_resources/logo/teamlogo/%s.png" % tid if tid else ""

@st.cache_data(show_spinner=False)
def fb_load_fotmob_crest(team: str):
    url = fb_fotmob_crest_url(team)
    if not url:
        return None
    return fb_load_remote_png(url)


# ---------------------------------------------------------
# 6C) Badge size normaliser (make badges fit like flags)
# ---------------------------------------------------------

def fb_zoom_to_fit(img, target_px: int = 28) -> float:
    """
    Scale any badge/flag image so its largest dimension becomes ~target_px.
    This stops big club PNGs (e.g. Barcelona) from blowing up the layout.
    """
    try:
        h, w = img.shape[0], img.shape[1]
        m = max(h, w)
        if m <= 0:
            return 1.0
        return float(target_px) / float(m)
    except Exception:
        return 1.0


# ---------------------------------------------------------
# 7) FOOTER LINES
# ---------------------------------------------------------

def fb_footer_lines_for_metric(metric_label: str, show_ls: bool):
    if metric_label == "Impact Score" or metric_label.startswith("Impact Score"):
        return [
            "Impact Score (FB): combines Aerial, Ground, Retention, Carrying, Playmaking and Chance Creation.",
            "Adjusted for minutes played and team context vs league.",
            "Displayed 0–100 vs the full selected pool "
            + ("(league strength applied)." if show_ls else "(no league-strength adjustment)."),
        ]
    if metric_label.startswith("Complete Score"):
        return [
            "Complete Score (FB): weighted blend of duels, passing, carrying, progression, chance creation and positioning.",
            "Displayed 0–100 vs the full selected pool "
            + ("(league strength applied)." if show_ls else "(no league-strength adjustment)."),
        ]
    if metric_label.startswith("Custom Combo"):
        return [
            "Custom Combo (FB): equal-weight blend of selected base scores (Aerial/Ground/Retention/Carrying/Playmaking/Chance Creation).",
            "Displayed vs the selected pool "
            + ("(league strength applied)." if show_ls else "(no league-strength adjustment)."),
        ]
    return [
        "%s (FB): ranks this metric only." % metric_label,
        "Displayed 0–100 vs the full selected pool "
        + ("(league strength applied)." if show_ls else "(no league-strength adjustment)."),
    ]


# ---------------------------------------------------------
# 8) RANKING IMAGE (Standard + 1920×1080)
# ---------------------------------------------------------

def fb_format_value(v):
    if v is None:
        return "—"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if np.isnan(v):
        return "—"

    # ✅ raw mode shows 2dp actual value; composite keeps old formatting
    if rank_mode == "Raw metric (any numeric column)":
        return "%.2f" % v

    av = abs(v)
    if av >= 100:
        return "%.0f" % v
    if av >= 10:
        return "%.1f" % v
    if av >= 1:
        return "%.2f" % v
    return "%.3f" % v


def fb_make_ranking_image(
    df_show: pd.DataFrame,
    metric_col: str,
    value_label_col: str,
    metric_label: str,
    title_lines,
    brand_logo_url=None,
    show_ls: bool = False,
    show_age: bool = False,
    highlight_players=None,
    export_mode: str = "Standard (auto)",
    theme: str = "Light",
    custom_footer_text: str = None,
) -> bytes:

    df_top = df_show.head(10).copy()
    if df_top.empty:
        return b""

    hi_set = set()
    if highlight_players:
        hi_set = {str(x).strip().lower() for x in highlight_players if str(x).strip()}

    def is_hi(row: pd.Series) -> bool:
        return str(row.get("Player", "")).strip().lower() in hi_set

    # theme palette
    if theme == "Dark":
        BG = "#0a0f1c"
        ROW_A, ROW_B = "#0f1628", "#0b1222"
        TXT, SUB, FOOT = "#ffffff", "#b8c0cf", "#9aa6bd"
        DIV = "#23304a"
        BAR_BG, BAR_FG = "#1a2540", "#6b7cff"
        RANK_BG, RANK_EDGE = "#111a2e", "#2b3a5a"
        HILITE, HILITE_EDGE = "#f6d46b", "#d2a100"
    else:
        BG = "#ffffff"
        ROW_A, ROW_B = "#f7f7f7", "#ffffff"
        TXT, SUB, FOOT = "#111111", "#777777", "#9b9b9b"
        DIV = "#e2e2e2"
        BAR_BG, BAR_FG = "#e1e1e1", "#bfbfbf"
        RANK_BG, RANK_EDGE = "#f3f3f3", "#c0c0c0"
        HILITE, HILITE_EDGE = "#f6d46b", "#d2a100"

    scores = pd.to_numeric(df_top[metric_col], errors="coerce")
    max_score = float(scores.max()) if scores.notna().any() else 1.0

    # Footer lines (auto vs custom)
    if custom_footer_text:
        footer_lines = [ln.strip() for ln in custom_footer_text.split("\n") if ln.strip()]
    else:
        footer_lines = fb_footer_lines_for_metric(metric_label, show_ls)

    # =====================================================
    # 1920×1080 banner
    # =====================================================
    if export_mode == "1920×1080 (banner)":
        DPI = 100
        fig = plt.figure(figsize=(1920.0 / DPI, 1080.0 / DPI), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, color=BG, zorder=0))

        LEFT, RIGHT = 0.045, 0.955

        t1 = title_lines[0].upper() if len(title_lines) > 0 else ""
        t2 = title_lines[1].upper() if len(title_lines) > 1 else ""
        t3 = title_lines[2].upper() if len(title_lines) > 2 else ""

        ax.text(LEFT, 0.972, t1, fontsize=48, fontweight="bold", color=TXT, ha="left", va="top")
        ax.text(LEFT, 0.912, t2, fontsize=34, fontweight="bold", color=TXT, ha="left", va="top")
        ax.text(LEFT, 0.870, t3, fontsize=20, color=SUB, ha="left", va="top")

        header_div_y = 0.835
        ax.plot([LEFT, RIGHT], [header_div_y, header_div_y], color=DIV, lw=2.2)

        footer_div_y = 0.040
        ax.plot([LEFT, RIGHT], [footer_div_y, footer_div_y], color=DIV, lw=2.2)

        for i, line in enumerate(footer_lines):
            ax.text(
                LEFT,
                footer_div_y - 0.018 - i * 0.024,
                line,
                fontsize=13,
                color=FOOT,
                ha="left",
                va="top",
                zorder=10,
            )

        ROW_TOP = header_div_y - 0.022
        ROW_BOT = footer_div_y + 0.010
        row_gap = (ROW_TOP - ROW_BOT) / 10.0
        row_h   = row_gap * 0.99

        RANK_X  = LEFT + 0.024
        CREST_X = LEFT + 0.112
        NAME_X  = LEFT + 0.190

        BAR_L   = LEFT + 0.63
        BAR_R   = RIGHT - 0.155
        BAR_W   = BAR_R - BAR_L
        BAR_H   = row_h * 0.26

        VAL_X   = RIGHT - 0.030

        NAME_FS = 28
        TEAM_FS = 19
        NAME_DY = row_h * 0.20
        TEAM_DY = row_h * 0.26

        for i, (_, row) in enumerate(df_top.iterrows()):
            y = ROW_TOP - (i + 0.5) * row_gap

            ax.add_patch(Rectangle(
                (LEFT, y - row_h / 2),
                RIGHT - LEFT,
                row_h,
                color=(ROW_A if i % 2 == 0 else ROW_B),
                zorder=1,
            ))

            if is_hi(row):
                ax.add_patch(Rectangle(
                    (LEFT, y - row_h / 2),
                    RIGHT - LEFT,
                    row_h,
                    color=HILITE,
                    alpha=0.22,
                    zorder=2,
                ))
                ax.add_patch(Rectangle(
                    (LEFT, y - row_h / 2),
                    RIGHT - LEFT,
                    row_h,
                    fill=False,
                    edgecolor=HILITE_EDGE,
                    lw=2.2,
                    zorder=3,
                ))

            ax.scatter(
                [RANK_X], [y],
                s=1320,
                facecolor=RANK_BG,
                edgecolor=(HILITE_EDGE if is_hi(row) else RANK_EDGE),
                linewidths=2.2,
                zorder=4,
            )
            ax.text(
                RANK_X, y, str(i + 1),
                fontsize=16, fontweight="bold", color=TXT,
                ha="center", va="center", zorder=5
            )

            badge = fb_get_team_badge(row)
            if badge is not None:
                z = fb_zoom_to_fit(badge, target_px=52)
                ax.add_artist(AnnotationBbox(
                    OffsetImage(badge, zoom=z),
                    (CREST_X, y),
                    frameon=False,
                    zorder=5,
                ))

            player = str(row.get("Player", "")).upper()
            team   = str(row.get("Team", ""))
            league = str(row.get("League", ""))

            ax.text(
                NAME_X, y + NAME_DY, player,
                fontsize=NAME_FS, fontweight="bold", color=TXT,
                ha="left", va="center", zorder=6
            )
            ax.text(
                NAME_X, y - TEAM_DY, "%s (%s)" % (team, league),
                fontsize=TEAM_FS, color=SUB,
                ha="left", va="center", zorder=6
            )

            v_bar = float(row[metric_col]) if pd.notna(row[metric_col]) else 0.0
            frac  = (v_bar / max_score) if max_score else 0.0
            frac  = max(0.0, min(1.0, frac))

            ax.add_patch(Rectangle((BAR_L, y - BAR_H/2), BAR_W, BAR_H, color=BAR_BG, zorder=2))
            ax.add_patch(Rectangle((BAR_L, y - BAR_H/2), BAR_W * frac, BAR_H, color=BAR_FG, zorder=3))

            v_lab = row.get(value_label_col)
            ax.text(
                VAL_X, y, fb_format_value(v_lab),
                fontsize=29, fontweight="bold", color=TXT,
                ha="right", va="center", zorder=6
            )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # =====================================================
    # Standard (auto height) – fixed canvas, no tight bbox
    # =====================================================
    N        = len(df_top)
    ROW_H    = 0.82
    HEADER_H = 1.70
    FOOT_H   = 0.70
    TOTAL_H  = HEADER_H + N * ROW_H + FOOT_H

    fig = plt.figure(figsize=(8.3, TOTAL_H), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])        # full-canvas axes
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, TOTAL_H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1.0, TOTAL_H, color=BG, zorder=0))

    t1 = title_lines[0].upper() if len(title_lines) > 0 else ""
    t2 = title_lines[1].upper() if len(title_lines) > 1 else ""
    t3 = title_lines[2].upper() if len(title_lines) > 2 else ""
    title_y = TOTAL_H - 0.25
    ax.text(0.04, title_y,        t1, fontsize=19, fontweight="bold", color=TXT, ha="left", va="top")
    ax.text(0.04, title_y - 0.34, t2, fontsize=14, fontweight="bold", color=TXT, ha="left", va="top")
    ax.text(0.04, title_y - 0.62, t3, fontsize=11, color=SUB, ha="left", va="top")

    base_y = TOTAL_H - HEADER_H
    ax.plot([0.04, 0.96], [base_y + ROW_H/2 + 0.02]*2, color=DIV, lw=1.1, zorder=2)

    LEFT, RIGHT = 0.04, 0.96
    BAR_L, BAR_R = 0.66, 0.82
    BAR_W = BAR_R - BAR_L
    BAR_H = 0.14
    VAL_X = 0.94
    crest_x = 0.14

    for i, (_, row) in enumerate(df_top.iterrows()):
        y = base_y - i * ROW_H

        ax.add_patch(Rectangle((LEFT, y - ROW_H/2), RIGHT - LEFT, ROW_H,
                               color=(ROW_A if i % 2 == 0 else ROW_B), zorder=1))

        if is_hi(row):
            ax.add_patch(Rectangle((LEFT, y - ROW_H/2), RIGHT - LEFT, ROW_H,
                                   color=HILITE, alpha=0.25, zorder=2))
            ax.add_patch(Rectangle((LEFT, y - ROW_H/2), RIGHT - LEFT, ROW_H,
                                   fill=False, edgecolor=HILITE_EDGE, lw=1.3, zorder=3))

        ax.scatter([0.07], [y], s=520, facecolor=RANK_BG,
                   edgecolor=(HILITE_EDGE if is_hi(row) else RANK_EDGE),
                   linewidths=1.2, zorder=4)
        ax.text(0.07, y, str(i+1), fontsize=10, fontweight="bold",
                color=TXT, ha="center", va="center", zorder=5)

        badge = fb_get_team_badge(row)
        if badge is not None:
            z = fb_zoom_to_fit(badge, target_px=40)
            ax.add_artist(AnnotationBbox(OffsetImage(badge, zoom=z),
                                         (crest_x, y), frameon=False, zorder=5))

        ax.text(0.21, y + 0.12, str(row.get("Player", "")).upper(),
                fontsize=16, fontweight="bold", color=TXT, ha="left", va="center", zorder=5)

        team = str(row.get("Team", ""))
        league = str(row.get("League", ""))
        ax.text(0.21, y - 0.10, "%s (%s)" % (team, league),
                fontsize=12, color=SUB, ha="left", va="center", zorder=5)

        v_bar = float(row[metric_col]) if pd.notna(row[metric_col]) else 0.0
        frac = (v_bar / max_score) if max_score else 0.0
        frac = max(0.0, min(1.0, frac))

        ax.add_patch(Rectangle((BAR_L, y - BAR_H/2), BAR_W, BAR_H, color=BAR_BG, zorder=2))
        ax.add_patch(Rectangle((BAR_L, y - BAR_H/2), BAR_W * frac, BAR_H, color=BAR_FG, zorder=3))

        v_lab = row.get(value_label_col)
        ax.text(VAL_X, y, fb_format_value(v_lab),
                fontsize=16, fontweight="bold", color=TXT, ha="right", va="center", zorder=6)

    ax.plot([LEFT, RIGHT], [0.82]*2, color=DIV, lw=0.9, zorder=2)
    for j, line in enumerate(footer_lines):
        ax.text(LEFT, 0.62 - j*0.18, line, fontsize=9.5, color=FOOT, ha="left", va="top", zorder=4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------
# 9) STREAMLIT OUTPUT – IMAGE
# ---------------------------------------------------------

st.subheader("🖼 Exportable CIES-style ranking image – Fullbacks")

default_t1_fb = "TOP FULLBACKS"
default_t2_fb = str(metric_label_for_image).upper()
default_t3_fb = "PERFORMANCE INDEX  |  Wyscout"

t1_fb = st.text_input("Title line 1 (FB)", default_t1_fb, key="fb_title1_%s" % selected_file)
t2_fb = st.text_input("Title line 2 (FB)", default_t2_fb, key="fb_title2_%s" % selected_file)
t3_fb = st.text_input("Title line 3 (FB)", default_t3_fb, key="fb_title3_%s" % selected_file)

use_custom_footer_fb = st.checkbox(
    "Custom footer text (FB)",
    value=False,
    key="fb_use_custom_footer_%s" % selected_file,
    help="Override the default footer description for this image.",
)

custom_footer_text_fb = ""
if use_custom_footer_fb:
    custom_footer_text_fb = st.text_area(
        "Footer text (multi-line, optional) – FB",
        value="",
        key="fb_footer_text_%s" % selected_file,
        help="Each line here will appear as a separate footer line under the graphic.",
    )

export_mode_fb = st.selectbox(
    "Export format (FB)",
    ["Standard (auto)", "1920×1080 (banner)"],
    index=0,
    key="fb_export_mode_%s" % selected_file,
)

fb_brand_logo_url = "https://image.pitchbook.com/1xOUzrEhnsKrJbNbN8Asf3LND2u1605464042293_200x200"

img_bytes_fb = fb_make_ranking_image(
    df_show=df_display,
    metric_col=display_metric_col,
    value_label_col=value_label_col,
    metric_label=str(metric_label_for_image),
    title_lines=[t1_fb, t2_fb, t3_fb],
    brand_logo_url=fb_brand_logo_url,
    show_ls=display_with_league_strength,
    show_age=show_age_in_image,
    highlight_players=(highlight_player_names if enable_highlight_players else None),
    export_mode=export_mode_fb,
    theme=image_theme,
    custom_footer_text=custom_footer_text_fb if use_custom_footer_fb else None,
)

if img_bytes_fb:
    st.image(img_bytes_fb, use_column_width=True)
    st.download_button("Download PNG (FB)", data=img_bytes_fb, file_name="fb_ranking.png", mime="image/png")
else:
    st.info("No data to generate image for fullbacks.")



# ----------------- ROLE SCORING (tables) -----------------
def compute_weighted_role_score(df_in: pd.DataFrame, metrics: dict, beta: float, league_weighting: bool) -> pd.Series:
    total_w = sum(metrics.values()) if metrics else 1.0
    wsum = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} Percentile"
        if col in df_in.columns:
            wsum += df_in[col].values * w
    player_score = wsum / total_w  # 0..100
    if league_weighting:
        league_scaled = df_in["League Strength"].fillna(50.0)  # already 0..100
        return (1 - beta) * player_score + beta * league_scaled.values
    return player_score

use_league_weighting = st.session_state[f"fb_use_lw_{selected_file}"]
beta = st.session_state[f"fb_beta_{selected_file}"]

for role_name, role_def in ROLES.items():
    df_f[f"{role_name} Score"] = compute_weighted_role_score(
        df_f, role_def["metrics"], beta=beta, league_weighting=use_league_weighting
    )

# ----------------- THRESHOLDS -----------------
enable_min_perf = st.session_state[f"fb_enable_min_perf_{selected_file}"]
sel_metrics = st.session_state[f"fb_sel_metrics_{selected_file}"]
min_pct = st.session_state[f"fb_min_pct_{selected_file}"]

if enable_min_perf and sel_metrics:
    keep_mask = np.ones(len(df_f), dtype=bool)
    for m in sel_metrics:
        pct_col = f"{m} Percentile"
        if pct_col in df_f.columns:
            keep_mask &= (df_f[pct_col] >= min_pct)
    df_f = df_f[keep_mask]
    if df_f.empty:
        st.warning("No players meet the minimum performance thresholds. Loosen thresholds.")
        st.stop()

# ----------------- HELPERS -----------------
round_to = st.session_state[f"fb_round_to_{selected_file}"]

def fmt_cols(df_in: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df_in.copy()
    out[score_col] = out[score_col].round(round_to).astype(int if round_to == 0 else float)
    cols = ["Player","Team","League","Position","Age","Contract expires","League Strength", score_col]
    return out[cols]

def top_table(df_in: pd.DataFrame, role: str, head_n: int) -> pd.DataFrame:
    col = f"{role} Score"
    ranked = df_in.dropna(subset=[col]).sort_values(col, ascending=False)
    ranked = fmt_cols(ranked, col).head(head_n).reset_index(drop=True)
    ranked.index = np.arange(1, len(ranked)+1)
    return ranked

def filtered_view(df_in: pd.DataFrame, *, age_max=None, contract_year=None, value_max=None):
    t = df_in.copy()
    if age_max is not None:
        t = t[t["Age"] <= age_max]
    if contract_year is not None:
        years = pd.to_datetime(t["Contract expires"], errors="coerce").dt.year
        t = t[years <= contract_year]
    if value_max is not None:
        t = t[t["Market value"] <= value_max]
    return t

# ----------------- TABS (tables) -----------------
top_n = int(st.session_state[f"fb_topn_{selected_file}"])
value_band_max = st.session_state[f"fb_value_band_{selected_file}"]

tabs = st.tabs([
    "Overall Top N", "U23 Top N", "Expiring Contracts",
    "Value Band (≤ max €)", "Pro Layout" ])  # 👈 new tab
for role, role_def in ROLES.items():
    with tabs[0]:
        st.subheader(f"{role} — Overall Top {top_n}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(df_f, role, top_n), use_container_width=True)
        st.divider()
    with tabs[1]:
        u23_cutoff = st.number_input(
            f"{role} — U23 cutoff", min_value=16, max_value=30, value=23, step=1, key=f"fb_u23_{role}_{selected_file}"
        )
        st.subheader(f"{role} — U{u23_cutoff} Top {top_n}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, age_max=u23_cutoff), role, top_n), use_container_width=True)
        st.divider()
    with tabs[2]:
        exp_year = st.number_input(
            f"{role} — Expiring by year", min_value=2024, max_value=2030,
            value=st.session_state[f"fb_cutoff_{selected_file}"], step=1,
            key=f"fb_exp_{role}_{selected_file}"
        )
        st.subheader(f"{role} — Contracts expiring ≤ {exp_year}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, contract_year=exp_year), role, top_n), use_container_width=True)
        st.divider()
    with tabs[3]:
        v_max = st.number_input(
            f"{role} — Max value (€)", min_value=0, value=value_band_max, step=100_000,
            key=f"fb_val_{role}_{selected_file}"
        )
        st.subheader(f"{role} — Value band ≤ €{v_max:,.0f}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, value_max=v_max), role, top_n), use_container_width=True)
        st.divider()

## ----------------- PRO LAYOUT TAB (tiles) — FULLBACKS -----------------
import re as _re
import unicodedata
import pandas as pd
import numpy as np
import streamlit as st

# ✅ Needed for FotMob squad lookup + URL quoting
import requests
from urllib.parse import quote

# ✅ Needed for override uploads
import base64
import io
import os


# ==========================================================
# ✅ Shared constants / helpers
# ==========================================================
PLAYER_PHOTO_OVERRIDES_JSON = "player_photo_overrides.json"

def load_local_photo_overrides(path: str) -> dict:
    try:
        import json
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

# --- FotMob team url mapping (your project may already provide this) ---
try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS  # must exist in your repo (same as CB page uses)
except Exception:
    FOTMOB_TEAM_URLS = {}

def get_fotmob_url(team: str) -> str:
    return (FOTMOB_TEAM_URLS.get(team) or "").strip()

def _norm(s: str) -> str:
    if not s:
        return ""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii").strip().lower()

def _fotmob_team_id_from_url(team_url: str) -> str:
    m = _re.search(r"/teams/(\d+)/", str(team_url or ""))
    return m.group(1) if m else ""

def _fotmob_crest_url(team_url: str) -> str:
    tid = _fotmob_team_id_from_url(team_url)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png" if tid else ""

def _player_surname(player: str) -> str:
    p = (player or "").strip()
    if not p:
        return ""
    # "Surname, Name" → surname else last token
    if "," in p:
        return p.split(",", 1)[0].strip()
    parts = p.split()
    return parts[-1].strip() if parts else ""


# ==========================================================
# ✅ URL photo system (same as FB/CM fixed version)
# ==========================================================
PLAYER_PHOTO_OVERRIDES_JSON = "player_photo_overrides.json"

def load_local_photo_overrides(path: str) -> dict:
    try:
        import json
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS
except Exception:
    FOTMOB_TEAM_URLS = {}

def get_fotmob_url(team: str) -> str:
    return (FOTMOB_TEAM_URLS.get(team) or "").strip()

def _fotmob_team_id_from_url(team_url: str) -> str:
    m = _re.search(r"/teams/(\d+)/", str(team_url or ""))
    return m.group(1) if m else ""

def _fotmob_crest_url(team_url: str) -> str:
    tid = _fotmob_team_id_from_url(team_url)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png" if tid else ""

def _player_surname(player: str) -> str:
    p = (player or "").strip()
    if not p:
        return ""
    if "," in p:
        return p.split(",", 1)[0].strip()
    parts = p.split()
    return parts[-1].strip() if parts else ""

# ✅ Accent tolerant slug
def _slug_name(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()

    repl = {
        "ø":"o","œ":"oe","æ":"ae","å":"a","ä":"a","ö":"o","ü":"u",
        "ß":"ss","ł":"l","đ":"d","ð":"d","þ":"th","ç":"c",
        "ş":"s","ğ":"g","ı":"i",
    }
    for k, v in repl.items():
        s = s.replace(k, v)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _re.sub(r"[^a-z0-9]+", "", s)
    return s

# ✅ fuzzy helper
from difflib import SequenceMatcher
def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _fotmob_team_squad(team_id: str) -> list[dict]:
    cache = st.session_state.setdefault("_fotmob_team_squad_cache", {})
    if team_id in cache:
        return cache[team_id] or []

    squad: list[dict] = []
    try:
        url = f"https://www.fotmob.com/api/teams?id={team_id}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            data = r.json() or {}
            raw_squad = data.get("squad", None)

            if isinstance(raw_squad, list):
                for section in raw_squad:
                    members = section.get("members") or section.get("players") or []
                    if isinstance(members, list):
                        squad.extend([m for m in members if isinstance(m, dict)])

            elif isinstance(raw_squad, dict):
                for k in ("members", "players"):
                    members = raw_squad.get(k)
                    if isinstance(members, list):
                        squad.extend([m for m in members if isinstance(m, dict)])

                nested = raw_squad.get("squad")
                if isinstance(nested, list):
                    for section in nested:
                        members = section.get("members") or section.get("players") or []
                        if isinstance(members, list):
                            squad.extend([m for m in members if isinstance(m, dict)])
    except Exception:
        squad = []

    cache[team_id] = squad
    return squad

def resolve_player_photo(player: str,
                         team: str,
                         league: str,
                         key_id: str,
                         session_photo_map: dict,
                         global_overrides: dict) -> str:
    if session_photo_map.get(key_id):
        return session_photo_map[key_id]

    if global_overrides.get(key_id):
        return global_overrides[key_id]

    team_url = get_fotmob_url(team)
    tid = _fotmob_team_id_from_url(team_url)
    if tid:
        squad = _fotmob_team_squad(tid)

        target_surname = _slug_name(_player_surname(player))
        target_full    = _slug_name(player)

        best_id = ""

        # ---- exact surname match first ----
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

        # ---- exact full-name contains ----
        if not best_id and target_full:
            for m in squad:
                name = m.get("name") or m.get("playerName") or ""
                pid = m.get("id") or m.get("playerId") or m.get("primaryId") or ""
                if not pid:
                    continue

                if target_full in _slug_name(name):
                    best_id = str(pid)
                    break

        # ---- FUZZY fallback (only if still not found) ----
        if not best_id and target_surname:
            best_score = 0.0
            best_pid = ""

            for m in squad:
                name = m.get("name") or m.get("playerName") or ""
                pid = m.get("id") or m.get("playerId") or m.get("primaryId") or ""
                if not pid:
                    continue

                sn = _slug_name(_player_surname(name))
                sc = _similar(sn, target_surname)

                if sc > best_score:
                    best_score = sc
                    best_pid = str(pid)

            if best_score >= 0.86:   # safe threshold
                best_id = best_pid

        if best_id and str(best_id).isdigit():
            url = f"https://images.fotmob.com/image_resources/playerimages/{best_id}.png"
            session_photo_map[key_id] = url
            return url

    return "https://i.redd.it/43axcjdu59nd1.jpeg"



# ==========================================================
# ✅ Metric helpers (raw value + percentile badge like CB)
# ==========================================================
def _available_metric_pairs(df_view: pd.DataFrame, pairs: list[tuple[str, str]]):
    cols = set(df_view.columns)
    out = []
    for lab, met in pairs:
        if met in cols or f"{met} Percentile" in cols:
            out.append((lab, met))
    return out

def _metric_pct(row: pd.Series, met: str):
    col = f"{met} Percentile"
    if col in row.index and not pd.isna(row[col]):
        try:
            return float(row[col])
        except Exception:
            return np.nan
    return np.nan

def _metric_val(row: pd.Series, met: str):
    if met in row.index and not pd.isna(row[met]):
        try:
            return float(row[met])
        except Exception:
            return row[met]
    return np.nan


# ----------------- helpers (kept as provided) -----------------
def _pro_rating_color(v: float) -> str:
    v = float(v)
    COLORS = [
        (85, "#2E6114"),
        (75, "#5C9E2E"),
        (66, "#7FBC41"),
        (55, "#A7D763"),
        (41, "#F6D645"),
        (25, "#D77A2E"),
        (0,  "#C63733"),
    ]
    for threshold, color in COLORS:
        if v >= threshold:
            return color
    return COLORS[-1][1]

def _pro_show99(x) -> int:
    try:
        return max(0, min(99, int(float(x))))
    except Exception:
        return 0

def _fmt2(n: int) -> str:
    try:
        return f"{int(n):02d}"
    except Exception:
        return "00"

_POS_COLORS = {
    "CF":"#6EA8FF","LWF":"#6EA8FF","LW":"#6EA8FF","LAMF":"#6EA8FF","RW":"#6EA8FF","RWF":"#6EA8FF","RAMF":"#6EA8FF",
    "AMF":"#7FE28A","LCMF":"#5FD37A","RCMF":"#5FD37A","RDMF":"#31B56B","LDMF":"#31B56B","DMF":"#31B56B",
    "LWB":"#FFD34D","RWB":"#FFD34D","LB":"#FF9A3C","RB":"#FF9A3C","RCB":"#D1763A","CB":"#D1763A","LCB":"#D1763A",
}
def _pro_chip_color(p: str) -> str:
    return _POS_COLORS.get(str(p).strip().upper(), "#2d3550")

TWEMOJI_SPECIAL = {
    "eng":"1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",
    "sct":"1f3f4-e0067-e0062-e0073-e0063-e0074-e007f",
    "wls":"1f3f4-e0067-e0062-e0077-e006c-e0073-e007f",
}

# (country map kept as in your snippets; unchanged)
COUNTRY_TO_CC = {
    "united kingdom":"gb","great britain":"gb","northern ireland":"nir","england":"eng","scotland":"sct","wales":"wls",
    "ireland":"ie","republic of ireland":"ie","spain":"es","france":"fr","germany":"de","italy":"it","portugal":"pt",
    "netherlands":"nl","belgium":"be","austria":"at","switzerland":"ch","denmark":"dk","sweden":"se","norway":"no",
    "finland":"fi","iceland":"is","poland":"pl","czech republic":"cz","czechia":"cz","slovakia":"sk","slovenia":"si",
    "croatia":"hr","serbia":"rs","bosnia and herzegovina":"ba","bosnia":"ba","montenegro":"me","kosovo":"xk","albania":"al",
    "greece":"gr","hungary":"hu","romania":"ro","bulgaria":"bg","russia":"ru","ukraine":"ua","georgia":"ge",
    "kazakhstan":"kz","azerbaijan":"az","armenia":"am","turkey":"tr","cyprus":"cy","luxembourg":"lu","andorra":"ad",
    "monaco":"mc","san marino":"sm","malta":"mt","moldova":"md","north macedonia":"mk","macedonia":"mk","estonia":"ee",
    "latvia":"lv","lithuania":"lt",
    "qatar":"qa","saudi arabia":"sa","uae":"ae","united arab emirates":"ae","israel":"il","japan":"jp","korea":"kr",
    "south korea":"kr","korea republic":"kr","china":"cn",
    "algeria":"dz","angola":"ao","benin":"bj","botswana":"bw","burkina faso":"bf","burundi":"bi","cabo verde":"cv",
    "cape verde":"cv","cameroon":"cm","central african republic":"cf","car":"cf","chad":"td","comoros":"km",
    "congo":"cg","republic of the congo":"cg","congo brazzaville":"cg",
    "dr congo":"cd","drc":"cd","democratic republic of the congo":"cd","congo kinshasa":"cd",
    "djibouti":"dj","egypt":"eg","equatorial guinea":"gq","eritrea":"er","eswatini":"sz","swaziland":"sz",
    "ethiopia":"et","gabon":"ga","gambia":"gm","ghana":"gh","guinea":"gn","guinea-bissau":"gw","guinea bissau":"gw",
    "ivory coast":"ci","cote d'ivoire":"ci","cote divoire":"ci","cote d ivoire":"ci","côte d’ivoire":"ci","côte d'ivoire":"ci",
    "kenya":"ke","lesotho":"ls","liberia":"lr","libya":"ly","madagascar":"mg","malawi":"mw","mali":"ml","mauritania":"mr",
    "mauritius":"mu","morocco":"ma","mozambique":"mz","namibia":"na","niger":"ne","nigeria":"ng","rwanda":"rw",
    "sao tome and principe":"st","sao tome":"st","são tomé and príncipe":"st","são tomé":"st","sao tome & principe":"st",
    "senegal":"sn","seychelles":"sc","sierra leone":"sl","somalia":"so","south africa":"za","south sudan":"ss","sudan":"sd",
    "tanzania":"tz","united republic of tanzania":"tz","togo":"tg","tunisia":"tn","uganda":"ug","zambia":"zm","zimbabwe":"zw",
    "western sahara":"eh","réunion":"re","reunion":"re","mayotte":"yt",
    "maroc":"ma","algerie":"dz","tunis":"tn","egypte":"eg","cameroun":"cm","cote d’ivoire":"ci","cote-d-ivoire":"ci",
    "somaliland":"so","ethiopie":"et",
    "eswatini (swaziland)":"sz","swaziland (eswatini)":"sz",
    "congo-brazzaville":"cg","congo-kinshasa":"cd","gbissau":"gw",
    "brazil":"br","argentina":"ar","uruguay":"uy","chile":"cl","colombia":"co","peru":"pe","ecuador":"ec","paraguay":"py",
    "bolivia":"bo","mexico":"mx","canada":"ca","united states":"us","usa":"us",
    "australia":"au","new zealand":"nz",
    "palestine":"ps","state of palestine":"ps","hong kong":"hk","macau":"mo","macao":"mo","curacao":"cw","curaçao":"cw",
    "cape verde islands":"cv",
}

def _cc_to_twemoji(cc: str) -> str | None:
    if not cc or len(cc) != 2:
        return None
    a, b = cc.upper()
    cp1 = 0x1F1E6 + (ord(a) - ord('A'))
    cp2 = 0x1F1E6 + (ord(b) - ord('A'))
    return f"{cp1:04x}-{cp2:04x}"

def _flag_html(country_name: str) -> str:
    if not country_name:
        return "<span class='chip'>—</span>"
    n = _norm(country_name)
    cc = COUNTRY_TO_CC.get(n, "")
    if not cc:
        return "<span class='chip'>—</span>"
    if cc in TWEMOJI_SPECIAL:
        code = TWEMOJI_SPECIAL[cc]
        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
        return f"<span class='flagchip'><img src='{src}' alt='{country_name}'></span>"
    code = _cc_to_twemoji(cc) if len(cc) == 2 else None
    if code:
        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
        return f"<span class='flagchip'><img src='{src}' alt='{country_name}'></span>"
    return f"<span class='chip'>{cc.upper()}</span>"

def _get_foot(row) -> str:
    for col in ("Foot","Preferred foot","Preferred Foot"):
        if col in row.index:
            val = row[col]
            try:
                import pandas as _pd
                if _pd.isna(val):
                    continue
            except Exception:
                pass
            if isinstance(val, str):
                s = val.strip()
                if s and s.lower() not in {"nan","none","null"}:
                    return s
            else:
                s = str(val).strip()
                if s and s.lower() not in {"nan","none","null"}:
                    return s
    return ""

# Label → (column_name, pretty_label)
_FB_ROLE_MAP = [
    ("Build Up FB Score", "Build Up FB"),
    ("Attacking FB Score", "Attacking FB"),
    ("Defensive FB Score", "Defensive FB"),
    ("Wide Creator Score", "Wide Creator"),
    ("Wide Carrier Score", "Wide Carrier"),
    ("PL Profile Score", "PL Profile"),
]


def _fmt_mv_gbp(v) -> str:
    """Formats market value to £k/£m/£bn. Handles values stored as full £ or as 'millions'."""
    try:
        x = float(v)
    except Exception:
        return "—"

    # handle NaN / <=0
    try:
        if np.isnan(x) or x <= 0:
            return "—"
    except Exception:
        if x <= 0:
            return "—"

    # If stored as millions (e.g., 12.5 => £12.5m)
    if x < 1_000:
        x *= 1_000_000

    if x >= 1_000_000_000:
        return f"£{x/1_000_000_000:.1f}bn".replace(".0", "")
    if x >= 1_000_000:
        return f"£{x/1_000_000:.1f}m".replace(".0", "")
    if x >= 1_000:
        return f"£{x/1_000:.0f}k"
    return f"£{x:.0f}"



def render_pro_layout_fb(df_view: pd.DataFrame, top_n: int = 20):
    # ---- CSS (CB-style + metrics raw value + badge) ----
    st.markdown("""
    <style>
    html, body, .block-container *{
      -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale; text-rendering:optimizeLegibility;
      font-feature-settings:"liga","kern","tnum"; font-variant-numeric:tabular-nums;
    }
    :root { --bg:#0c0e13; --card:#141823; --soft:#1e2533; }

    .pro-wrap{ display:flex; justify-content:center; }
    .pro-card{
      position:relative; width:min(420px,96%); display:grid; grid-template-columns:96px 1fr 48px; gap:12px; align-items:start;
      background:var(--card); border:1px solid rgba(255,255,255,.06); border-radius:20px; padding:16px; margin-bottom:12px;
      box-shadow:inset 0 1px 0 rgba(255,255,255,.03), 0 6px 24px rgba(0,0,0,.35);
    }

    .pro-avatar{ width:96px; height:96px; border-radius:12px; border:1px solid #2a3145; overflow:hidden; background:#0b0d12; }
    .pro-avatar img{ width:100%; height:100%; object-fit:cover; image-rendering:auto; transform:translateZ(0); }

    .flagchip{ display:inline-flex; align-items:center; gap:6px; background:transparent; border:none; padding:0; height:auto;}
    .flagchip img{ width:26px; height:18px; border-radius:2px; display:block; }

    .chip{ background:transparent; color:#a6a6a6; border:none; padding:0; border-radius:0; font-size:15px; line-height:18px; opacity:.92; }
    .row{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:2px 0; }
    .leftrow1{ margin-top:6px; } .leftrow-foot{ margin-top:2px; } .leftrow-contract{ margin-top:10px; }

    .pill{ padding:2px 6px; min-width:36px; border-radius:6px; font-weight:700; font-size:18px; line-height:1; color:#0b0d12; text-align:center; display:inline-block; box-shadow:none; }

    .name{ font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; letter-spacing:.2px; line-height:1.15; }
    .sub{ color:#a8b3cf; font-size:15px; opacity:.9; }

    .posrow{ margin-top:13.5px; }
    .postext{ font-weight:600; font-size:14.5px; letter-spacing:.2px; margin-right:11px; }

    .rank{ position:absolute; top:10.5px; right:14px; color:#b7bfe1; font-weight:800; font-size:18px; text-align:right; pointer-events:none; }

    .teamline{ color:#dbe3ff; font-size:14px; font-weight:600; margin-top:6.5px; letter-spacing:.05px; opacity:.95; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .tl-wrap{ position:relative; }
    .tl-has-crest{ padding-left:24px; }
    .crest-icon{ height:1.35em; width:auto; object-fit:contain; image-rendering:auto; }
    .crest-abs{ position:absolute; left:0; top:50%; transform:translateY(-50%); pointer-events:none; }

    /* ---- Individual Metrics (CB style: label + raw + badge) ---- */
    .m-sec{ background:#121621; border:1px solid #242b3b; border-radius:16px; padding:10px 12px; }
    .m-title{ color:#e8ecff; font-weight:800; letter-spacing:.02em; margin:4px 0 10px 0; }

    .m-row{ display:flex; align-items:center; gap:10px; padding:8px 8px; border-radius:10px; }
    .m-label{ color:#c9d3f2; font-size:15.5px; letter-spacing:.1px; flex:1 1 0%; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .m-right{ display:flex; align-items:center; gap:10px; flex:0 0 auto; }
    .m-val{ color:#a8b3cf; font-size:13px; opacity:.9; min-width:54px; text-align:right; }
    .m-badge{ flex:0 0 auto; min-width:44px; text-align:center; padding:2px 10px; border-radius:8px;
              font-weight:800; font-size:18.5px; color:#0b0d12; border:1px solid rgba(0,0,0,.15); box-shadow:none; }

    .metrics-grid{ display:grid; grid-template-columns:1fr; gap:12px; }
    @media (min-width: 720px){ .metrics-grid{ grid-template-columns:repeat(3,1fr);} }
    </style>
    """, unsafe_allow_html=True)

    # ---- CB-style globals ----
    global_photo_overrides = load_local_photo_overrides(PLAYER_PHOTO_OVERRIDES_JSON)
    st.session_state.setdefault("photo_map", {})
    st.session_state.setdefault("crest_map", {})

    # ---- UI ROW 1: Age, Role sort, Search ----
    c1, c2, c3 = st.columns([1, 1.4, 2])
    with c1:
        age_choice = st.selectbox(
            "Age",
            ["All","U18","U20","U21","U22","U23","U25","U30","25+","28+","30+"],
            index=0,
            key="pro_age_filter_fb",
            label_visibility="visible"
        )
    with c2:
        ROLE_SCORE_COLS = [col for col, _ in _FB_ROLE_MAP if col in df_view.columns]
        sort_candidates = ["All In Score"] + ROLE_SCORE_COLS
        sort_by = st.selectbox("Role (sort by)", options=sort_candidates, index=0, key="pro_sort_by_fb")
    with c3:
        search_q = st.text_input("🔎 Search player / team / league", "", key="fb_global_search")

    # ---- Order toggle ----
    sort_dir_label = st.radio("Order", options=["High → Low", "Low → High"], index=0, key="pro_sort_dir_fb", horizontal=True)
    asc = (sort_dir_label == "Low → High")

    # ---- Pill toggle (choose 3 from FB set) ----
    available = [(col, label) for col, label in _FB_ROLE_MAP if col in df_view.columns]
    if not available:
        st.info("No fullback role score columns found in the dataframe.")
        return
    labels = [label for _, label in available]
    label_to_col = {label: col for col, label in available}

    default_labels = ["Build Up FB","Attacking FB","Defensive FB"]
    default_labels = [lbl for lbl in default_labels if lbl in labels]
    for lbl in labels:
        if len(default_labels) >= 3:
            break
        if lbl not in default_labels:
            default_labels.append(lbl)

    selected_labels = st.multiselect(
        "Displayed role pills (pick 3)",
        options=labels,
        default=default_labels[:3],
        key="fb_pill_select"
    )
    if len(selected_labels) != 3:
        st.warning("Please select exactly 3 pills — showing the first three available.")
        selected_labels = (selected_labels + [lbl for lbl in labels if lbl not in selected_labels])[:3]

    # ---- start from full table ----
    df_filtered = df_view.copy()

    # -----------------
    # Market Value (display-only) + print toggle
    # -----------------
    show_mv_next_to_contract = st.checkbox(
        "Show Market Value next to contract",
        value=False,
        key="pro_show_mv_next_contract_fb"
    )

    mv_mode = st.selectbox(
        "Market Value filter (display-only)",
        ["Off", "Max only", "Range"],
        index=0,
        key="pro_mv_mode_fb"
    )

    df_filtered["__mv"] = pd.to_numeric(
        df_filtered.get("Market value"),
        errors="coerce"
    )

    mv_mask = df_filtered["__mv"] < 1_000
    df_filtered.loc[mv_mask, "__mv"] = df_filtered.loc[mv_mask, "__mv"] * 1_000_000

    mv_min = None
    mv_max = None

    if mv_mode == "Max only":
        mv_max_m = st.slider(
            "Max Market Value (millions)",
            0.0, 400.0,
            30.0,
            step=0.5,
            key="pro_mv_max_m_fb"
        )
        mv_max = mv_max_m * 1_000_000

    elif mv_mode == "Range":
        mv_min_m, mv_max_m = st.slider(
            "Market Value Range (millions)",
            0.0, 400.0,
            (0.0, 30.0),
            step=0.5,
            key="pro_mv_range_m_fb"
        )
        mv_min = mv_min_m * 1_000_000
        mv_max = mv_max_m * 1_000_000

    if mv_max is not None:
        df_filtered = df_filtered[df_filtered["__mv"] <= mv_max]
    if mv_min is not None:
        df_filtered = df_filtered[df_filtered["__mv"] >= mv_min]



    # ---- Global search (Player / Team / League) ----
    if search_q:
        s = str(search_q).strip().lower()
        cols = [c for c in ("Player", "Team", "League") if c in df_filtered.columns]
        if cols:
            mask_any = pd.Series(False, index=df_filtered.index)
            for c in cols:
                mask_any = mask_any | df_filtered[c].astype(str).str.lower().str.contains(s, na=False)
            df_filtered = df_filtered[mask_any]

    # ---- Age filter ----
    if "Age" in df_filtered.columns and age_choice != "All":
        try:
            df_filtered["Age_num"] = pd.to_numeric(df_filtered["Age"], errors="coerce")
            if   age_choice == "U18": df_filtered = df_filtered[df_filtered["Age_num"] <= 18]
            elif age_choice == "U20": df_filtered = df_filtered[df_filtered["Age_num"] <= 20]
            elif age_choice == "U21": df_filtered = df_filtered[df_filtered["Age_num"] <= 21]
            elif age_choice == "U22": df_filtered = df_filtered[df_filtered["Age_num"] <= 22]
            elif age_choice == "U23": df_filtered = df_filtered[df_filtered["Age_num"] <= 23]
            elif age_choice == "U25": df_filtered = df_filtered[df_filtered["Age_num"] <= 25]
            elif age_choice == "U30": df_filtered = df_filtered[df_filtered["Age_num"] <= 30]
            elif age_choice == "25+": df_filtered = df_filtered[df_filtered["Age_num"] >= 25]
            elif age_choice == "28+": df_filtered = df_filtered[df_filtered["Age_num"] >= 28]
            elif age_choice == "30+": df_filtered = df_filtered[df_filtered["Age_num"] >= 30]
        except Exception:
            pass

    # ---- Contract expiry filter (max year) ----
    if "Contract expires" in df_filtered.columns:
        contract_choice = st.selectbox(
            "Contract expires (max year)",
            ["Any", "2024", "2025", "2026", "2027", "2028"],
            index=0,
            key="pro_contract_filter_fb",
            label_visibility="visible"
        )
        if contract_choice != "Any":
            try:
                max_year = int(contract_choice)
                df_filtered["_contract_year"] = pd.to_datetime(df_filtered["Contract expires"], errors="coerce").dt.year
                df_filtered = df_filtered[df_filtered["_contract_year"] <= max_year]
            except Exception:
                pass

    # ---- Birth country filter ----
    if "Birth country" in df_filtered.columns:
        country_vals = (
            df_filtered["Birth country"].dropna().astype(str).str.strip()
        )
        country_vals = sorted({c for c in country_vals if c and c.lower() not in {"nan","none","null"}})
        selected_countries = st.multiselect("Birth country", options=country_vals, default=[], key="pro_birth_country_filter_fb")
        if selected_countries:
            df_filtered = df_filtered[df_filtered["Birth country"].isin(selected_countries)]

    # ---- Foot filter ----
    df_filtered["__foot"] = df_filtered.apply(_get_foot, axis=1)
    foot_vals = (
        df_filtered["__foot"].dropna().astype(str).str.strip()
    )
    foot_vals = sorted({f for f in foot_vals if f and f.lower() not in {"nan","none","null"}})
    if foot_vals:
        selected_feet = st.multiselect("Foot", options=foot_vals, default=[], key="pro_foot_filter_fb")
        if selected_feet:
            df_filtered = df_filtered[df_filtered["__foot"].isin(selected_feet)]

    # ---------- optional minimum role score filters ----------
    ROLE_SCORE_COLS_FILTER = [col for col, _ in _FB_ROLE_MAP if col in df_filtered.columns]
    use_role_filters = st.checkbox("Filter by minimum role score(s)", value=False, key="fb_role_filter_toggle")

    if use_role_filters and ROLE_SCORE_COLS_FILTER:
        st.write("Set minimum scores (0–99). Any slider > 0 will filter that role.")
        minima = {}
        for col in ROLE_SCORE_COLS_FILTER:
            pretty = col.replace("Score", "").strip()
            minima[col] = st.slider(f"Min {pretty}", 0, 99, 0, 1, key=f"fb_min_{_norm(col)}")
        for col, thr in minima.items():
            if thr > 0:
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")
                df_filtered = df_filtered[df_filtered[col] >= thr]

    # ---- data check ----
    all_col = "All In Score"
    if all_col not in df_view.columns:
        st.info("Pro Layout needs the role scores. Make sure the table section above ran first.")
        return
    if df_filtered.empty:
        st.info("No players match the selected filters.")
        return

    # ---- Sorting ----
    _sort_col = "__sort_val"
    df_filtered[_sort_col] = pd.to_numeric(df_filtered.get(sort_by, pd.Series(index=df_filtered.index)), errors="coerce")
    ranked = (
        df_filtered
        .sort_values([_sort_col, all_col], ascending=[asc, False], na_position="last")
        .drop(columns=[_sort_col])
        .head(top_n)
        .reset_index(drop=True)
    )

    # ========================= RENDER CARDS =========================
    for i, row in ranked.iterrows():
        player = str(row.get("Player", "")) or ""
        team = str(row.get("Team", "")) or ""
        league = str(row.get("League", "")) or ""
        pos = str(row.get("Position", "")) or ""

        # Age text
        try:
            age_val = int(row.get("Age")) if not pd.isna(row.get("Age", None)) else int(row.get("Age_num", 0))
        except Exception:
            age_val = 0
        age_txt = f"{age_val}y.o." if age_val > 0 else "—"

        cy = pd.to_datetime(row.get("Contract expires"), errors="coerce")
        cyr = int(cy.year) if pd.notna(cy) else 0
        contract_txt = f"{cyr}" if cyr > 0 else "—"

        mv_txt = _fmt_mv_gbp(row.get("Market value")) if "Market value" in ranked.columns else "—"
        if show_mv_next_to_contract and mv_txt != "—":
            contract_txt = f"{contract_txt} · {mv_txt}" if contract_txt != "—" else mv_txt

        birth = row.get("Birth country", "") if "Birth country" in row else ""
        flag = _flag_html(birth)
        foot = _get_foot(row) or "—"

        # ---- role pills (driven by selection) ----
        pill_triplet = []
        for lbl in selected_labels:
            col = label_to_col.get(lbl, "")
            val = _pro_show99(row.get(col, 0))
            pill_triplet.append((lbl, val, _fmt2(val)))

        while len(pill_triplet) < 3:
            pill_triplet.append(("—", 0, "00"))
        pill_triplet = pill_triplet[:3]
        (l1, v1, t1), (l2, v2, t2), (l3, v3, t3) = pill_triplet

        # ---- positions — preserve dataset order & de-dupe ----
        raw = (pos or "").strip().upper()
        codes = [c for c in _re.split(r"[,\s/;]+", raw) if c]
        seen = set()
        ordered = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        pos_html = "".join(
            f"<span class='postext' style='color:{_pro_chip_color(c)}'>{c}</span>"
            for c in ordered
        )

        # ✅ CB-style avatar resolver (team + surname -> squad)
        key_id = f"{_norm(player)}|{_norm(team)}"
        avatar_url = resolve_player_photo(
            player=player,
            team=team,
            league=league,
            key_id=key_id,
            session_photo_map=st.session_state["photo_map"],
            global_overrides=global_photo_overrides,
        )

        # ✅ crest resolver (same as CB)
        crest_store_key = f"{_norm(team)}|{_norm(league)}"
        crest_url = st.session_state.get("crest_map", {}).get(crest_store_key, "")
        if not crest_url:
            team_url = get_fotmob_url(team)
            crest_url = _fotmob_crest_url(team_url) if team_url else ""

        if crest_url:
            teamline_html = (
                f"<div class='teamline tl-wrap tl-has-crest'>"
                f"<img class='crest-icon crest-abs' src='{crest_url}' alt=''>"
                f"<span class='teamtext'>{team} · {league}</span>"
                f"</div>"
            )
        else:
            teamline_html = f"<div class='teamline'>{team} · {league}</div>"

        _subtitle_map = {
            "Build Up FB":"Build Up FB",
            "Attacking FB":"Attacking FB",
            "Defensive FB":"Defensive FB",
            "Wide Creator":"Wide Creator",
            "Wide Carrier":"Wide Carrier",
            "PL Profile":"PL Profile",
        }

        st.markdown(f"""
        <div class='pro-wrap'>
          <div class='pro-card'>
            <div class='leftcol'>
              <div class='pro-avatar'>
                <img src="{avatar_url}" srcset="{avatar_url} 1x, {avatar_url} 2x" alt="{player}" loading="lazy" />
              </div>
              <div class='row leftrow1'>{flag}<span class='chip'>{age_txt}</span></div>
              <div class='row leftrow-foot'><span class='chip'>{foot}</span></div>
              <div class='row leftrow-contract'><span class='chip'>{contract_txt}</span></div>
            </div>
            <div>
              <div class='name'>{player}</div>
              <div class='row' style='align-items:center;'><span class='pill' style='background:{_pro_rating_color(v1)}'>{t1}</span><span class='sub'>{_subtitle_map.get(l1,l1)}</span></div>
              <div class='row' style='align-items:center;'><span class='pill' style='background:{_pro_rating_color(v2)}'>{t2}</span><span class='sub'>{_subtitle_map.get(l2,l2)}</span></div>
              <div class='row' style='align-items:center;'><span class='pill' style='background:{_pro_rating_color(v3)}'>{t3}</span><span class='sub'>{_subtitle_map.get(l3,l3)}</span></div>
              <div class='row posrow'>{pos_html}</div>
              {teamline_html}
            </div>
            <div class='rank'>#{_fmt2(i+1)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ========================= EXPANDER (FULL, BACK) =========================
        with st.expander("Individual Metrics", expanded=False):

            ATT = [
                ("Crosses","Crosses per 90"),
                ("Crossing Accuracy %","Accurate crosses, %"),
                ("Goals: Non-Penalty","Non-penalty goals per 90"),
                ("xG","xG per 90"),
                ("Expected Assists","xA per 90"),
                ("Offensive Duels","Offensive duels per 90"),
                ("Offensive Duel Success %","Offensive duels won, %"),
                ("Progressive Runs","Progressive runs per 90"),
                ("Shots","Shots per 90"),
                ("Shooting Accuracy %","Shots on target, %"),
                ("Touches in Opposition Box","Touches in box per 90"),
            ]

            DEF = [
                ("Aerial Duels","Aerial duels per 90"),
                ("Aerial Duel Success %","Aerial duels won, %"),
                ("Defensive Duels","Defensive duels per 90"),
                ("Defensive Duel Success %","Defensive duels won, %"),
                ("Shots Blocked","Shots blocked per 90"),
                ("PAdj. Interceptions","PAdj Interceptions"),
            ]

            POS = [
                ("Accelerations","Accelerations per 90"),
                ("Deep Completions","Deep completions per 90"),
                ("Dribbles","Dribbles per 90"),
                ("Dribbling Success %","Successful dribbles, %"),
                ("Forward Passes","Forward passes per 90"),
                ("Forward Passing %","Accurate forward passes, %"),
                ("Key passes","Key passes per 90"),
                ("Long Passes","Long passes per 90"),
                ("Long Passing %","Accurate long passes, %"),
                ("Passes","Passes per 90"),
                ("Passing %","Accurate passes, %"),
                ("Passes to Final 3rd","Passes to final third per 90"),
                ("Passes to Final 3rd %","Accurate passes to final third, %"),
                ("Passes to Penalty Area","Passes to penalty area per 90"),
                ("Pass to Penalty Area %","Accurate passes to penalty area, %"),
                ("Progessive Passes","Progressive passes per 90"),
                ("Progessive Passing %","Accurate progressive passes, %"),
                ("Smart Passes","Smart passes per 90"),
            ]

            def _sec_html(title, pairs):
                pairs = _available_metric_pairs(df_view, pairs)
                rows = []
                for lab, met in pairs:
                    pct = _metric_pct(row, met)
                    p = _pro_show99(pct if not pd.isna(pct) else 0.0)
                    ptxt = _fmt2(p)

                    raw = _metric_val(row, met)
                    raw_txt = "—" if pd.isna(raw) else f"{raw:.2f}".rstrip("0").rstrip(".")

                    rows.append(
                        "<div class='m-row'>"
                        f"<div class='m-label'>{lab}</div>"
                        "<div class='m-right'>"
                        f"<div class='m-val'>{raw_txt}</div>"
                        f"<div class='m-badge' style='background:{_pro_rating_color(p)}'>{ptxt}</div>"
                        "</div></div>"
                    )
                return f"<div class='m-sec'><div class='m-title'>{title}</div>{''.join(rows) if rows else ''}</div>"

            st.markdown(
                "<div class='metrics-grid'>"
                + _sec_html("ATTACKING", ATT)
                + _sec_html("DEFENSIVE", DEF)
                + _sec_html("POSSESSION", POS)
                + "</div>",
                unsafe_allow_html=True
            )

            # --- Player image override (per-player keys) ---
            img_key = f"imgurl_{i}_{key_id}"
            default_url = st.session_state.get("photo_map", {}).get(key_id, "")
            uploaded_file = st.file_uploader(
                "Upload player image (PNG/JPG)",
                type=["png","jpg","jpeg"],
                key=f"upload_{i}_{key_id}"
            )
            _ = st.text_input(
                "Custom image URL (override avatar — e.g., https://images.fotmob.com/image_resources/playerimages/1199383.png)",
                value=default_url,
                key=img_key
            )

            col_a, col_b = st.columns([1, 3])
            with col_a:
                if st.button("Apply to this player", key=f"apply_{i}_{key_id}"):
                    if uploaded_file is not None:
                        try:
                            data = uploaded_file.getvalue()
                            try:
                                from PIL import Image
                                Image.open(io.BytesIO(data))
                            except Exception:
                                pass

                            mime = getattr(uploaded_file, "type", "") or ""
                            if not mime.startswith("image/"):
                                ext = os.path.splitext(uploaded_file.name or "")[1].lower()
                                if ext == ".svg": mime = "image/svg+xml"
                                elif ext == ".png": mime = "image/png"
                                elif ext in (".jpg", ".jpeg"): mime = "image/jpeg"
                                else: mime = "image/png"

                            b64 = base64.b64encode(data).decode("ascii")
                            st.session_state.setdefault("photo_map", {})[key_id] = f"data:{mime};base64,{b64}"
                            st.success("Uploaded image saved!")
                            try: st.rerun()
                            except Exception: st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Couldn't process the uploaded image: {e}")
                    else:
                        val = (st.session_state.get(img_key, "") or "").strip()
                        if not val:
                            st.error("Please upload an image or paste an image URL.")
                        elif not (val.startswith("http://") or val.startswith("https://") or val.startswith("data:image/")):
                            st.error("Image URL must start with http://, https://, or data:image/…")
                        else:
                            st.session_state.setdefault("photo_map", {})[key_id] = val
                            st.success("Saved!")
                            try: st.rerun()
                            except Exception: st.experimental_rerun()
            with col_b:
                if st.button("Clear override", key=f"clear_{i}_{key_id}"):
                    st.session_state.setdefault("photo_map", {}).pop(key_id, None)
                    st.info("Cleared.")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()

            # --- Club crest override (stored per-club) ---
            crest_widget_ns = f"{crest_store_key}|{key_id}|{i}"
            crest_default = st.session_state.get("crest_map", {}).get(crest_store_key, "")
            crest_upload = st.file_uploader(
                "Upload club crest (SVG/PNG/JPG)",
                type=["svg","png","jpg","jpeg"],
                key=f"crest_upload_{crest_widget_ns}"
            )
            _ = st.text_input(
                "Custom crest URL (e.g., https://…/club.svg or .png)",
                value=crest_default,
                key=f"crest_url_{crest_widget_ns}"
            )

            col_c, col_d = st.columns([1, 3])
            with col_c:
                if st.button("Apply crest", key=f"apply_crest_{crest_widget_ns}"):
                    if crest_upload is not None:
                        try:
                            data = crest_upload.getvalue()
                            mime = crest_upload.type or ""
                            if not mime.startswith("image/"):
                                ext = os.path.splitext(crest_upload.name or "")[1].lower()
                                if ext == ".svg": mime = "image/svg+xml"
                                elif ext == ".png": mime = "image/png"
                                elif ext in (".jpg",".jpeg"): mime = "image/jpeg"
                                else: mime = "image/png"
                            b64 = base64.b64encode(data).decode("ascii")
                            st.session_state.setdefault("crest_map", {})[crest_store_key] = f"data:{mime};base64,{b64}"
                            st.success("Crest saved!")
                            try: st.rerun()
                            except Exception: st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Couldn't process crest: {e}")
                    else:
                        val = (st.session_state.get(f"crest_url_{crest_widget_ns}", "") or "").strip()
                        if not val:
                            st.error("Upload a crest or paste a crest URL.")
                        elif not (val.startswith("http://") or val.startswith("https://") or val.startswith("data:image/")):
                            st.error("Crest URL must start with http://, https://, or data:image/…")
                        else:
                            st.session_state.setdefault("crest_map", {})[crest_store_key] = val
                            st.success("Crest URL saved!")
                            try: st.rerun()
                            except Exception: st.experimental_rerun()
            with col_d:
                if st.button("Clear crest", key=f"clear_crest_{crest_widget_ns}"):
                    st.session_state.setdefault("crest_map", {}).pop(crest_store_key, None)
                    st.info("Crest cleared.")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()


# ---- TAB HOOK (identical pattern) ----
with tabs[4]:
    st.subheader("Pro Layout — Top Fullbacks (Tiles)")
    render_pro_layout_fb(df_f, top_n=top_n)
# ----------------- END PRO LAYOUT TAB — FULLBACKS -----------------


# ----------------- METRIC LEADERBOARD — themed + palettes + custom title + highlights (UPDATED) -----------------
import re, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams, font_manager as fm
import pandas as pd
import streamlit as st

# --- Rendering crispness & font setup
rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.antialiased": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter","Roboto","SF Pro Text","Segoe UI","Helvetica Neue","Arial"],
})
for p in ["./fonts/Inter-Variable.ttf","./fonts/Inter-Regular.ttf"]:
    try: fm.fontManager.addfont(p)
    except: pass

st.markdown("---")

with st.expander("Leaderboard settings", expanded=False):
    # Basic controls
    default_metric = "xA per 90" if "Progressive passes per 90" in FEATURES else FEATURES[0]
    metric_pick   = st.selectbox("Metric", FEATURES, index=FEATURES.index(default_metric))
    top_n         = st.slider("Top N", 5, 40, 20, 5)

    # Theme (backgrounds must be identical for page & plot)
    theme = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True, key="lb_theme")
    if theme == "Light":
        PAGE_BG = "#ebebeb"
        PLOT_BG = "#ebebeb"  # same as page per request
        GRID_MAJ = "#d7d7d7"
        TXT      = "#111111"
        TICK_NUM = "#111111"  # axis numbers (ticks)
        SPINE    = "#c8c8c8"
    else:
        PAGE_BG = "#0a0f1c"
        PLOT_BG = "#0a0f1c"  # same as page per request
        GRID_MAJ = "#3a4050"
        TXT      = "#f5f5f5"
        TICK_NUM = "#ffffff"  # axis numbers (ticks)
        SPINE    = "#6b7280"

    # Palette options (same set as scatterplot + new uniform red/blue/green)
    palette_options = [
        "Red–Gold–Green (diverging)",
        "Light-grey → Black",
        "Light-Red → Dark-Red",
        "Light-Blue → Dark-Blue",
        "Light-Green → Dark-Green",
        "Purple ↔ Gold (diverging)",
        "All White",
        "All Black",
        "All Red",    # NEW
        "All Blue",   # NEW
        "All Green",  # NEW
    ]
    palette_choice = st.selectbox("Palette", palette_options, index=palette_options.index("All Black"), key="lb_palette")
    reverse_scale  = st.checkbox("Reverse colours", value=False, key="lb_reverse")

    # Labels
    show_team_names = st.checkbox("Show team names", value=True, key="lb_show_team")  # NEW

    # Custom title
    show_title   = st.checkbox("Show custom title", value=False, key="lb_show_title")
    custom_title = st.text_input("Custom title", "Top N – Metric", key="lb_title")

# --- Data
val_col = metric_pick
plot_df = df_f[["Player","Team",val_col]].dropna(subset=[val_col]).copy()
plot_df[val_col] = pd.to_numeric(plot_df[val_col], errors="coerce")
plot_df = plot_df.dropna(subset=[val_col])
plot_df = plot_df.sort_values(val_col, ascending=False).head(int(top_n)).reset_index(drop=True)

# Option: highlight a single player (from current Top N)
highlight_player = st.selectbox(
    "Highlight single player (from Top N)", ["(None)"] + plot_df["Player"].astype(str).tolist(),
    index=0, key="lb_highlight_player"
)

# --- Label helpers
def abbrev_name(player):
    tokens = re.split(r"\s+", str(player).strip())
    if tokens and tokens[0]:
        initial = tokens[0][0]
        last = re.sub(r"[^\w\-’']", "", tokens[-1])
        return f"{initial}.{last}"
    return str(player)

p_abbr = [abbrev_name(p) for p in plot_df["Player"]]
teams  = plot_df["Team"].astype(str).tolist()
vals   = plot_df[val_col].astype(float).values if len(plot_df) else np.array([0.0])

# --- Colour mapping (same logic as scatterplot, plus uniform colours)
def interp(a, b, u):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    return (a + (b - a) * np.clip(u, 0, 1)) / 255.0

def color_mapper(palette, t):
    if palette == "Red–Gold–Green (diverging)":
        red, gold, green = [199,54,60], [240,197,106], [61,166,91]
        return interp(red, gold, t/0.5) if t <= 0.5 else interp(gold, green, (t-0.5)/0.5)
    if palette == "Light-grey → Black":
        return interp([210,214,220], [20,23,31], t)
    if palette == "Light-Red → Dark-Red":
        return interp([252,190,190], [139,0,0], t)
    if palette == "Light-Blue → Dark-Blue":
        return interp([191,210,255], [10,42,102], t)
    if palette == "Light-Green → Dark-Green":
        return interp([196,235,203], [12,92,48], t)
    if palette == "Purple ↔ Gold (diverging)":
        purple, mid, gold = [96,55,140], [180,150,210], [240,197,106]
        return interp(purple, mid, t/0.5) if t <= 0.5 else interp(mid, gold, (t-0.5)/0.5)
    if palette == "All White":
        return np.array([255,255,255])/255.0
    if palette == "All Black":
        return np.array([0,0,0])/255.0
    if palette == "All Red":
        return np.array([197, 30, 30])/255.0
    if palette == "All Blue":
        return np.array([15, 70, 180])/255.0
    if palette == "All Green":
        return np.array([20, 120, 60])/255.0
    return np.array([0,0,0])/255.0

if len(vals) > 1:
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax: vmax = vmin + 1e-6
    ts = (vals - vmin) / (vmax - vmin)
else:
    ts = np.zeros_like(vals)

if reverse_scale:
    ts = 1.0 - ts
# Build colors (handles both gradients and uniform)
bar_colors = [tuple(color_mapper(palette_choice, float(t))) for t in ts]

# --- Figure
fig, ax = plt.subplots(figsize=(11.5, 6.2))
fig.patch.set_facecolor(PAGE_BG)
ax.set_facecolor(PLOT_BG)

# Title (reduce by 4 pts → 26)
default_title = f"Top {len(plot_df)} – {metric_pick}"
title_text = custom_title.strip() if (show_title and custom_title.strip()) else default_title
fig.suptitle(title_text, fontsize=26, fontweight="bold", color=TXT, y=0.985)

# Layout
plt.subplots_adjust(top=0.90, left=0.30, right=0.965, bottom=0.14)

# Bars
ypos = np.arange(len(vals))
bars = ax.barh(ypos, vals, color=bar_colors, edgecolor="none", zorder=2)

# Highlight a single player
if highlight_player and highlight_player != "(None)":
    mask = plot_df["Player"].astype(str) == highlight_player
    if mask.any():
        idxs = np.where(mask.values)[0]
        for i in idxs:
            bars[i].set_color("#f59e0b")
            bars[i].set_edgecolor("white")
            bars[i].set_linewidth(1.6)
            bars[i].set_zorder(5)

# Axis & labels
ax.invert_yaxis()
ax.set_yticks(ypos)
if show_team_names:
    yticklabels_math = [rf'$\bf{{{p}}}$, {t}' for p, t in zip(p_abbr, teams)]
else:
    yticklabels_math = [rf'$\bf{{{p}}}$' for p in p_abbr]
ax.set_yticklabels(yticklabels_math, fontsize=10.5, color=TXT)
ax.set_ylabel("")
ax.set_xlabel(val_col, color=TXT, labelpad=6, fontsize=10.5, fontweight="semibold")

# Gridlines
ax.grid(axis="x", color=GRID_MAJ, linewidth=0.8, zorder=1)

# Spines & ticks
for side in ["top","right","left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(SPINE)
ax.tick_params(axis="y", length=0)

# X ticks formatting + themed colour + medium weight
def fmt(x, _): return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"
ax.xaxis.set_major_formatter(FuncFormatter(fmt))
for tick in ax.get_xticklabels():
    tick.set_fontweight("medium")
    tick.set_color(TICK_NUM)  # black on light, white on dark

# Range & padding
xmax = float(vals.max()) if len(vals) else 1.0
ax.set_xlim(0, xmax * 1.1)

# Value labels (8.5 pt beside bars)
pad = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
for rect, v in zip(bars, vals):
    ax.text(rect.get_width() + pad,
            rect.get_y() + rect.get_height()/2,
            fmt(v, None),
            va="center", ha="left", fontsize=8.5, color=TXT)

st.pyplot(fig, use_container_width=True)
# ----------------- END -----------------




# ----------------- SINGLE PLAYER ROLE PROFILE (REPLACED) -----------------
# ================= TOP OF INDIVIDUAL PLAYER PROFILE =================
# Assumes df_f exists and has at least columns: Player, Position, League

st.subheader("🎯 Single Player Role Profile")

# 1) Player picker (from df_f) + persist to session state
player_name = st.selectbox("Choose player", sorted(df_f["Player"].dropna().unique()))
st.session_state["selected_player"] = player_name  # <-- critical for downstream defaults

# 2) Pull the player's row and safe defaults used by other blocks
player_row = df_f[df_f["Player"] == player_name].head(1)

# robust default position prefix & default league for pools
if not player_row.empty:
    _pos = str(player_row.iloc[0].get("Position", ""))
    default_pos_prefix = (_pos[:2] if len(_pos) >= 2 else _pos) or "CF"
    default_league_for_pool = [player_row.iloc[0].get("League")]
else:
    default_pos_prefix = "CF"
    default_league_for_pool = []

# (Optional) small helper to fetch the current selected name downstream
def _selected_name() -> str:
    return st.session_state.get("selected_player", player_name)
# ================= END TOP OF INDIVIDUAL PLAYER PROFILE =============


# derive defaults from selected player (to propagate)
default_pos_prefix = str(player_row["Position"].iloc[0])[:2] if not player_row.empty else "CF"
default_league_for_pool = [player_row["League"].iloc[0]] if not player_row.empty else []

# Pool controls (for chart + notes only; NOT used for role scores)
st.caption("Percentiles & chart computed against the pool below (defaults to the player's league).")
with st.container():
    c1, c2, c3 = st.columns([2,1,1])
    leagues_pool = c1.multiselect("Comparison leagues", sorted(df["League"].dropna().unique()), default=default_league_for_pool)
    min_minutes_pool, max_minutes_pool = c2.slider("Pool minutes", 0, 5000, (500, 5000))
    age_min_pool, age_max_pool = c3.slider("Pool age", 14, 45, (16, 40))  # default 16–40
    same_pos = st.checkbox("Limit pool to current position prefix", value=True)

def build_pool_df():
    if not leagues_pool:
        return pd.DataFrame([], columns=df.columns)
    pool = df[df["League"].isin(leagues_pool)].copy()
    pool["Minutes played"] = pd.to_numeric(pool["Minutes played"], errors="coerce")
    pool["Age"] = pd.to_numeric(pool["Age"], errors="coerce")
    pool = pool[pool["Minutes played"].between(min_minutes_pool, max_minutes_pool)]
    pool = pool[pool["Age"].between(age_min_pool, age_max_pool)]
    if same_pos and not player_row.empty:
        pool = pool[pool["Position"].astype(str).apply(position_filter)]
    pool = pool.dropna(subset=POLAR_METRICS)
    return pool

def clean_attacker_label(s: str) -> str:
    s = s.replace("Aerial duels won, %", "Aerial %")
    s = s.replace("xA per 90", "xA")
    s = s.replace("Defensive duels won, %", "Defensive Duel %")
    s = s.replace("Passes per 90", "Passes")
    s = s.replace("Dribbles per 90", "Dribbles")
    s = s.replace("Defensive duels per 90", "Defensive duels")
    s = s.replace("Touches in box per 90", "Touches in box")
    s = s.replace("Progressive passes per 90", "Progressive Passes")
    s = s.replace("Progressive runs per 90", "Progressive runs")
    s = s.replace("Passes to penalty area per 90", "Passes to Pen area")
    s = s.replace("Accurate passes, %", "Pass %")
    return s

def percentiles_for_player_in_pool(pool_df: pd.DataFrame, ply_row: pd.Series) -> dict:
    if pool_df.empty:
        return {}
    pct_map = {}
    for m in POLAR_METRICS:
        if m not in pool_df.columns or pd.isna(ply_row[m]): 
            continue
        series = pd.to_numeric(pool_df[m], errors="coerce").dropna()
        if series.empty: 
            continue
        rank = (series < float(ply_row[m])).mean() * 100.0
        eq_share = (series == float(ply_row[m])).mean() * 100.0
        pct_map[m] = min(100.0, rank + 0.5 * eq_share)
    return pct_map

# Polar chart for attacker metrics
def plot_attacker_polar_chart(labels, vals):
    N = len(labels)
    color_scale = ["#be2a3e", "#e25f48", "#f88f4d", "#f4d166", "#90b960", "#4b9b5f", "#22763f"]
    cmap = LinearSegmentedColormap.from_list("custom_scale", color_scale)
    bar_colors = [cmap(v/100.0) for v in vals]

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)[::-1]
    rotation_shift = np.deg2rad(75) - angles[0]
    ang = (angles + rotation_shift) % (2*np.pi)
    width = 2*np.pi / N

    fig = plt.figure(figsize=(8.2, 6.6), dpi=180)
    fig.patch.set_facecolor('#f3f4f6')
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.74], polar=True)
    ax.set_facecolor('#f3f4f6')
    ax.set_rlim(0, 100)

    for i in range(N):
        ax.bar(ang[i], vals[i], width=width, color=bar_colors[i], edgecolor='black', linewidth=1.0, zorder=3)
        label_pos = max(12, vals[i] * 0.75)
        ax.text(ang[i], label_pos, f"{int(round(vals[i]))}", ha='center', va='center',
                fontsize=9, weight='bold', color='white', zorder=4)

    outer = plt.Circle((0, 0), 100, transform=ax.transData._b, color='black', fill=False, linewidth=2.2, zorder=5)
    ax.add_artist(outer)
    for i in range(N):
        sep_angle = (ang[i] - width/2) % (2*np.pi)
        is_cross = any(np.isclose(sep_angle, a, atol=0.01) for a in [0, np.pi/2, np.pi, 3*np.pi/2])
        ax.plot([sep_angle, sep_angle], [0, 100], color='black' if is_cross else '#b0b0b0',
                linewidth=1.6 if is_cross else 1.0, zorder=2)

    label_r = 120
    for i, lab in enumerate(labels):
        ax.text(ang[i], label_r, lab, ha='center', va='center', fontsize=8.5, weight='bold', color='#111827', zorder=6)

    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['polar'].set_visible(False); ax.grid(False)
    return fig

# ---- render section ----
if player_row.empty:
    st.info("Pick a player above.")
else:
    ply = player_row.iloc[0]
    meta = player_row[["Team","League","Age","Contract expires","League Strength","Market value"]].iloc[0]

    # --- New: pull extra stats ---
    matches  = int(ply["Matches played"]) if "Matches played" in ply else "—"
    minutes  = int(ply["Minutes played"]) if "Minutes played" in ply else "—"
    goals    = int(ply["Goals"]) if "Goals" in ply else "—"
    assists  = int(ply["Assists"]) if "Assists" in ply else "—"

    # --- Caption with extra info ---
    st.caption(
        f"**{player_name}** — {meta['Team']} • {meta['League']} • "
        f"Age {int(meta['Age']) if pd.notna(meta['Age']) else 'N/A'} • "
        f"Apps: {matches}, {minutes} mins • G/A: {goals}/{assists} • "
        f"Contract: {pd.to_datetime(meta['Contract expires']).date() if pd.notna(meta['Contract expires']) else 'N/A'} • "
        f"League Strength {meta['League Strength']:.1f} • "
        f"Value €{meta['Market value']:,.0f}"
    )

    # Build pool & compute player percentiles within that pool
    pool_df = build_pool_df()
    if pool_df.empty:
        st.warning("Comparison pool is empty. Add at least one league.")
        pct_map = {}
    else:
        pct_map = percentiles_for_player_in_pool(pool_df, ply)

    # ---------- 1) PERFORMANCE CHART FIRST ----------
    labels = [clean_attacker_label(m) for m in POLAR_METRICS if m in pct_map]
    vals   = [pct_map[m] for m in POLAR_METRICS if m in pct_map]
    if vals:
        fig = plot_attacker_polar_chart(labels, vals)
        team = str(ply["Team"]); league = str(ply["League"])

# Minutes → 90s; goals/assists already parsed above
minutes_safe = minutes if isinstance(minutes, (int, float)) else 0
nineties = round(minutes_safe / 90.0, 1)
goals_safe = goals if isinstance(goals, (int, float)) else 0
assists_safe = assists if isinstance(assists, (int, float)) else 0

fig.text(0.06, 0.94, f"{player_name} — Performance Chart",
         fontsize=16, weight='bold', ha='left', color='#111827')
fig.text(0.06, 0.915, f"{team} • {league} • {nineties} 90's • Goals: {int(goals_safe)} • Assists: {int(assists_safe)}",
         fontsize=9, ha='left', color='#6b7280')

st.pyplot(fig, use_container_width=True)

   # ---------- 2) NOTES: Style / Strengths / Weaknesses ----------

EXTRA_METRICS = [ 'Defensive duels per 90', 'Defensive duels won, %',
    'Aerial duels per 90', 'Aerial duels won, %', 'Shots blocked per 90',
    'PAdj Interceptions', 'Non-penalty goals per 90', 'xG per 90', 
    'Shots per 90', 'Shots on target, %', 'Crosses per 90',
    'Accurate crosses, %', 'Dribbles per 90', 'Successful dribbles, %',
    'Offensive duels per 90', 'Offensive duels won, %', 'Touches in box per 90',
    'Progressive runs per 90', 'Accelerations per 90', 'Passes per 90',
    'Accurate passes, %', 'Forward passes per 90', 'Accurate forward passes, %',
    'Long passes per 90', 'Accurate long passes, %', 'xA per 90',
    'Smart passes per 90', 'Key passes per 90', 'Passes to final third per 90',
    'Accurate passes to final third, %', 'Passes to penalty area per 90',
    'Accurate passes to penalty area, %', 'Deep completions per 90',
    'Progressive passes per 90', 'Accurate progressive passes, %' 
]
STYLE_MAP = {
    'Defensive duels per 90': {
        'style': 'Ball Winner',
        'sw': 'Defensive Duel Attempts',
    },
    'Aerial duels won, %': {
        'style': None,
        'sw': 'Aerial Duels',
    },
    'Defensive duels won, %': {
        'style': None,
        'sw': 'Tackling %',
    },
    'Long passes per 90': {
        'style': 'Long Passer',
        'sw': None,
    },
    'xG per 90': {
        'style': None,
        'sw': 'Goal Threat',
    },
    'Shots per 90': {
        'style': 'Takes many shots',
        'sw': None,
    },
    'PAdj Interceptions': {
        'style': 'Cuts out opposition attacks',
        'sw': 'Defensive positioning',
    },
    'Accurate forward passes, %': {
        'style': None,
        'sw': 'Forward Passing Accuracy',
    },
    'Dribbles per 90': {
        'style': 'Dribbler',
        'sw': 'Dribble Volume',
    },
    'Successful dribbles, %': {
        'style': None,
        'sw': 'Dribbling Efficiency',
    },
    'Touches in box per 90': {
        'style': 'Busy in the penalty box',
        'sw': 'Penalty-box Coverage',
    },
    'Progressive runs per 90': {
        'style': 'Gets team up the pitch via carries',
        'sw': 'Progressive Runs',
    },
    'Passes per 90': {
        'style': 'Involved in build-up',
        'sw': 'Passing Involvement',
    },
    'Accurate passes, %': {
        'style': 'Secure Passer',
        'sw': 'Passing Retention',
    },
    'xA per 90': {
        'style': 'Creates goal scoring chances',
        'sw': 'Creativity',
    },
    'Passes to penalty area per 90': {
        'style': 'Creates openings',
        'sw': 'Passes to Penalty Area',
    },
    'Deep completions per 90': {
        'style': 'Gets ball into the box',
        'sw': None,
    },
    'Progressive passes per 90': {
        'style': 'Build up Passer',
        'sw': 'Ball progression via passes',
    },
    'Smart passes per 90': {
        'style': 'Attempts through balls',
        'sw': None,
    },
}

HI, LO, STYLE_T = 70, 30, 65

def percentile_in_series(value, series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or pd.isna(value): 
        return np.nan
    rank = (s < float(value)).mean() * 100.0
    eq_share = (s == float(value)).mean() * 100.0
    return min(100.0, rank + 0.5 * eq_share)

def chips(items, color):
    if not items: return "_None identified._"
    spans = [
        f"<span style='background:{color};color:#111;padding:2px 6px;border-radius:10px;margin:0 6px 6px 0;display:inline-block'>{txt}</span>"
        for txt in items[:10]
    ]
    return " ".join(spans)

# Build pool-based percentiles for EXTRA_METRICS; fallback to league-table percentiles on the player row
pct_extra = {}
if isinstance(pool_df, pd.DataFrame) and not pool_df.empty:
    for m in EXTRA_METRICS:
        if m in df.columns and m in pool_df.columns and pd.notna(ply.get(m)):
            pct_extra[m] = percentile_in_series(ply[m], pool_df[m])
for m in EXTRA_METRICS:
    if m not in pct_extra or pd.isna(pct_extra[m]):
        col = f"{m} Percentile"
        if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
            pct_extra[m] = float(player_row[col].iloc[0])

# Enforce style-only vs. strength/weakness-only via STYLE_MAP:
# - If 'sw' is None -> do NOT score strengths/weaknesses
# - If 'style' is None -> do NOT flag style
strengths, weaknesses, styles = [], [], []
for m, v in pct_extra.items():
    if pd.isna(v): 
        continue
    cfg = STYLE_MAP.get(m, {})
    sw_label = cfg.get('sw')          # keep None if absent
    style_tag = cfg.get('style')      # keep None if absent

    # Strengths/Weaknesses only if an sw label exists
    if sw_label:
        if v >= HI:
            strengths.append((sw_label, v))
        elif v <= LO:
            weaknesses.append((sw_label, v))

    # Style flag only if a style phrase exists
    if style_tag and v >= STYLE_T:
        styles.append((style_tag, v))

# De-dupe & sort nicely
if strengths:
    strength_best = {name: max(p for n,p in strengths if n==name) for name,_ in strengths}
    strengths = [name for name,_ in sorted(strength_best.items(), key=lambda kv: -kv[1])]
if weaknesses:
    weakness_worst = {name: min(p for n,p in weaknesses if n==name) for name,_ in weaknesses}
    weaknesses = [name for name,_ in sorted(weakness_worst.items(), key=lambda kv: kv[1])]
if styles:
    style_best = {name: max(p for n,p in styles if n==name) for name,_ in styles}
    styles = [name for name,_ in sorted(style_best.items(), key=lambda kv: -kv[1])]

# Summary + chips
st.markdown(
    f"**Profile:** {player_name} — {ply.get('Team','?')} ({ply.get('League','?')}), "
    f"age {int(ply['Age']) if pd.notna(ply.get('Age')) else '—'}, "
    f"minutes {int(ply['Minutes played']) if pd.notna(ply['Minutes played']) else '—'}."
)
st.markdown("**Style:**")
st.markdown(chips(styles, "#bfdbfe"), unsafe_allow_html=True)   # light blue
st.markdown("**Strengths:**")
st.markdown(chips(strengths, "#a7f3d0"), unsafe_allow_html=True)  # light green
st.markdown("**Weaknesses:**")
st.markdown(chips(weaknesses, "#fecaca"), unsafe_allow_html=True) # light red

# ---------- 3) ROLE SCORES (MATCH TABLES EXACTLY) ----------
def table_style_role_scores_from_row(row):
    """Use per-league percentiles from df_f (already computed) + sidebar league weighting."""
    rs = {}
    for role, rd in ROLES.items():
        total_w = sum(rd["metrics"].values()) or 1.0
        metric_score = 0.0
        for m, w in rd["metrics"].items():
            pct_col = f"{m} Percentile"
            if pct_col in row.index and pd.notna(row[pct_col]):
                metric_score += float(row[pct_col]) * w
        metric_score /= total_w
        if use_league_weighting:
            league_scaled = float(row.get("League Strength", 50.0))  # 0..100
            metric_score = (1 - beta) * metric_score + beta * league_scaled
        rs[role] = metric_score
    return rs

role_scores = table_style_role_scores_from_row(player_row.iloc[0])

# Best role line — choose ONLY among the first three roles in ROLES
if role_scores:
    role_list = list(ROLES.keys())[:3]
    candidates = [(r, role_scores.get(r, np.nan)) for r in role_list]
    candidates = [(r, v) for r, v in candidates if pd.notna(v)]
    if candidates:
        best_role = max(candidates, key=lambda kv: kv[1])[0]
        st.markdown(f"**Best role:** {best_role}.")

# Role table with gradient colors (show all roles)
def score_to_color(v: float) -> str:
    if pd.isna(v): return "background-color: #ffffff"
    if v <= 50:
        r1,g1,b1 = (190,42,62); r2,g2,b2 = (244,209,102); t = v/50
    else:
        r1,g1,b1 = (244,209,102); r2,g2,b2 = (34,197,94); t = (v-50)/50
    r = int(r1 + (r2-r1)*t); g = int(g1 + (g2-g1)*t); b = int(b1 + (b2-b1)*t)
    return f"background-color: rgb({r},{g},{b})"

rows = [{"Role": r, "Percentile": role_scores.get(r, np.nan)} for r in ROLES.keys()]
role_df = pd.DataFrame(rows).set_index("Role")
styled = (
    role_df.style
    .applymap(lambda x: score_to_color(float(x)) if pd.notna(x) else "background-color:#fff", subset=["Percentile"])
    .format({"Percentile": lambda x: f"{int(round(x))}" if pd.notna(x) else "—"})
)
st.dataframe(styled, use_container_width=True)
# ----------------- END SINGLE PLAYER ROLE PROFILE -----------------


# =====================================================================
# ============== BELOW THE NOTES: 3 EXTRA FEATURE BLOCKS ==============
# =====================================================================

# ============================ (E) ONE-PAGER — WIDER PANELS, SMALLER CENTER GAP, EXTRA TOP-LEFT PADDING ============================

from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.markdown("---")

if player_row.empty:
    st.info("Pick a player above.")
else:
    # --------- palette / tokens ---------
    PAGE_BG   = "#0a0f1c"
    PANEL_BG  = "#11161C"
    TRACK_BG  = "#222c3d"
    TEXT      = "#E5E7EB"
    ROLE_GREY = "#737373"

    CHIP_G_BG = "#22C55E"; CHIP_R_BG = "#EF4444"; CHIP_B_BG = "#60A5FA"

    # --------- layout / padding knobs ---------
    NAME_X   = 0.055   # more breathing room on the left
    META_X   = 0.055
    CHIP_X0  = 0.055   # chips/roles start x
    GUTTER_PAD  = 0.006

    # ----------------- helpers -----------------
    def div_color_tuple(v: float):
        if pd.isna(v): return (0.6,0.63,0.66)
        v = float(v)
        if v <= 50:
            t = v/50.0;  c1, c2 = np.array([239,68,68]),  np.array([234,179,8])
        else:
            t = (v-50)/50.0; c1, c2 = np.array([234,179,8]), np.array([34,197,94])
        return tuple(((c1 + (c2-c1)*t)/255.0).astype(float))

    def _text_width_frac(fig, s, *, fontsize=8, weight="normal"):
        t = fig.text(0, 0, s, fontsize=fontsize, fontweight=weight, transform=fig.transFigure, alpha=0)
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        w_px = t.get_window_extent(renderer=r).width; t.remove()
        return w_px / fig.bbox.width

    def _text_height_frac(fig, s, *, fontsize=8, weight="normal"):
        t = fig.text(0, 0, s, fontsize=fontsize, fontweight=weight, transform=fig.transFigure, alpha=0)
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        h_px = t.get_window_extent(renderer=r).height; t.remove()
        return h_px / fig.bbox.height

    # chips — max_per_row + slightly tighter spacing
    def chip_row_exact(fig, items, y, bg, *, fs=10.1, weight="900", max_rows=2, gap_x=0.006, max_per_row=None):
        if not items: return y
        x0 = x = CHIP_X0
        row_gap = 0.026
        pad_x = 0.004
        pad_y = 0.002
        h = _text_height_frac(fig, "Hg", fontsize=fs, weight=weight) + pad_y*2
        per_row = 0
        for s in items[:60]:
            w = _text_width_frac(fig, s, fontsize=fs, weight=weight) + pad_x*2
            need_wrap = (x + w > 0.965) or (max_per_row and per_row >= max_per_row)
            if need_wrap:
                max_rows -= 1
                if max_rows <= 0: break
                x = x0; y -= row_gap; per_row = 0
            fig.patches.append(
                mpatches.FancyBboxPatch((x, y - h*0.74), w, h,
                    boxstyle=f"round,pad=0.001,rounding_size={h*0.45}",
                    transform=fig.transFigure, facecolor=bg, edgecolor="none")
            )
            fig.text(x + pad_x, y - h*0.33, s, fontsize=fs, color="#FFFFFF",
                     va="center", ha="left", fontweight=weight)
            x += w + gap_x
            per_row += 1
        return y - row_gap

    # roles row — slightly squarer corners
    def roles_row_tight(fig, rs: dict, y, *, fs=10.6):
        if not isinstance(rs, dict) or not rs: return y
        rs = {k: v for k, v in rs.items() if k.strip().lower() != "all in"}
        if not rs: return y

        x0 = x = CHIP_X0
        row_gap = 0.041
        gap = 0.003
        pad_x = 0.006
        pad_y = 0.003

        for r, v in sorted(rs.items(), key=lambda kv: -kv[1])[:12]:
            text_w = _text_width_frac(fig, r, fontsize=fs, weight="800")
            text_h = _text_height_frac(fig, "Hg", fontsize=fs, weight="800")
            role_w = text_w + pad_x*2
            role_h = text_h + pad_y*2

            num_text = f"{int(round(v))}"
            num_wt = _text_width_frac(fig, num_text, fontsize=fs-0.6, weight="900")
            num_ht = _text_height_frac(fig, "Hg", fontsize=fs-0.6, weight="900")
            num_w  = num_wt + pad_x*2 * 0.9
            num_h  = num_ht + pad_y*2 * 0.9

            total = role_w + gap + num_w
            if x + total > 0.965:
                x = x0; y -= row_gap

            fig.patches.append(mpatches.FancyBboxPatch((x, y - role_h*0.78), role_w, role_h,
                              boxstyle=f"round,pad=0.001,rounding_size={role_h*0.25}",
                              transform=fig.transFigure, facecolor=ROLE_GREY, edgecolor="none"))
            fig.text(x + pad_x, y - role_h*0.33, r, fontsize=fs, color="#FFFFFF",
                     va="center", ha="left", fontweight="800")

            R,G,B = [int(255*c) for c in div_color_tuple(v)]
            bx = x + role_w + gap
            fig.patches.append(mpatches.FancyBboxPatch((bx, y - num_h*0.78), num_w, num_h,
                              boxstyle=f"round,pad=0.001,rounding_size={num_h*0.25}",
                              transform=fig.transFigure, facecolor=f"#{R:02x}{G:02x}{B:02x}", edgecolor="none"))
            fig.text(bx + num_w/2, y - num_h*0.33, num_text, fontsize=fs-0.6, color="#FFFFFF",
                     va="center", ha="center", fontweight="900")

            x = bx + num_w + 0.010
        return y - row_gap

    # percentiles + actuals
    def pct_of(metric: str) -> float:
        if isinstance(pct_extra, dict) and metric in pct_extra and pd.notna(pct_extra[metric]):
            return float(pct_extra[metric])
        col = f"{metric} Percentile"
        if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
            return float(player_row[col].iloc[0])
        return np.nan

    def val_of(metric: str):
        ply = player_row.iloc[0]
        if metric not in ply.index or pd.isna(ply[metric]): return np.nan, "—"
        v = float(ply[metric]); m = metric.lower()
        if "%" in metric or "percent" in m: return v, f"{int(round(v))}%"
        if "per 90" in m or "xg" in m or "xa" in m: return v, f"{v:.2f}"
        return v, f"{v:.2f}"

    # -------- exact same pixel bar height & gap; panel height flexes with row count --------
    BAR_PX = 24
    GAP_PX = 6
    SEP_PX = 2
    STEP_PX = BAR_PX + GAP_PX

    LABEL_FS    = 10.6
    VALUE_FS    = 8.5
    TITLE_FS    = 20

    def bar_panel(fig, left, top, width, n_rows, title, triples):
        """Panel with left gutter (labels + title share the same left start)."""
        fig.canvas.draw()
        fig_px_h = fig.bbox.height

        # panel height in fig fraction
        ax_h_frac = (n_rows * STEP_PX) / fig_px_h
        bottom = top - ax_h_frac

        # Compute max label width to size the gutter
        labels = [t[0] for t in triples]
        max_label_w_frac = max(_text_width_frac(fig, s, fontsize=LABEL_FS, weight="bold") for s in labels) if labels else 0
        gutter_w = max_label_w_frac + GUTTER_PAD

        # Panel background (full width)
        ax_panel = fig.add_axes([left, bottom, width, ax_h_frac])
        ax_panel.set_facecolor(PANEL_BG)
        ax_panel.set_xticks([]); ax_panel.set_yticks([])
        for sp in ax_panel.spines.values(): sp.set_visible(False)

        # Bars axis (to the right of the gutter)
        bar_left  = left + gutter_w
        bar_width = max(0.001, width - gutter_w - 0.004)  # tiny right margin
        ax = fig.add_axes([bar_left, bottom, bar_width, ax_h_frac])
        ax.set_facecolor(PANEL_BG)

        pcts  = [float(np.nan_to_num(t[1], nan=0.0)) for t in triples]
        texts = [t[2] for t in triples]
        n = len(labels)

        bar_du = BAR_PX / STEP_PX
        gap_du = GAP_PX / STEP_PX
        sep_du = SEP_PX / STEP_PX

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, n - 0.5)
        y_idx = np.arange(n)[::-1]

        # tracks
        track_h = bar_du + gap_du - sep_du
        for yi in y_idx:
            ax.add_patch(mpatches.Rectangle((0, yi - track_h/2), 100, track_h,
                                            facecolor=TRACK_BG, edgecolor='none'))

        # bars + value labels
        for yi, v, t in zip(y_idx, pcts, texts):
            ax.add_patch(mpatches.Rectangle((0, yi - bar_du/2), v, bar_du,
                                            facecolor=div_color_tuple(v), edgecolor='none'))
            ax.text(1.0, yi, t, va="center", ha="left", color="#0B0B0B", fontsize=VALUE_FS + 0.5, weight="700")

        # clean axis
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=0)
        ax.grid(False)

        # midline
        ax.axvline(50, color="#94A3B8", linestyle=":", linewidth=1.2, zorder=2)

        # metric labels in gutter (left-aligned)
        for yi, lab in zip(y_idx, labels):
            y_fig = bottom + ax_h_frac * ((yi + 0.5) / max(1, n))
            fig.text(left + GUTTER_PAD/2, y_fig, lab,
                     color=TEXT, fontsize=LABEL_FS, fontweight="bold",
                     va="center", ha="left")

        # title aligned to the same gutter start
        title_y = bottom + ax_h_frac + 0.008
        fig.text(left + GUTTER_PAD/2, title_y, title,
                 color=TEXT, fontsize=TITLE_FS, fontweight="900", ha="left", va="bottom")
        ax.plot([0, 1], [1, 1], transform=ax.transAxes, color="#94A3B8", linewidth=0.8, alpha=0.35)

        return bottom

    # ----------------- figure & header -----------------
    W, H = 1500, 1080
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor(PAGE_BG)

    ply = player_row.iloc[0]
    team   = str(ply.get("Team","?"))
    league = str(ply.get("League","?"))
    pos    = str(ply.get("Position","?"))
    age    = int(ply["Age"]) if pd.notna(ply.get("Age")) else None
    mins   = int(ply.get("Minutes played", np.nan)) if pd.notna(ply.get("Minutes played")) else None
    matches= int(ply.get("Matches played", np.nan)) if pd.notna(ply.get("Matches played")) else None
    goals  = int(ply.get("Goals", np.nan)) if pd.notna(ply.get("Goals")) else 0

    if "xG" in ply.index and pd.notna(ply["xG"]):
        xg_total = float(ply["xG"])
    else:
        xg_per90 = float(ply.get("xG per 90", np.nan)) if pd.notna(ply.get("xG per 90")) else np.nan
        xg_total = float(xg_per90) * (float(mins) / 90.0) if (pd.notna(xg_per90) and mins) else np.nan
    xg_total_str = f"{xg_total:.2f}" if pd.notna(xg_total) else "—"
    assists= int(ply.get("Assists", np.nan)) if pd.notna(ply.get("Assists")) else 0

    # Name + league-adjusted badge
    name_fs = 28
    name_text = fig.text(NAME_X, 0.962, f"{player_name}", color="#FFFFFF",
                         fontsize=name_fs, fontweight="900", va="top", ha="left")
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    name_bbox = name_text.get_window_extent(renderer=r)
    name_w_frac = name_bbox.width / fig.bbox.width
    name_h_frac = name_bbox.height / fig.bbox.height
    badge_x = NAME_X + name_w_frac + 0.010

    if isinstance(role_scores, dict) and role_scores:
        _, best_val_raw = max(role_scores.items(), key=lambda kv: kv[1])
        _ls_map = globals().get("LEAGUE_STRENGTHS", {})
        league_strength = float(_ls_map.get(league, 50.0))
        BETA_BADGE = 0.40
        best_val_adj = (1.0 - BETA_BADGE) * float(best_val_raw) + BETA_BADGE * league_strength

        R, G, B = [int(255*c) for c in div_color_tuple(best_val_adj)]
        bh = name_h_frac; bw = bh; by = 0.962 - bh
        fig.patches.append(mpatches.FancyBboxPatch(
            (badge_x, by), bw, bh,
            boxstyle="round,pad=0.001,rounding_size=0.011",
            transform=fig.transFigure,
            facecolor=f"#{R:02x}{G:02x}{B:02x}", edgecolor="none"
        ))
        fig.text(badge_x + bw/2, by + bh/2 - 0.0005, f"{int(round(best_val_adj))}",
                 fontsize=18.6, color="#FFFFFF", va="center", ha="center", fontweight="900")

    # Meta row (more left padding)
    x_meta = META_X; y_meta = 0.905; gap = 0.004
    runs = [
        (f"{pos} — ", "normal"),
        (team, "bold"),
        (" — ", "normal"),
        (league, "bold"),
        (f" — Age {age if age else '—'} — Minutes {mins if mins else '—'} — "
         f"Matches {matches if matches else '—'} — Goals {goals} — xG {xg_total_str} — Assists {assists}", "normal")
    ]
    for txt, weight in runs:
        fig.text(x_meta, y_meta, txt, color="#FFFFFF", fontsize=13,
                 fontweight=("900" if weight == "bold" else "normal"), ha="left", va="center")
        x_meta += _text_width_frac(fig, txt, fontsize=13.5,
                                   weight=("900" if weight == "bold" else "normal")) + (gap if txt.strip() else 0)

    # ----------------- chips + roles -----------------
    y = 0.868  # a touch lower to create more breathing room under meta
    y = chip_row_exact(fig, strengths or [],  y, CHIP_G_BG, fs=10.1, max_per_row=5)
    y = chip_row_exact(fig, weaknesses or [], y, CHIP_R_BG, fs=10.1, max_per_row=5)
    y = chip_row_exact(fig, styles or [],     y, CHIP_B_BG, fs=10.1, max_per_row=5)
    y -= 0.015
    y = roles_row_tight(fig, role_scores if isinstance(role_scores, dict) else {}, y, fs=10.6)

    # ----------------- metric groups -----------------
    ATTACKING = []
    for lab, met in [
        ("Crosses", "Crosses per 90"),
        ("Crossing %", "Accurate crosses, %"),
        ("Goals: Non-Penalty", "Non-penalty goals per 90"),
        ("xG", "xG per 90"),
        ("Expected Assists", "xA per 90"),
        ("Offensive Duels", "Offensive duels per 90"),
        ("Offensive Duel %", "Offensive duels won, %"),
        ("Shots", "Shots per 90"),
        ("Shooting %", "Shots on target, %"),
        ("Touches in box", "Touches in box per 90"),
    ]: ATTACKING.append((lab, pct_of(met), val_of(met)[1]))

    DEFENSIVE = []
    for lab, met in [
        ("Aerial Duels", "Aerial duels per 90"),
        ("Aerial Win %", "Aerial duels won, %"),
        ("Defensive Duels", "Defensive duels per 90"),
        ("Defensive Duel %", "Defensive duels won, %"),
        ("PAdj Interceptions", "PAdj Interceptions"),
        ("Shots blocked", "Shots blocked per 90"),
    ]: DEFENSIVE.append((lab, pct_of(met), val_of(met)[1]))

    POSSESSION = []
    for lab, met in [
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
        ("Passes to F3rd", "Passes to final third per 90"),
        ("Passes F3rd %", "Accurate passes to final third, %"),
        ("Passes Pen-Area", "Passes to penalty area per 90"),
        ("Pass Pen-Area %", "Accurate passes to penalty area, %"),
        ("Progessive Passes", "Progressive passes per 90"),
        ("Prog Pass %", "Accurate progressive passes, %"),
        ("Progressive Runs", "Progressive runs per 90"),
        ("Smart Passes", "Smart passes per 90"),
    ]: POSSESSION.append((lab, pct_of(met), val_of(met)[1]))

    # ----------------- layout (wider cards, smaller middle gap) -----------------
    LEFT = 0.050
    WIDTH_L = 0.41
    MID_GAP = 0.040
    RIGHT = LEFT + WIDTH_L + MID_GAP
    WIDTH_R = 0.41

    TOP = 0.66
    V_GAP_FRAC = 0.050

    # Left column
    att_bottom = bar_panel(fig, LEFT, TOP, WIDTH_L, len(ATTACKING), "Attacking",  ATTACKING)
    def_bottom = bar_panel(fig, LEFT, att_bottom - V_GAP_FRAC, WIDTH_L, len(DEFENSIVE), "Defensive", DEFENSIVE)

    # Right column
    _ = bar_panel(fig, RIGHT, TOP, WIDTH_R, len(POSSESSION), "Possession", POSSESSION)

    # ----------------- render + download -----------------
    st.pyplot(fig, use_container_width=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    st.download_button("⬇️ Download one-pager (PNG)",
                       data=buf.getvalue(),
                       file_name=f"{str(player_name).replace(' ','_')}_onepager.png",
                       mime="image/png")

# ============================ END — WIDER PANELS, SMALLER CENTER GAP, EXTRA TOP-LEFT PADDING ============================

# ============================ (F) THREE-PANEL PERCENTILE BOARD — Uniform rows + visible gridlines (numbers centered; custom % at 0/100) ============================
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation  # pixel-like offsets

st.markdown("---")
st.header("📋 Feature F — Percentile Board (uniform rows)")

# --- NEW: footer label controls ---
_footer_default = "Percentile Rank"
_edit_footer = st.toggle("Edit footer label", value=False)
if _edit_footer:
    footer_label = st.text_input("Footer label", value=_footer_default)
else:
    footer_label = _footer_default
# --- END NEW ---

if player_row.empty:
    st.info("Pick a player above.")
else:
    # ----- assemble sections from your existing calcs -----
    ATTACKING = []
    for lab, met in [
        ("Crosses", "Crosses per 90"),
        ("Crossing Accuracy %", "Accurate crosses, %"),
        ("Goals: Non-Penalty", "Non-penalty goals per 90"),
        ("xG", "xG per 90"),
        ("Expected Assists", "xA per 90"),
        ("Offensive Duels", "Offensive duels per 90"),
        ("Offensive Duel Success %", "Offensive duels won, %"),
        ("Progressive Runs", "Progressive runs per 90"),
        ("Shots", "Shots per 90"),
        ("Touches in Opposition Box", "Touches in box per 90"),
    ]:
        ATTACKING.append((lab, float(np.nan_to_num(pct_of(met), nan=0.0)), val_of(met)[1]))

    DEFENSIVE = []
    for lab, met in [
        ("Aerial Duels", "Aerial duels per 90"),
        ("Aerial Duel Success %", "Aerial duels won, %"),
        ("Defensive Duels", "Defensive duels per 90"),
        ("Defensive Duel Success %", "Defensive duels won, %"),
        ("Shots Blocked", "Shots blocked per 90"),
        ("PAdj. Interceptions", "PAdj Interceptions"),
    ]:
        DEFENSIVE.append((lab, float(np.nan_to_num(pct_of(met), nan=0.0)), val_of(met)[1]))

    POSSESSION = []
    for lab, met in [
        ("Deep Completions", "Deep completions per 90"),
        ("Dribbles", "Dribbles per 90"),
        ("Dribbling Success %", "Successful dribbles, %"),
        ("Forward Passes", "Forward passes per 90"),
        ("Forward Passing %", "Accurate forward passes, %"),
        ("Key passes", "Key passes per 90"),
        ("Long Passes", "Long passes per 90"),
        ("Long Passing %", "Accurate long passes, %"),
        ("Passes", "Passes per 90"),
        ("Passing %", "Accurate passes, %"),
        ("Passes to Final 3rd", "Passes to final third per 90"),
        ("Passes to Final 3rd %", "Accurate passes to final third, %"),
        ("Passes to Penalty Area", "Passes to penalty area per 90"),
        ("Pass to Penalty Area %", "Accurate passes to penalty area, %"),
        ("Progessive Passes", "Progressive passes per 90"),
        ("Progessive Passing %", "Accurate progressive passes, %"),
        ("Progressive Runs", "Progressive runs per 90"),
        ("Smart Passes", "Smart passes per 90"),
    ]:
        POSSESSION.append((lab, float(np.nan_to_num(pct_of(met), nan=0.0)), val_of(met)[1]))

    sections = [("Attacking", ATTACKING), ("Defensive", DEFENSIVE), ("Possession", POSSESSION)]
    sections = [(t, lst) for t, lst in sections if lst]

    # ----- styling (dark Tableau-ish canvas) -----
    PAGE_BG = "#0a0f1c"
    AX_BG   = "#0f151f"
    TRACK   = "#1b2636"
    TITLE   = "#f3f5f7"
    LABEL   = "#e8eef8"
    DIVIDER = "#ffffff"

    # Tableau-like diverging ramp (0→red, 50→gold, 100→green)
    TAB_RED   = np.array([199, 54, 60], dtype=float)    # #C7363C
    TAB_GOLD  = np.array([240, 197, 106], dtype=float)  # #F0C56A
    TAB_GREEN = np.array([61, 166, 91], dtype=float)    # #3DA65B

    def _blend(c1, c2, t):
        c = c1 + (c2 - c1) * np.clip(t, 0.0, 1.0)
        return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"

    def pct_to_rgb(v):
        v = float(np.clip(v, 0, 100))
        return _blend(TAB_RED, TAB_GOLD, v/50.0) if v <= 50 else _blend(TAB_GOLD, TAB_GREEN, (v-50.0)/50.0)

    # ----- layout: identical bar height across all sections -----
    total_rows = sum(len(lst) for _, lst in sections)
    fig = plt.figure(figsize=(10, 8), dpi=100)  # 1000x800 px
    fig.patch.set_facecolor(PAGE_BG)

    left_margin  = 0.035
    right_margin = 0.020
    top_margin   = 0.035
    bot_margin   = 0.095
    header_h     = 0.06
    gap_between  = 0.020

    rows_space_total = 1 - (top_margin + bot_margin) - header_h * len(sections) - gap_between * (len(sections) - 1)
    row_slot = rows_space_total / max(total_rows, 1)
    BAR_FRAC = 0.85

    # label gutter width
    probe = fig.text(0, 0, "Successful Defensive Actions", fontsize=11, fontweight="bold", color=LABEL, alpha=0)
    fig.canvas.draw()
    lab_w = probe.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.bbox.width
    probe.remove()
    gutter = 0.215


    ticks = np.arange(0, 101, 10)  # 0,10,...,100

    # visual center for footer text
    x_center_plot = (left_margin + gutter + (1 - right_margin)) / 2.0

    def draw_panel(panel_top, title, tuples, *, show_xticks=False, draw_bottom_divider=True):
        n = len(tuples)
        panel_h = header_h + n * row_slot

        # Section title
        fig.text(left_margin, panel_top - 0.012, title, ha="left", va="top",
                 fontsize=20, fontweight="900", color=TITLE)

        # Bars axis
        ax = fig.add_axes([
            left_margin + gutter,
            panel_top - header_h - n*row_slot,
            1 - left_margin - right_margin - gutter,
            n * row_slot
        ])
        ax.set_facecolor(AX_BG)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, n - 0.5)

        # Hide default spines/ticks; draw custom
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False, length=0)

        # ---- Tracks ----
        for i in range(n):
            y = i
            ax.add_patch(plt.Rectangle((0, y - (BAR_FRAC/2)), 100, BAR_FRAC,
                                       color=TRACK, ec="none", zorder=0.5))

        # ---- Vertical gridlines at each 10% ----
        for gx in ticks:
            ax.vlines(gx, -0.5, n - 0.5, colors=(1, 1, 1, 0.16), linewidth=0.8, zorder=0.75)

        # ---- Bars + value labels ----
        for i, (lab, pct, val_str) in enumerate(tuples[::-1]):  # reverse for top-first
            y = i
            bar_w = max(0.0, min(100.0, float(pct)))
            ax.add_patch(plt.Rectangle((0, y - (BAR_FRAC/2)), bar_w, BAR_FRAC,
                                       color=pct_to_rgb(bar_w), ec="none", zorder=1.0))
            ax.text(1.0, y, val_str, ha="left", va="center",
                    fontsize=8, fontweight="400", color="#0B0B0B", zorder=2.0)

        # ---- Dotted 50% reference line (over bars) ----
        ax.axvline(50, color="#FFFFFF", ls=(0, (4, 4)), lw=1.5, alpha=0.85, zorder=3.5)

        # Metric labels in left gutter
        for i, (lab, _, _) in enumerate(tuples[::-1]):
            y_fig = (panel_top - header_h - n*row_slot) + ((i + 0.5) * row_slot)
            fig.text(left_margin, y_fig, lab, ha="left", va="center",
                     fontsize=10, fontweight="bold", color=LABEL)

        # ---- Manually centered bottom ticks ONLY on last panel ----
        if show_xticks:
            trans = ax.get_xaxis_transform()  # x in data, y in axis coords

            # Adjustable offsets in points (pt) → convert to inches via /72
            INNER_PCT_OFFSET_PT    = 7   # offset for the "%" on inner ticks (keeps digits visually centered)
            EDGE_PCT_OFFSET_0_PT   = 4   # offset for "%" at 0  (push right)
            EDGE_PCT_OFFSET_100_PT = 10   # offset for "%" at 100 (push right)

            offset_inner = ScaledTranslation(INNER_PCT_OFFSET_PT/72, 0, fig.dpi_scale_trans)
            offset_pct_0 = ScaledTranslation(EDGE_PCT_OFFSET_0_PT/72, 0, fig.dpi_scale_trans)
            offset_pct_100 = ScaledTranslation(EDGE_PCT_OFFSET_100_PT/72, 0, fig.dpi_scale_trans)

            y_label = -0.075

            for gx in ticks:
                # tiny tick mark
                ax.plot([gx, gx], [-0.03, 0.0], transform=trans,
                        color=(1, 1, 1, 0.6), lw=1.1, clip_on=False, zorder=4)
                # number centered on gridline
                ax.text(gx, y_label, f"{int(gx)}", transform=trans,
                        ha="center", va="top", fontsize=10, fontweight="700",
                        color="#FFFFFF", zorder=4, clip_on=False)
                # percent sign with custom offsets
                if gx == 0:
                    ax.text(gx, y_label, "%", transform=trans + offset_pct_0,
                            ha="left", va="top", fontsize=10, fontweight="700",
                            color="#FFFFFF", zorder=4, clip_on=False)
                elif gx == 100:
                    ax.text(gx, y_label, "%", transform=trans + offset_pct_100,
                            ha="left", va="top", fontsize=10, fontweight="700",
                            color="#FFFFFF", zorder=4, clip_on=False)
                else:
                    ax.text(gx, y_label, "%", transform=trans + offset_inner,
                            ha="left", va="top", fontsize=10, fontweight="700",
                            color="#FFFFFF", zorder=4, clip_on=False)

        # Section divider
        if draw_bottom_divider:
            y0 = panel_top - panel_h - 0.008
            fig.lines.append(plt.Line2D([left_margin, 1 - right_margin], [y0, y0],
                                        transform=fig.transFigure, color=DIVIDER, lw=1.2, alpha=0.95))
        return panel_top - panel_h - gap_between

    # Render panels; only the last shows tick labels
    y_top = 1 - top_margin
    for idx, (title, data) in enumerate(sections):
        is_last = (idx == len(sections) - 1)
        y_top = draw_panel(y_top, title, data, show_xticks=is_last, draw_bottom_divider=not is_last)

    # Bottom caption — slightly lower
    fig.text(x_center_plot, bot_margin * 0.38, footer_label,
             ha="center", va="center", fontsize=11, fontweight="bold", color=LABEL)

    st.pyplot(fig, use_container_width=True)

    # download
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    st.download_button("⬇️ Download Feature F (PNG)",
                       data=buf.getvalue(),
                       file_name=f"{str(player_name).replace(' ','_')}_featureF.png",
                       mime="image/png")
# ============================ END — Feature F ============================

# ============================ (Z) THREE-PANEL PERCENTILE BOARD — safe headroom + tight, even badges ============================
from io import BytesIO
import uuid, numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from PIL import Image
import streamlit as st

# ============================
# ✅ NEW: auto image fetch + FotMob resolver (same as Pro Layout method)
# ============================
import io
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
import requests

# --- NEW: league logos lookup (your new page) ---
try:
    from league_logo_urls import get_league_logo_url
except Exception:
    def get_league_logo_url(_league: str) -> str:
        return ""

# --- NEW: optional FotMob team URL mapping (if you have it) ---
try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS as _FZ_FOTMOB_TEAM_URLS
except Exception:
    _FZ_FOTMOB_TEAM_URLS = {}

# optional: external getter if you have it
try:
    from team_fotmob_urls import get_fotmob_url as _external_get_fotmob_url  # returns "" if not found
except Exception:
    _external_get_fotmob_url = None


st.markdown("---")
st.header("📋 Feature Z — White Percentile Board")

with st.expander("Feature Z options", expanded=False):
    enable_images = st.checkbox("Add header images", value=True)
    show_height   = st.checkbox("Show height in info row", value=True)
    name_override_on = st.checkbox("Edit player display name", value=False)
    name_override    = st.text_input("Display name", "", disabled=not name_override_on)

    # --- Height (existing pattern) ---
    default_height = ""
    try:
        if not player_row.empty:
            for col in ["Height","Height (ft)","Height ft","Height (cm)"]:
                if col in player_row.columns and str(player_row.iloc[0][col]).strip():
                    default_height = str(player_row.iloc[0][col]).strip(); break
    except Exception: pass
    height_text = st.text_input("Height value (e.g., 6'2\")", default_height)

    # --- NEW: editable footer caption (toggle) ---
    _CAPTION_DEFAULT = "Percentile Rank"
    _edit_footer = st.toggle("Edit footer caption", value=False, key="fz_edit_footer")
    footer_caption_text = st.text_input("Footer caption", _CAPTION_DEFAULT, disabled=not _edit_footer, key="fz_footer_text")

    # --- NEW: Edit 'Foot' in information row (like Height) ---
    default_foot = ""
    try:
        if not player_row.empty:
            for col in ["Foot","Preferred Foot"]:
                if col in player_row.columns and str(player_row.iloc[0][col]).strip():
                    default_foot = str(player_row.iloc[0][col]).strip(); break
    except Exception: pass
    foot_override_on = st.checkbox("Edit foot in info row", value=False, key="fz_foot_edit")
    foot_override_text = st.text_input("Foot value (e.g., Left)", default_foot, disabled=not foot_override_on, key="fz_foot_text")

    if enable_images:
        # ✅ NEW: auto images toggle (uploads still override)
        auto_images = st.checkbox(
            "Auto header images (league logo / team badge / player photo) if not uploaded",
            value=True,
            key="fz_auto_images_toggle"
        )

        # ✅ NEW: option to remove ONLY player photo (leftmost badge)
        hide_player_photo = st.checkbox("Hide player photo (left badge)", value=False, key="fz_hide_player_photo")

        st.caption("Upload up to three header images (PNG recommended). Rightmost is the anchor.")
        up_img1 = st.file_uploader("Image 1 (rightmost)", type=["png","jpg","jpeg","webp"], key="fz_img1")
        up_img2 = st.file_uploader("Image 2 (middle)",   type=["png","jpg","jpeg","webp"], key="fz_img2")
        up_img3 = st.file_uploader("Image 3 (leftmost)", type=["png","jpg","jpeg","webp"], key="fz_img3")

        # Spacing presets
        spacing_preset = st.selectbox(
            "Badge spacing",
            ["Tight (default)", "Tight +", "Medium", "Wide"],
            index=0,
            help="Keeps equal gaps; each step is a little wider than the previous."
        )

        # --- NEW: per-image horizontal fine-tune (figure fraction; negative=left, positive=right) ---
        st.caption("Fine-tune each image’s horizontal position (− left, + right).")
        img1_dx = st.slider("Shift Image 1 (rightmost)", min_value=-0.05, max_value=0.05, value=0.00, step=0.001, key="fz_dx_img1")
        img2_dx = st.slider("Shift Image 2 (middle)",    min_value=-0.05, max_value=0.05, value=0.00, step=0.001, key="fz_dx_img2")
        img3_dx = st.slider("Shift Image 3 (leftmost)",  min_value=-0.05, max_value=0.05, value=0.00, step=0.001, key="fz_dx_img3")
    else:
        up_img1 = up_img2 = up_img3 = None
        spacing_preset = "Tight (default)"
        img1_dx = img2_dx = img3_dx = 0.0
        auto_images = False
        hide_player_photo = False


def _safe_get(df_or_series, key, default="—"):
    try:
        if hasattr(df_or_series, "iloc"): v = df_or_series.iloc[0].get(key, default)
        else:                              v = df_or_series.get(key, default)
        s = "" if v is None else str(v)
        return default if s.strip() == "" else s
    except Exception:
        return default

def _font_name_or_fallback(pref, fallback="DejaVu Sans"):
    installed = {f.name for f in fm.fontManager.ttflist}
    for n in pref:
        if n in installed: return n
    return fallback

FONT_TITLE_FAMILY = _font_name_or_fallback(["Tableau Bold","Tableau Sans Bold","Tableau"])
FONT_BOOK_FAMILY  = _font_name_or_fallback(["Tableau Book","Tableau Sans","Tableau"])
TITLE_FP     = FontProperties(family=FONT_TITLE_FAMILY, weight='bold',     size=24)
H2_FP        = FontProperties(family=FONT_TITLE_FAMILY, weight='semibold', size=20)
LABEL_FP     = FontProperties(family=FONT_BOOK_FAMILY,  weight='medium',   size=10)
INFO_LABEL_FP= FontProperties(family=FONT_BOOK_FAMILY,  weight='bold',     size=10)
INFO_VALUE_FP= FontProperties(family=FONT_BOOK_FAMILY,  weight='regular',  size=10)
BAR_VALUE_FP = FontProperties(family=FONT_BOOK_FAMILY,  weight='regular',  size=8)
TICK_FP      = FontProperties(family=FONT_BOOK_FAMILY,  weight='medium',   size=10)
FOOTER_FP    = FontProperties(family=FONT_BOOK_FAMILY,  weight='medium', size=10)


# ============================
# ✅ NEW: auto image utilities
# ============================
def _try_load_img(url: str):
    """
    Returns an image array for a valid URL, else None.
    Robust: matplotlib first, PIL fallback.
    """
    if not url or not (str(url).startswith("http://") or str(url).startswith("https://")):
        return None
    try:
        r = requests.get(str(url), timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.content:
            return None

        try:
            return plt.imread(io.BytesIO(r.content))
        except Exception:
            pass

        try:
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            return np.array(im)
        except Exception:
            return None
    except Exception:
        return None

def _npimg_to_pil_rgba(arr):
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        if a.shape[-1] == 4:
            if a.dtype != np.uint8:
                a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(a, "RGBA")
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(a, "RGB").convert("RGBA")
    except Exception:
        return None

BADGE_DIRS = [Path.cwd() / "badges", Path.cwd() / "crests"]
for d in BADGE_DIRS:
    try: d.mkdir(exist_ok=True)
    except Exception: pass

def _clean_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").lower()).strip("_")

@st.cache_data(show_spinner=False)
def load_local_badge(team: str):
    key = _clean_filename(team)
    if not key:
        return None
    for folder in BADGE_DIRS:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = folder / f"{key}{ext}"
            if p.exists():
                try:
                    return plt.imread(str(p))
                except Exception:
                    continue
    return None

def _norm(s: str) -> str:
    if not s: return ""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii").strip().lower()

def get_fotmob_url(team: str) -> str:
    """
    Priority:
    1) dict from team_fotmob_urls (FOTMOB_TEAM_URLS)
    2) external get_fotmob_url(team) if present
    """
    t = _norm(team)
    for k, v in (_FZ_FOTMOB_TEAM_URLS or {}).items():
        if _norm(k) == t and str(v).strip():
            return str(v).strip()
    if _external_get_fotmob_url is not None:
        try:
            u = _external_get_fotmob_url(team) or ""
            return str(u).strip()
        except Exception:
            pass
    return ""

def _fotmob_team_id_from_url(team_url: str) -> str:
    m = re.search(r"/teams/(\d+)/", str(team_url or ""))
    return m.group(1) if m else ""

def _fotmob_crest_url_from_team_url(team_url: str) -> str:
    tid = _fotmob_team_id_from_url(team_url)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png" if tid else ""

def _fotmob_crest_url(team: str) -> str:
    team_url = get_fotmob_url(team)
    return _fotmob_crest_url_from_team_url(team_url) if team_url else ""

def load_fotmob_crest(team: str):
    url = _fotmob_crest_url(team)
    if not url:
        return None
    return _try_load_img(url)

def _auto_team_badge(player_row_like):
    team = _safe_get(player_row_like, "Team", "").strip()
    if not team:
        return None
    img = load_local_badge(team)
    if img is not None:
        return img
    crest = load_fotmob_crest(team)
    if crest is not None:
        return crest
    return None

def _auto_league_logo(player_row_like):
    league = _safe_get(player_row_like, "League", "").strip()
    url = (get_league_logo_url(league) or "").strip()
    if not url:
        return None
    return _try_load_img(url)

# ============================
# ✅ NEW: FotMob player photo resolver (teams API method)
# IMPORTANT: if it can't load -> return None (blank), NO placeholder
# ============================
def _player_surname(player: str) -> str:
    p = (player or "").strip()
    if not p:
        return ""
    if "," in p:
        return p.split(",", 1)[0].strip()
    parts = p.split()
    return parts[-1].strip() if parts else ""

def _slug_name(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    repl = {
        "ø":"o","œ":"oe","æ":"ae","å":"a","ä":"a","ö":"o","ü":"u",
        "ß":"ss","ł":"l","đ":"d","ð":"d","þ":"th","ç":"c",
        "ş":"s","ğ":"g","ı":"i",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

@st.cache_data(show_spinner=False, ttl=60*60*6)
def _fotmob_team_squad(team_id: str) -> list[dict]:
    squad: list[dict] = []
    if not team_id:
        return squad
    try:
        url = f"https://www.fotmob.com/api/teams?id={team_id}"
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return squad
        data = r.json() or {}
        raw_squad = data.get("squad", None)

        if isinstance(raw_squad, list):
            for section in raw_squad:
                members = section.get("members") or section.get("players") or []
                if isinstance(members, list):
                    squad.extend([m for m in members if isinstance(m, dict)])

        elif isinstance(raw_squad, dict):
            for k in ("members", "players"):
                members = raw_squad.get(k)
                if isinstance(members, list):
                    squad.extend([m for m in members if isinstance(m, dict)])

            nested = raw_squad.get("squad")
            if isinstance(nested, list):
                for section in nested:
                    members = section.get("members") or section.get("players") or []
                    if isinstance(members, list):
                        squad.extend([m for m in members if isinstance(m, dict)])
    except Exception:
        squad = []
    return squad

def _resolve_fotmob_player_photo_url(player: str, team: str) -> str:
    team_url = get_fotmob_url(team)
    tid = _fotmob_team_id_from_url(team_url)
    if not tid:
        return ""
    squad = _fotmob_team_squad(tid)

    target_surname = _slug_name(_player_surname(player))
    target_full    = _slug_name(player)

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
        best_score = 0.0
        best_pid = ""
        for m in squad:
            name = m.get("name") or m.get("playerName") or ""
            pid = m.get("id") or m.get("playerId") or m.get("primaryId") or ""
            if not pid:
                continue
            sn = _slug_name(_player_surname(name))
            sc = _similar(sn, target_surname)
            if sc > best_score:
                best_score = sc
                best_pid = str(pid)
        if best_score >= 0.86:
            best_id = best_pid

    if best_id and str(best_id).isdigit():
        return f"https://images.fotmob.com/image_resources/playerimages/{best_id}.png"
    return ""

def _auto_player_photo(player_row_like):
    player_name = _safe_get(player_row_like, "Player", _safe_get(player_row_like, "Name", "")).strip()
    team = _safe_get(player_row_like, "Team", "").strip()
    if not player_name or not team:
        return None
    url = _resolve_fotmob_player_photo_url(player_name, team)
    if not url:
        return None
    return _try_load_img(url)  # may be None -> blank


if player_row.empty:
    st.info("Pick a player above.")
else:
    pos   = _safe_get(player_row, "Position", "CM/DM/RW")
    name_ = _safe_get(player_row, "Player", _safe_get(player_row, "Name", "Kadeem Harris"))
    if name_override_on and name_override.strip(): name_ = name_override.strip()
    team  = _safe_get(player_row, "Team", "Carlisle United")
    age_raw = _safe_get(player_row, "Age", "31.0")
    try: age = f"{float(age_raw):.0f}"
    except Exception: age = age_raw
    games   = _safe_get(player_row, "Matches played", _safe_get(player_row, "Games", _safe_get(player_row, "Apps", "—")))
    minutes = _safe_get(player_row, "Minutes", _safe_get(player_row, "Minutes played", "—"))
    goals   = _safe_get(player_row, "Goals", "—")
    assists = _safe_get(player_row, "Assists", "—")
    foot    = _safe_get(player_row, "Foot", _safe_get(player_row, "Preferred Foot", "—"))

    foot_display = (foot_override_text.strip() if (foot_override_on and foot_override_text and foot_override_text.strip()) else foot)

    # === sections (unchanged) ===
    ATTACKING = []
    for lab, met in [
        ("Crosses", "Crosses per 90"),
        ("Crossing Accuracy %", "Accurate crosses, %"),
        ("Goals: Non-Penalty", "Non-penalty goals per 90"),
        ("xG", "xG per 90"),
        ("Expected Assists", "xA per 90"),
        ("Offensive Duels", "Offensive duels per 90"),
        ("Offensive Duel Success %", "Offensive duels won, %"),
        ("Progressive Runs", "Progressive runs per 90"),
        ("Shots", "Shots per 90"),
        ("Touches in Opposition Box", "Touches in box per 90"),
    ]:
        ATTACKING.append((lab, float(np.nan_to_num(pct_of(met), nan=0.0)), val_of(met)[1]))

    DEFENSIVE = []
    for lab, met in [
        ("Aerial Duels", "Aerial duels per 90"),
        ("Aerial Duel Success %", "Aerial duels won, %"),
        ("Defensive Duels", "Defensive duels per 90"),
        ("Defensive Duel Success %", "Defensive duels won, %"),
        ("PAdj. Interceptions", "PAdj Interceptions"),
        ("Shots blocked", "Shots blocked per 90"),
    ]:
        DEFENSIVE.append((lab, float(np.nan_to_num(pct_of(met), nan=0.0)), val_of(met)[1]))

    POSSESSION = []
    for lab, met in [
        ("Deep Completions", "Deep completions per 90"),
        ("Dribbles", "Dribbles per 90"),
        ("Dribbling Success %", "Successful dribbles, %"),
        ("Forward Passes", "Forward passes per 90"),
        ("Forward Passing %", "Accurate forward passes, %"),
        ("Key passes", "Key passes per 90"),
        ("Long Passes", "Long passes per 90"),
        ("Long Passing %", "Accurate long passes, %"),
        ("Passes", "Passes per 90"),
        ("Passing %", "Accurate passes, %"),
        ("Passes to Final 3rd", "Passes to final third per 90"),
        ("Passes to Final 3rd %", "Accurate passes to final third, %"),
        ("Passes to Penalty Area", "Passes to penalty area per 90"),
        ("Pass to Penalty Area %", "Accurate passes to penalty area, %"),
        ("Progessive Passes", "Progressive passes per 90"),
        ("Progessive Passing %", "Accurate progressive passes, %"),
        ("Progressive Runs", "Progressive runs per 90"),
        ("Smart Passes", "Smart passes per 90"),
    ]:
        POSSESSION.append((lab, float(np.nan_to_num(pct_of(met), nan=0.0)), val_of(met)[1]))

    sections = [("Attacking",ATTACKING),("Defensive",DEFENSIVE),("Possession",POSSESSION)]
    sections = [(t,lst) for t,lst in sections if lst]

    # === styling ===
    PAGE_BG = "#ebebeb"; AX_BG = "#f3f3f3"; TRACK="#d6d6d6"
    TITLE_C="#111111"; LABEL_C="#222222"; DIVIDER="#000000"
    TAB_RED=np.array([199,54,60]); TAB_GOLD=np.array([240,197,106]); TAB_GREEN=np.array([61,166,91])
    def _blend(c1,c2,t): c=c1+(c2-c1)*np.clip(t,0,1); return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"
    def pct_to_rgb(v): v=float(np.clip(v,0,100)); return _blend(TAB_RED,TAB_GOLD,v/50) if v<=50 else _blend(TAB_GOLD,TAB_GREEN,(v-50)/50)

    # === layout (KEEP SAME) ===
    if not enable_images:
        fig_size   = (10, 8); dpi = 100
        title_row_h = 0.075
        header_block_h = title_row_h + 0.020
        img_box_w = img_box_h = 0.09; img_gap = 0.012
    else:
        fig_size   = (11.8, 9.6); dpi = 120
        title_row_h = 0.125
        header_block_h = title_row_h + 0.055
        img_box_w = img_box_h = 0.16

        preset_map = {
            "Tight (default)": {"img_gap": 0.0001, "s0": 0.02, "s1": 0.050},
            "Tight +":         {"img_gap": 0.0030, "s0": 0.02, "s1": 0.047},
            "Medium":          {"img_gap": 0.0060, "s0": 0.02, "s1": 0.044},
            "Wide":            {"img_gap": 0.0100, "s0": 0.02, "s1": 0.040},
        }
        _p = preset_map.get(spacing_preset, preset_map["Tight (default)"])
        img_gap = _p["img_gap"]
        _s0, _s1, _s2 = _p["s0"], _p["s1"], 2 * _p["s1"]

    GLOBAL_LEFT_PAD = 0.02
    BASE_LEFT, RIGHT = 0.035, 0.020
    LEFT = BASE_LEFT + GLOBAL_LEFT_PAD
    TITLE_LEFT_NUDGE = -0.001
    TOP, BOT = 0.035, 0.07
    header_h, GAP = 0.045, 0.020

    total_rows = sum(len(lst) for _, lst in sections)
    fig = plt.figure(figsize=fig_size, dpi=dpi); fig.patch.set_facecolor(PAGE_BG)

    rows_space_total = 1 - (TOP + BOT) - header_block_h - header_h*len(sections) - GAP*(len(sections)-1)
    row_slot = rows_space_total / max(total_rows,1)
    BAR_FRAC = 0.92
    gutter = 0.215
    ticks = np.arange(0,101,10)

    fig.text(LEFT + TITLE_LEFT_NUDGE, 1 - TOP - 0.010, f"{name_}\u2009|\u2009{team}",
             ha="left", va="top", color=TITLE_C, fontproperties=TITLE_FP)

    def draw_pairs_line(pairs_line, y):
        x = LEFT; renderer = fig.canvas.get_renderer()
        for i,(lab,val) in enumerate(pairs_line):
            t1 = fig.text(x, y, lab, ha="left", va="top", color=LABEL_C, fontproperties=INFO_LABEL_FP)
            fig.canvas.draw(); x += t1.get_window_extent(renderer).width / fig.bbox.width
            t2 = fig.text(x, y, str(val), ha="left", va="top", color=LABEL_C, fontproperties=INFO_VALUE_FP)
            fig.canvas.draw(); x += t2.get_window_extent(renderer).width / fig.bbox.width
            if i != len(pairs_line)-1:
                t3 = fig.text(x, y, "  |  ", ha="left", va="top", color="#555555", fontproperties=INFO_VALUE_FP)
                fig.canvas.draw(); x += t3.get_window_extent(renderer).width / fig.bbox.width

    if not enable_images:
        pairs = [("Position: ",pos), ("Age: ",age)]
        if show_height and height_text.strip(): pairs.append(("Height: ",height_text.strip()))
        pairs += [("Foot: ",foot_display), ("Games: ",games), ("Minutes: ",minutes), ("Goals: ",goals), ("Assists: ",assists)]
        draw_pairs_line(pairs, 1 - TOP - title_row_h + 0.010)
    else:
        row1 = [("Position: ",pos), ("Age: ",age), ("Height: ", (height_text.strip() if (show_height and height_text.strip()) else "—"))]
        row2 = [("Games: ",games), ("Goals: ",goals), ("Assists: ",assists)]
        row3 = [("Minutes: ",minutes), ("Foot: ",foot_display)]

        title_y = 1 - TOP - 0.010
        y1 = title_y - 0.055
        y2 = y1 - 0.039
        y3 = y2 - 0.039

        draw_pairs_line(row1, y1)
        draw_pairs_line(row2, y2)
        draw_pairs_line(row3, y3)

    # --- images (KEEP SAME, but add auto + hide player photo) ---
    def _open_upload(u):
        if u is None: return None
        try: return Image.open(u).convert("RGBA")
        except Exception: return None

    if enable_images:
        auto_pil_right = auto_pil_mid = auto_pil_left = None
        if auto_images:
            # rightmost = league logo
            auto_pil_right = _npimg_to_pil_rgba(_auto_league_logo(player_row))
            # middle    = team badge
            auto_pil_mid   = _npimg_to_pil_rgba(_auto_team_badge(player_row))
            # leftmost  = player photo (unless hidden)
            if not hide_player_photo:
                auto_pil_left  = _npimg_to_pil_rgba(_auto_player_photo(player_row))

        def add_header_image(pil_img, right_index=0):
            if pil_img is None: return
            x_right_edge = 1 - RIGHT
            x = x_right_edge - (right_index + 1) * img_box_w - right_index * img_gap
            per_image_shift = {
                0: _s0 + img1_dx,
                1: _s1 + img2_dx,
                2: _s2 + img3_dx
            }
            x += per_image_shift.get(right_index, 0.0)
            y_top_band = 1 - TOP - 0.006
            y = y_top_band - img_box_h
            ax_img = fig.add_axes([x, y, img_box_w, img_box_h])
            ax_img.imshow(pil_img); ax_img.axis("off")

        # Uploads override each slot:
        # rightmost slot (img1) = league logo
        # middle slot   (img2) = team badge
        # leftmost slot (img3) = player photo (unless hidden)
        pil_right = _open_upload(up_img1) or auto_pil_right
        pil_mid   = _open_upload(up_img2) or auto_pil_mid
        pil_left  = None if hide_player_photo else (_open_upload(up_img3) or auto_pil_left)

        add_header_image(pil_right, right_index=0)
        add_header_image(pil_mid,   right_index=1)
        add_header_image(pil_left,  right_index=2)

    fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT],
                                [1 - TOP - header_block_h + 0.004]*2,
                                transform=fig.transFigure, color=DIVIDER, lw=0.8, alpha=0.35))

    def draw_panel(panel_top, title, tuples, *, show_xticks=False, draw_bottom_divider=True):
        n = len(tuples); panel_h = header_h + n*row_slot
        fig.text(LEFT, panel_top - 0.012, title, ha="left", va="top", color=TITLE_C, fontproperties=H2_FP)

        ax = fig.add_axes([LEFT + gutter, panel_top - header_h - n*row_slot, 1 - LEFT - RIGHT - gutter, n*row_slot])
        ax.set_facecolor(AX_BG); ax.set_xlim(0,100); ax.set_ylim(-0.5,n-0.5)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False, length=0)
        ax.tick_params(axis="y", left=False,  labelleft=False,  length=0)
        ax.set_yticks([]); ax.get_yaxis().set_visible(False)

        for i in range(n):
            ax.add_patch(plt.Rectangle((0, i-(BAR_FRAC/2)), 100, BAR_FRAC, color=TRACK, ec="none", zorder=0.5))
        for gx in ticks:
            ax.vlines(gx, -0.5, n-0.5, colors=(0,0,0,0.16), linewidth=0.8, zorder=0.75)

        for i,(lab,pct,val_str) in enumerate(tuples[::-1]):
            y = i; bar_w = float(np.clip(pct,0,100))
            ax.add_patch(plt.Rectangle((0, y-(BAR_FRAC/2)), bar_w, BAR_FRAC, color=pct_to_rgb(bar_w), ec="none", zorder=1.0))
            x_text = 1.0 if bar_w >= 3 else min(100.0, bar_w + 0.8)
            ax.text(x_text, y, val_str, ha="left", va="center", color="#0B0B0B", fontproperties=BAR_VALUE_FP, zorder=2.0, clip_on=False)

        ax.axvline(50, color="#000000", ls=(0,(4,4)), lw=1.5, alpha=0.7, zorder=3.5)

        for i,(lab,_,_) in enumerate(tuples[::-1]):
            y_fig = (panel_top - header_h - n*row_slot) + ((i + 0.5) * row_slot)
            fig.text(LEFT, y_fig, lab, ha="left", va="center", color=LABEL_C, fontproperties=LABEL_FP)

        if show_xticks:
            trans = ax.get_xaxis_transform()
            offset_inner   = ScaledTranslation(7/72,0,fig.dpi_scale_trans)
            offset_pct_0   = ScaledTranslation(4/72,0,fig.dpi_scale_trans)
            offset_pct_100 = ScaledTranslation(10/72,0,fig.dpi_scale_trans)
            y_label = -0.075
            for gx in ticks:
                ax.plot([gx,gx],[-0.03,0.0], transform=trans, color=(0,0,0,0.6), lw=1.1, clip_on=False, zorder=4)
                ax.text(gx, y_label, f"{int(gx)}", transform=trans, ha="center", va="top", color="#000", fontproperties=TICK_FP, zorder=4, clip_on=False)
                if gx==0:   ax.text(gx, y_label, "%", transform=trans+offset_pct_0,   ha="left", va="top", color="#000", fontproperties=TICK_FP)
                elif gx==100: ax.text(gx, y_label, "%", transform=trans+offset_pct_100, ha="left", va="top", color="#000", fontproperties=TICK_FP)
                else:       ax.text(gx, y_label, "%", transform=trans+offset_inner,   ha="left", va="top", color="#000", fontproperties=TICK_FP)

        if draw_bottom_divider:
            y0 = panel_top - panel_h - 0.008
            fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT], [y0, y0], transform=fig.transFigure, color=DIVIDER, lw=1.2, alpha=0.35))
        return panel_top - panel_h - GAP

    y_top = 1 - TOP - header_block_h
    for idx,(title,data) in enumerate(sections):
        is_last = idx == len(sections)-1
        y_top = draw_panel(y_top, title, data, show_xticks=is_last, draw_bottom_divider=not is_last)

    fig.text((LEFT + gutter + (1 - RIGHT))/2.0, BOT * 0.1, footer_caption_text,
             ha="center", va="center", color=LABEL_C, fontproperties=FOOTER_FP)

    st.pyplot(fig, use_container_width=True)

    buf = BytesIO(); fig.savefig(buf, format="png", dpi=(150 if enable_images else 130),
                                 bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.download_button(
        "⬇️ Download Feature Z (PNG)",
        data=buf.getvalue(),
        file_name=f"{str(name_).replace(' ','_')}_featureZ.png",
        mime="image/png",
        key=f"download_feature_z_{uuid.uuid4().hex}"
    )
    plt.close(fig)
# ============================ END — Feature Z ============================





# ============================== FEATURE Q — CB ARCHETYPE SCATTER ==============================
from scipy.stats import rankdata
from io import BytesIO
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib import patheffects as pe

st.markdown("---")
st.header("🧭 Feature Q — FB Archetype Map")

# ------------------------------------------------------------------
# SETTINGS PANEL
# ------------------------------------------------------------------
with st.expander("Scatter settings", expanded=False):

    leagues_available_sc = sorted(df["League"].dropna().unique().tolist())
    player_league = player_row.iloc[0]["League"] if not player_row.empty else None
    # Selected player name for default labelling
    selected_player_name = player_row.iloc[0]["Player"] if not player_row.empty else None

    preset_sc = st.selectbox(
        "League preset",
        ["Player's league", "Top 5 Europe", "Top 20 Europe", "EFL (England 2–4)", "Custom"],
        index=0,
        key="fq_preset",
    )

    preset_map_sc = {
        "Player's league": {player_league} if player_league else set(),
        "Top 5 Europe": set(PRESET_LEAGUES.get("Top 5 Europe", [])),
        "Top 20 Europe": set(PRESET_LEAGUES.get("Top 20 Europe", [])),
        "EFL (England 2–4)": set(PRESET_LEAGUES.get("EFL (England 2–4)", [])),
        "Custom": set(),
    }

    add_leagues_sc = st.multiselect("Add leagues", leagues_available_sc, default=[], key="fq_add")
    leagues_scatter = sorted(preset_map_sc[preset_sc] | set(add_leagues_sc))
    if not leagues_scatter and player_league:
        leagues_scatter = [player_league]

    # Filters
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes_s, max_minutes_s = st.slider("Minutes", 0, 5000, (500, 5000), key="fq_min")
    min_age_s, max_age_s = st.slider("Age", 14, 45, (16, 40), key="fq_age")
    min_strength_s, max_strength_s = st.slider("League Strength", 0, 101, (0, 101), key="fq_ls")

    # Labels
    show_labels = st.toggle("Show labels", value=True, key="fq_lab")
    label_mode = st.selectbox(
        "Label mode",
        ["Selected player only", "All players", "U23 only", "U21 only", "U18 only"],
        index=0,
        key="fq_label_mode",
    )
    label_size = st.slider("Label size", 8, 20, 12, 1, key="fq_lblsize")

    # Points
    point_size = st.slider("Point size", 24, 300, 225, 2, key="fq_pts")
    point_alpha = st.slider("Point opacity", 0.2, 1.0, 0.92, 0.02, key="fq_alpha")

    # TEAM HIGHLIGHT – USED ONLY FOR LABEL FILTER
    teams_available_hl = sorted(df[df["League"].isin(leagues_scatter)]["Team"].dropna().unique())
    team_highlight = st.selectbox(
        "Highlight team (labels only shown for this team)",
        ["(None)"] + teams_available_hl,
        index=0,
        key="fq_team",
    )

    # Theme toggle (kept for future but background fixed dark)
    theme = st.radio("Theme", ["Dark", "Light"], index=0, horizontal=True, key="fq_theme")

    # === FIXED DARK BACKGROUND FOR PAGE & PLOT ===
    PAGE_BG = "#0a0f1c"
    PLOT_BG = "#0a0f1c"
    GRID_MAJ = "#3a4050"
    txt_col = "#f1f5f9"

    # Canvas
    canvas_preset = st.selectbox(
        "Canvas size",
        ["1280×720", "1600×900", "1920×820", "1920×1080"],
        index=1,
        key="fq_canvas",
    )
    w_px, h_px = map(int, canvas_preset.replace("×", "x").split("x"))

    top_gap_px = st.slider("Top gap (px)", 0, 240, 80, 5, key="fq_gap")
    render_exact = st.checkbox("Render exact pixels (PNG)", value=True, key="fq_exact")

# ------------------------------------------------------------------
# CB FILTER + SCORE CALCULATION
# ------------------------------------------------------------------
pool_sc = df[df["League"].isin(leagues_scatter)].copy()
pool_sc["Primary Position"] = pool_sc["Position"].astype(str).str.split(",").str[0].str.strip()
pool_sc = pool_sc[pool_sc["Primary Position"].isin(["LB", "LWB", "RB","RWB"])]

pool_sc["Minutes played"] = pd.to_numeric(pool_sc["Minutes played"], errors="coerce")
pool_sc["Age"] = pd.to_numeric(pool_sc["Age"], errors="coerce")
pool_sc["League Strength"] = pool_sc["League"].map(LEAGUE_STRENGTHS).fillna(0)

pool_sc = pool_sc[
    pool_sc["Minutes played"].between(min_minutes_s, max_minutes_s)
    & pool_sc["Age"].between(min_age_s, max_age_s)
    & pool_sc["League Strength"].between(min_strength_s, max_strength_s)
]

if pool_sc.empty:
    st.info("No CBs after filtering.")
    st.stop()

metric_groups = {
        'def_score': {
            'Defensive duels per 90': 0.4,
            'Defensive duels won, %': 0.3,
            'PAdj Interceptions': 0.2,
            'Shots blocked per 90': 0.1
        },
        'poss_score': {
            'Passes per 90': 0.1,
            'Crosses per 90': 0.1,
            'Forward passes per 90': 0.1,
            'Progressive passes per 90': 0.2,
            'xA per 90': 0.1,
            'Dribbles per 90': 0.1,
            'Progressive runs per 90': 0.2,
            'Passes to penalty area per 90': 0.1
        },
        'carry_score': {
            'Dribbles per 90': 0.4,
            'Successful dribbles, %': 0.1,
            'Progressive runs per 90': 0.3,
            'Accelerations per 90': 0.2
        },
        'boxing_score': {
            'xG per 90': 0.3,
            'Non-penalty goals per 90': 0.4,
            'Touches in box per 90': 0.3
        }
    }

def weighted_percentile(df_sub, row, mgrp):
    total = 0
    for m, w in mgrp.items():
        vals = df_sub[m].fillna(0)
        pct = rankdata(vals) / len(vals)
        total += pct[df_sub.index.get_loc(row.name)] * w
    return total * 100

for sn, grp in metric_groups.items():
    pool_sc[sn] = pool_sc.apply(lambda r: weighted_percentile(pool_sc, r, grp), axis=1)

def classify(r):
    if r["def_score"] >= 50 and r["poss_score"] >= 50:
        return "Two-Way"
    if r["def_score"] >= 50:
        return "Lockdown"
    if r["poss_score"] >= 50:
        return "Build-Up"
    return "Limited"

pool_sc["Archetype"] = pool_sc.apply(classify, axis=1)
pool_sc["Box-to-Box Ball Carrier"] = pool_sc["carry_score"] >= 70

# ------------------------------------------------------------------
# SCATTER GRAPH
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
fig.patch.set_facecolor(PAGE_BG)
ax.set_facecolor(PLOT_BG)

# Axes & labels
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("Possession Score", fontsize=16, fontweight="semibold", color=txt_col)
ax.xaxis.labelpad = 14
ax.set_ylabel("Defensive Score", fontsize=16, fontweight="semibold", color=txt_col)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(10))
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("semibold")
    tick.set_color(txt_col)
    tick.set_fontsize(14)

# Grid & spines
ax.grid(True, color=GRID_MAJ, linewidth=0.6)
for s in ax.spines.values():
    s.set_color("#e5e7eb")
    s.set_linewidth(1.1)

# Quadrant lines
line_col = "#FFFFFF"
ax.axvline(50, color=line_col, linestyle=(0, (4, 4)), lw=1.5)
ax.axhline(50, color=line_col, linestyle=(0, (4, 4)), lw=1.5)

# Quadrant labels
quad_fs = 16
bbox_style = dict(boxstyle="round,pad=0.35", facecolor="#d1d5db", edgecolor="none", alpha=0.9)
ax.text(6, 94, "LOCKDOWN", fontsize=quad_fs, weight="bold", bbox=bbox_style)
ax.text(94, 94, "COMPLETE", fontsize=quad_fs, weight="bold", ha="right", bbox=bbox_style)
ax.text(6, 6, "LIMITED", fontsize=quad_fs, weight="bold", bbox=bbox_style)
ax.text(94, 6, "BUILD UP/ATTACKING", fontsize=quad_fs, weight="bold", ha="right", bbox=bbox_style)

# Archetype colours
arch_colors = {
    "Build-Up": "#76B7B2",
    "Lockdown": "#F28E2B",
    "Two-Way": "#4E79A7",
    "Limited": "#E15759",
}

# Points
effective_point_size = point_size * 1.5
for (arch, carrier), grp in pool_sc.groupby(["Archetype", "Box-to-Box Ball Carrier"]):
    ax.scatter(
        grp["poss_score"],
        grp["def_score"],
        s=effective_point_size,
        c=arch_colors[arch],
        alpha=point_alpha,
        marker="s" if carrier else "o",
        edgecolors="none",
        linewidth=0,
        zorder=2,
    )

# ------------------------------------------------------------------
# LABEL HANDLING
# ------------------------------------------------------------------
highlight_grp = pool_sc[pool_sc["Team"] == team_highlight] if team_highlight != "(None)" else None
texts = []
if show_labels:
    if highlight_grp is not None and not highlight_grp.empty:
        label_df = highlight_grp
    else:
        if label_mode == "Selected player only" and selected_player_name:
            label_df = pool_sc[pool_sc["Player"] == selected_player_name]
        elif label_mode == "All players":
            label_df = pool_sc
        elif label_mode == "U23 only":
            label_df = pool_sc[pool_sc["Age"] < 23]
        elif label_mode == "U21 only":
            label_df = pool_sc[pool_sc["Age"] < 21]
        elif label_mode == "U18 only":
            label_df = pool_sc[pool_sc["Age"] < 18]
        else:
            label_df = pool_sc

    for _, r in label_df.iterrows():
        t = ax.annotate(
            r["Player"],
            (r["poss_score"], r["def_score"]),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=label_size + 2,
            color=txt_col,
            weight="semibold",
            ha="left",
            va="bottom",
            zorder=6,
        )
        t.set_path_effects([pe.withStroke(linewidth=2, foreground="#020617", alpha=0.9)])
        texts.append(t)

# ------------------------------------------------------------------
# SINGLE, PERFECTLY ALIGNED LEGEND BLOCK
# ------------------------------------------------------------------
# Compact spacing so the gap between Archetype and Ball Carrier is small
legend_kwargs = dict(
    loc="upper left",
    frameon=False,
    handlelength=1.1,
    handletextpad=0.4,
    borderpad=0.25,
    labelspacing=0.55,
    borderaxespad=0.0,
)

# Handles and labels in one legend:
#   4 archetypes
#   1 text row "Ball Carrier" (no marker)
#   2 marker rows: Yes (square), No (circle)
handles = [
    # Archetypes
    Line2D([0], [0], marker="s", linestyle="None", color="none",
           markerfacecolor=arch_colors["Build-Up"], markersize=16, label="Build-Up"),
    Line2D([0], [0], marker="s", linestyle="None", color="none",
           markerfacecolor=arch_colors["Lockdown"], markersize=16, label="Lockdown"),
    Line2D([0], [0], marker="s", linestyle="None", color="none",
           markerfacecolor=arch_colors["Two-Way"], markersize=16, label="Two-Way"),
    Line2D([0], [0], marker="s", linestyle="None", color="none",
           markerfacecolor=arch_colors["Limited"], markersize=16, label="Limited"),
    # Ball Carrier header row (no marker)
    Line2D([], [], linestyle="None", color="none", label="Ball Carrier"),
    # No / Yes markers
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        color="none",
        markeredgecolor=txt_col,
        markerfacecolor="#f1f5f9",
        markeredgewidth=1.4,
        markersize=16,
        label="No",
    ),
    Line2D(
        [0], [0],
        marker="s",
        linestyle="None",
        color="none",
        markeredgecolor=txt_col,
        markerfacecolor="#f1f5f9",
        markeredgewidth=1.4,
        markersize=16,
        label="Yes",
    ),
]

labels = [
    "Build Up",
    "Lockdown",
    "Complete",
    "Limited",
    "Ball Carrier",  # header row, no marker
    "No",
    "Yes",
]

legend = ax.legend(
    handles=handles,
    labels=labels,
    title="Archetype",
    title_fontsize=15,
    fontsize=14,
    bbox_to_anchor=(1.01, 1.00),   # X for the whole block
    **legend_kwargs,
)
legend.get_title().set_color(txt_col)
legend.get_title().set_fontweight("semibold")

# style all label text
for i, txt in enumerate(legend.get_texts()):
    txt.set_color(txt_col)
    txt.set_fontweight("semibold")
    # make the "Ball Carrier" header row stand out slightly
    if labels[i] == "Ball Carrier":
        txt.set_fontstyle("italic")

# ------------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------------
fig.subplots_adjust(
    left=0.06,
    right=0.865,
    bottom=0.11,
    top=1.02 - top_gap_px / float(h_px),
)

# ------------------------------------------------------------------
# RENDER
# ------------------------------------------------------------------
if render_exact:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=PAGE_BG)
    buf.seek(0)
    st.image(buf, width=w_px)
    st.download_button(
        "⬇️ Download Feature Q (PNG)",
        data=buf.getvalue(),
        file_name=f"feature_q_cb_{uuid.uuid4().hex[:6]}.png",
        mime="image/png",
    )
else:
    st.pyplot(fig)

plt.close(fig)
# ============================== END FEATURE Q ============================================================






# ----------------- (B) COMPARISON RADAR — decile tick values (1dp) + light/dark theme + exact edge + centered/upright outside labels -----------------
import re

st.markdown("---")
st.header("📊 Player Comparison Radar")

DEFAULT_RADAR_METRICS = [
    "Defensive duels per 90","Defensive duels won, %","PAdj Interceptions","Aerial duels won, %",
    "Passes per 90","Accurate passes, %","Progressive passes per 90", "Passes to penalty area per 90",
    "Progressive runs per 90","Dribbles per 90",
    "xA per 90",
]

def _clean_radar_label(s: str) -> str:
    s = s.replace("Aerial duels won, %", "Aerial %")
    s = s.replace("xA per 90", "xA")
    s = s.replace("Dribbles per 90", "Dribbles")
    s = s.replace("Defensive duels won, %", "Defensive Duel %")
    s = s.replace("Defensive duels per 90", "Defensive Duels").replace("Passes per 90", "Passes")
    s = s.replace("Progressive runs per 90", "Progressive Runs").replace("Progressive passes per 90", "Progressive Passes")
    s = s.replace("Passes to penalty area per 90", "Passes to Pen Area").replace("Accurate passes, %", "Passing %")
    return re.sub(r"\s*per\s*90", "", s, flags=re.I)

# Theme selector
with st.expander("Radar settings", expanded=False):
    radar_theme = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True, key="radar_theme")

# Colors per theme
if radar_theme == "Dark":
    PAGE_BG = "#0a0f1c"
    AX_BG   = "#0a0f1c"
    GRID_BAND_OUTER = "#162235"
    GRID_BAND_INNER = "#0d1524"
    RING_COLOR_INNER = "#3a4050"
    RING_COLOR_OUTER = "#cbd5e1"
    LABEL_COLOR = "#f5f5f5"
    TICK_COLOR  = "#e5e7eb"
    MINUTES_CLR = "#f5f5f5"
else:
    PAGE_BG = "#ffffff"
    AX_BG   = "#ebebeb"
    GRID_BAND_OUTER = "#e5e7eb"
    GRID_BAND_INNER = "#ffffff"
    RING_COLOR_INNER = RING_COLOR_OUTER = "#d1d5db"
    LABEL_COLOR = "#0f172a"
    TICK_COLOR  = "#6b7280"
    MINUTES_CLR = "#374151"

if player_row.empty:
    st.info("Pick a player above to draw the radar.")
else:
    # Player A is the selected player
    pA = player_name
    rowA_all = df[df["Player"] == pA]
    if rowA_all.empty:
        st.info("Selected player not found in dataset.")
    else:
        rowA = rowA_all.iloc[0]

        # Player B options using the universal position_filter
        pool_pos = df[df["Position"].astype(str).apply(position_filter)].copy()
        players_b = sorted(pool_pos["Player"].dropna().unique().tolist())
        players_b = [p for p in players_b if p != pA]

        if not players_b:
            st.info("No comparison players available for the current universal position filter.")
        else:
            pB = st.selectbox("Player B (blue)", players_b, index=0, key="radar_pb")

            rowB_all = df[df["Player"] == pB]
            if rowB_all.empty:
                st.info("Comparison player not found in dataset.")
            else:
                rowB = rowB_all.iloc[0]

                # Numeric radar metrics
                numeric_cols = set(df.select_dtypes(include="number").columns.tolist())
                radar_metrics = [m for m in DEFAULT_RADAR_METRICS if m in df.columns and m in numeric_cols]
                if not radar_metrics:
                    st.info("No numeric radar metrics available in dataset.")
                else:
                    # Pool = A∪B leagues, same universal position filter
                    union_leagues = {rowA["League"], rowB["League"]}
                    pool = df[
                        (df["League"].isin(union_leagues)) &
                        (df["Position"].astype(str).apply(position_filter))
                    ].copy()

                    for m in radar_metrics:
                        pool[m] = pd.to_numeric(pool[m], errors="coerce")
                    pool = pool.dropna(subset=radar_metrics + ["Player"])

                    if pool.empty:
                        st.info("No players in the combined A∪B league pool after applying the universal position filter.")
                    else:
                        # Percentiles for A & B vs pool (0–100 scale)
                        pool_pct = pool[radar_metrics].rank(pct=True) * 100.0

                        def pct_for(name: str) -> np.ndarray:
                            idx = pool[pool["Player"] == name].index
                            if len(idx) == 0:
                                return np.full(len(radar_metrics), np.nan)
                            return pool_pct.loc[idx, :].mean(axis=0).values

                        A_r = pct_for(pA)
                        B_r = pct_for(pB)

                        # Labels
                        labels = [_clean_radar_label(m) for m in radar_metrics]

                        # TRUE deciles (0..100) for each metric — displayed at 1dp
                        qs = np.linspace(0, 100, 11)
                        axis_ticks = [np.nanpercentile(pool[m].values, qs) for m in radar_metrics]

                        # ---- draw radar ----
                        COL_A = "#C81E1E"; COL_B = "#1D4ED8"
                        FILL_A = (200/255, 30/255, 30/255, 0.60)
                        FILL_B = (29/255, 78/255, 216/255, 0.60)
                        RING_LW = 1.0
                        TITLE_FS = 26; SUB_FS = 12; AXIS_FS = 10
                        TICK_FS = 7; INNER_HOLE = 10

                        from matplotlib.patches import Wedge, Circle
                        import matplotlib.pyplot as plt
                        import numpy as np
                        import pandas as pd

                        def _tangent_rotation(ax, theta):
                            """Tangential rotation in display space, respecting theta offset/direction."""
                            return np.degrees(ax.get_theta_direction() * theta + ax.get_theta_offset()) - 90.0

                        def draw_radar(labels, A_r, B_r, ticks, headerA, subA, headerB, subB):
                            N = len(labels)
                            theta = np.linspace(0, 2*np.pi, N, endpoint=False)
                            theta_c = np.concatenate([theta, theta[:1]])
                            Ar = np.concatenate([A_r, A_r[:1]])
                            Br = np.concatenate([B_r, B_r[:1]])

                            fig = plt.figure(figsize=(13.2, 8.0), dpi=260)
                            fig.patch.set_facecolor(PAGE_BG)
                            ax = plt.subplot(111, polar=True); ax.set_facecolor(AX_BG)

                            # Orientation like your original
                            ax.set_theta_offset(np.pi/2)
                            ax.set_theta_direction(-1)

                            ax.set_xticks(theta)
                            ax.set_xticklabels([])  # custom labels below
                            ax.set_yticks([])
                            ax.grid(False)
                            [s.set_visible(False) for s in ax.spines.values()]

                            # radial bands (10 bands from INNER_HOLE to 100)
                            ring_edges = np.linspace(INNER_HOLE, 100, 11)
                            for i in range(10):
                                r0, r1 = ring_edges[i], ring_edges[i+1]
                                band = GRID_BAND_OUTER if ((9 - i) % 2 == 0) else GRID_BAND_INNER
                                ax.add_artist(Wedge(
                                    (0,0), r1, 0, 360, width=(r1-r0),
                                    transform=ax.transData._b, facecolor=band,
                                    edgecolor="none", zorder=0.8
                                ))

                            # ring outlines — ONLY the outermost ring brighter in dark theme
                            ring_t = np.linspace(0, 2*np.pi, 361)
                            for j, r in enumerate(ring_edges):
                                col = RING_COLOR_OUTER if j == len(ring_edges)-1 else RING_COLOR_INNER
                                ax.plot(ring_t, np.full_like(ring_t, r), color=col, lw=RING_LW, zorder=0.9)

                            # numeric tick labels at each ring = TRUE dataset quantiles (rounded to 1dp)
                            start_idx = 2  # show from 20th to reduce clutter
                            for i, ang in enumerate(theta):
                                vals = ticks[i]
                                for rr, v in zip(ring_edges[start_idx:], vals[start_idx:]):
                                    ax.text(ang, rr-1.8, f"{float(v):.1f}",
                                            ha="center", va="center",
                                            fontsize=TICK_FS, color=TICK_COLOR, zorder=1.1)

                            # --- Outside metric labels: centered, flipped only if upside-down, pushed further out ---
                            OUTER_LABEL_R = 105.6  # distance from outer ring; try 105.0–107.0
                            for ang, lab in zip(theta, labels):
                                rot = _tangent_rotation(ax, ang)  # tangential angle in display space
                                # Keep text upright: flip if rotation would be upside-down
                                rot_norm = ((rot + 180.0) % 360.0) - 180.0
                                if rot_norm > 90 or rot_norm < -90:
                                    rot += 180.0
                                ax.text(
                                    ang, OUTER_LABEL_R, lab,
                                    rotation=rot, rotation_mode="anchor",
                                    ha="center", va="center",
                                    fontsize=AXIS_FS, color=LABEL_COLOR, fontweight=600,
                                    clip_on=False, zorder=2.2
                                )

                            # center hole
                            ax.add_artist(Circle((0,0), radius=INNER_HOLE-0.6, transform=ax.transData._b,
                                                 color=PAGE_BG, zorder=1.2, ec="none"))

                            # A & B polygons (percentile radii)
                            ax.plot(theta_c, Ar, color=COL_A, lw=2.2, zorder=3)
                            ax.fill(theta_c, Ar, color=FILL_A, zorder=2.5)
                            ax.plot(theta_c, Br, color=COL_B, lw=2.2, zorder=3)
                            ax.fill(theta_c, Br, color=FILL_B, zorder=2.5)

                            # keep edge exactly at 100; labels allowed outside via clip_on=False
                            ax.set_rlim(0, 100)

                            # headers (teams / leagues / minutes)
                            minsA = f"{int(pd.to_numeric(rowA.get('Minutes played',0))):,} mins" if pd.notna(rowA.get('Minutes played')) else "Minutes: N/A"
                            minsB = f"{int(pd.to_numeric(rowB.get('Minutes played',0))):,} mins" if pd.notna(rowB.get('Minutes played')) else "Minutes: N/A"

                            fig.text(0.12, 0.96,  headerA, color=COL_A, fontsize=TITLE_FS, fontweight="bold", ha="left")
                            fig.text(0.12, 0.935, subA, color=COL_A, fontsize=SUB_FS, ha="left")
                            fig.text(0.12, 0.915, minsA, color=MINUTES_CLR, fontsize=10, ha="left")

                            fig.text(0.88, 0.96,  headerB, color=COL_B, fontsize=TITLE_FS, fontweight="bold", ha="right")
                            fig.text(0.88, 0.935, subB, color=COL_B, fontsize=SUB_FS, ha="right")
                            fig.text(0.88, 0.915, minsB, color=MINUTES_CLR, fontsize=10, ha="right")

                            return fig

                        fig_r = draw_radar(
                            labels, A_r, B_r, axis_ticks,
                            headerA=pA, subA=f"{rowA['Team']} — {rowA['League']}",
                            headerB=pB, subB=f"{rowB['Team']} — {rowB['League']}",
                        )
                        st.caption(
                            "Ring labels show the **actual dataset values** at each decile (0–100th), rounded to **1 decimal place**. "
                            "Axis labels are centered on their metric angle, auto-flipped upright, and placed outside the 100 ring."
                        )
                        st.pyplot(fig_r, use_container_width=True)
# ----------------- END Radar -----------------








# ----------------- (C) SIMILAR PLAYERS (adjustable pool — FIXED PRESET UI) -----------------
st.markdown("---")
st.header("🧭 Similar players (within adjustable pool)")

# --- Feature basket declared FIRST so UI can use it ---
SIM_FEATURES = [
    'Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
    'Aerial duels won, %','Shots blocked per 90','PAdj Interceptions','Non-penalty goals per 90','xG per 90',
    'Shots per 90','Dribbles per 90','Successful dribbles, %','Offensive duels per 90','Offensive duels won, %',
    'Touches in box per 90','Progressive runs per 90','Accelerations per 90','Passes per 90','Accurate passes, %',
    'Forward passes per 90','Accurate forward passes, %','Long passes per 90','Accurate long passes, %','xA per 90',
    'Smart passes per 90','Key passes per 90','Passes to final third per 90','Accurate passes to final third, %',
    'Passes to penalty area per 90','Accurate passes to penalty area, %','Deep completions per 90',
    'Progressive passes per 90',
]

# league strength map (supports either variable name)
LS_MAP = globals().get('LEAGUE_STRENGTHS', globals().get('league_strengths', {}))

# defaults for advanced weights (others default to 1)
DEFAULT_SIM_WEIGHTS = {f: 1 for f in SIM_FEATURES}
DEFAULT_SIM_WEIGHTS.update({
    'Passes per 90': 3, 'Passes to penalty area per 90': 2, 'Dribbles per 90': 2, 'xA per 90': 2,
    'Progressive passes per 90': 3, 'Defensive duels per 90': 2, 'Forward passes per 90': 3,
    'PAdj Interceptions': 2, 'Aerial duels won, %': 2, 'Touches in box per 90': 2,
})

# --- Build local presets safely (no reliance on _PRESETS_CF existing) ---
_leagues_from_df = df['League'].dropna().unique().tolist() if 'League' in df.columns else []
_included_from_global = list(globals().get('INCLUDED_LEAGUES', []))
_included_leagues_cf = sorted(set(_included_from_global) | set(_leagues_from_df))

_PRESET_LEAGUES_SAFE = globals().get('PRESET_LEAGUES', {})  # may be missing; that's ok
_PRESETS_SIM = {
    "All listed leagues": _included_leagues_cf,
    "T5":  sorted(list(_PRESET_LEAGUES_SAFE.get("Top 5 Europe", []))),
    "T20": sorted(list(_PRESET_LEAGUES_SAFE.get("Top 20 Europe", []))),
    "EFL": sorted(list(_PRESET_LEAGUES_SAFE.get("EFL (England 2–4)", []))),
    "Custom": None,
}

# ====================== UI (fixed preset behavior; multiselect always editable) ======================
with st.expander("Similarity settings", expanded=False):
    # options
    candidate_league_options = sorted(_included_leagues_cf or _leagues_from_df)
    default_sel = leagues_sel if 'leagues_sel' in globals() else candidate_league_options

    sim_preset_choices = list(_PRESETS_SIM.keys())
    sim_preset = st.selectbox(
        "Candidate league preset",
        sim_preset_choices,
        index=sim_preset_choices.index("All listed leagues"),
        key="sim_preset"
    )

    # compute preset values; keep only leagues that exist in options
    preset_vals_raw = _PRESETS_SIM.get(sim_preset) or []
    preset_vals = sorted([lg for lg in preset_vals_raw if lg in candidate_league_options])

    # if preset changed, seed the selection once
    _last_key = "_last_sim_preset"
    if st.session_state.get(_last_key) != sim_preset:
        st.session_state["sim_leagues"] = preset_vals if preset_vals else default_sel
        st.session_state[_last_key] = sim_preset

    # ALWAYS editable multiselect (no disabled=…)
    sim_leagues = st.multiselect(
        "Candidate leagues",
        candidate_league_options,
        default=st.session_state.get("sim_leagues", preset_vals if preset_vals else default_sel),
        key="sim_leagues",
    )

    if preset_vals_raw and not preset_vals:
        st.warning("Preset has leagues, but none match your allowed list/dataset.")
    elif preset_vals_raw:
        st.caption(f"Preset: {sim_preset} — {len(preset_vals)} league(s). You can add/prune below.")

    # Base filters
    sim_min_minutes, sim_max_minutes = st.slider("Minutes played (candidates)", 0, 5000, (500, 5000), key="sim_min")
    sim_min_age, sim_max_age = st.slider("Age (candidates)", 14, 45, (16, 40), key="sim_age")

    # Optional league quality filter (0–101)
    use_strength_filter = st.toggle("Filter by league quality (0–101)", value=False, key="sim_use_strength")
    if use_strength_filter:
        sim_min_strength, sim_max_strength = st.slider("League quality (strength)", 0, 101, (0, 101), key="sim_strength")

    # Blending
    percentile_weight = st.slider("Percentile weight", 0.0, 1.0, 0.7, 0.05, key="sim_pw")

    # League difficulty adjustment
    apply_league_adjust = st.toggle("Apply league difficulty adjustment", value=True, key="sim_apply_ladj")
    league_weight_sim = st.slider(
        "League weight (difficulty adj.)", 0.0, 1.0, 0.2, 0.05, key="sim_lw",
        disabled=not apply_league_adjust
    )

    # Advanced weights
    with st.expander("Advanced feature weights (1–5)", expanded=False):
        adv_weights = {}
        for f in SIM_FEATURES:
            key = "simw_" + f.replace(" ", "_").replace("%", "pct").replace(",", "").replace(".", "_")
            adv_weights[f] = st.slider(f"Weight — {f}", 1, 5, int(st.session_state.get(key, DEFAULT_SIM_WEIGHTS.get(f, 1))), key=key)

    top_n_sim = st.number_input("Show top N", min_value=5, max_value=200, value=50, step=5, key="sim_top")

# ====================== Similarity computation ======================
if not player_row.empty:
    from sklearn.preprocessing import StandardScaler
    target_row_full = df[df['Player'] == player_name].head(1).iloc[0]
    target_league = target_row_full['League']

    df_candidates = df[df['League'].isin(sim_leagues)].copy()

    # optional league quality filter
    if use_strength_filter and LS_MAP:
        df_candidates['League strength'] = df_candidates['League'].map(LS_MAP).fillna(0.0)
        df_candidates = df_candidates[
            (df_candidates['League strength'] >= float(sim_min_strength)) &
            (df_candidates['League strength'] <= float(sim_max_strength))
        ]

    # position filter (reuse your global position_filter)
    if 'Position' in df_candidates.columns:
        df_candidates = df_candidates[df_candidates['Position'].astype(str).apply(position_filter)]
    else:
        st.warning("No 'Position' column found; cannot apply position filter.")

    # base filters
    df_candidates['Minutes played'] = pd.to_numeric(df_candidates['Minutes played'], errors='coerce')
    df_candidates['Age'] = pd.to_numeric(df_candidates['Age'], errors='coerce')
    df_candidates = df_candidates[
        df_candidates['Minutes played'].between(sim_min_minutes, sim_max_minutes) &
        df_candidates['Age'].between(sim_min_age, sim_max_age)
    ]

    # one row per player (keep most minutes, then stronger league)
    df_candidates['League strength'] = df_candidates['League'].map(LS_MAP).fillna(0.0) if LS_MAP else 0.0
    df_candidates = (
        df_candidates.sort_values(['Player','Minutes played','League strength'], ascending=[True, False, False])
                   .drop_duplicates(subset=['Player'], keep='first')
    )
    df_candidates = df_candidates[df_candidates['Player'] != player_name]

    # ensure features numeric
    df_candidates = df_candidates.dropna(subset=SIM_FEATURES)
    for f in SIM_FEATURES:
        df_candidates[f] = pd.to_numeric(df_candidates[f], errors='coerce')
    df_candidates = df_candidates.dropna(subset=SIM_FEATURES)

    # target percentiles vs target league
    league_mask = (df['League'] == target_league)
    league_block = df.loc[league_mask, SIM_FEATURES].apply(pd.to_numeric, errors='coerce')
    league_ranks = league_block.rank(pct=True)
    target_mask_in_league = league_mask & (df['Player'] == player_name)
    if not target_mask_in_league.any():
        st.info("Target player not found in league block for percentile calc.")
        target_percentiles_vec = np.full(len(SIM_FEATURES), 0.5)
    else:
        target_percentiles_vec = league_ranks.loc[target_mask_in_league].iloc[0].values

    if not df_candidates.empty:
        # percentile ranks for candidates (per-league)
        percl = df_candidates.groupby('League')[SIM_FEATURES].rank(pct=True).values

        # standardize on candidate pool (actual values)
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(df_candidates[SIM_FEATURES])
        target_features_standardized = scaler.transform([target_row_full[SIM_FEATURES].astype(float).values])

        # weights
        weights_vec = np.array([float(adv_weights.get(f, 1)) for f in SIM_FEATURES], dtype=float)

        # distances + blend
        percentile_distances = np.linalg.norm((percl - target_percentiles_vec) * weights_vec, axis=1)
        actual_value_distances = np.linalg.norm((standardized_features - target_features_standardized) * weights_vec, axis=1)
        combined = percentile_distances * percentile_weight + actual_value_distances * (1.0 - percentile_weight)

        # normalize -> similarity 0..100
        arr = np.asarray(combined, dtype=float).ravel()
        rng = np.ptp(arr)
        norm = (arr - arr.min()) / (rng if rng != 0 else 1.0)
        similarities = ((1.0 - norm) * 100.0).round(2)

        out = df_candidates[['Player','Team','League','Position','Age','Minutes played','Market value']].copy()
        out['League strength'] = out['League'].map(LS_MAP).fillna(0.0) if LS_MAP else 0.0
        tgt_ls = float(LS_MAP.get(target_league, 1.0)) if LS_MAP else 1.0

        # symmetric league ratio (≤1)
        eps = 1e-6
        cand_ls = np.maximum(out['League strength'].astype(float), eps)
        tgt_ls_safe = max(tgt_ls, eps)
        league_ratio = np.minimum(cand_ls / tgt_ls_safe, tgt_ls_safe / cand_ls)

        out['Similarity'] = similarities
        out['Adjusted Similarity'] = (
            out['Similarity'] * ((1 - league_weight_sim) + league_weight_sim * league_ratio)
        ) if apply_league_adjust else out['Similarity']

        out = out.sort_values('Adjusted Similarity', ascending=False).reset_index(drop=True)
        out.insert(0, 'Rank', np.arange(1, len(out) + 1))
        st.caption(f"Candidates after filters: {len(out):,}")
        st.dataframe(out.head(int(top_n_sim)), use_container_width=True)
    else:
        st.info("No candidates after similarity filters.")
else:
    st.caption("Pick a player to see similar players.")



# ---------------------------- (D) CLUB FIT — FIXED & SYNCED TO SELECTED PLAYER ----------------------------
st.markdown("---")
st.header("🏟️ Club Fit Finder")

# ---------- SAFE FALLBACKS ----------
if 'INCLUDED_LEAGUES' in globals():
    _included_leagues_cf = list(INCLUDED_LEAGUES)
else:
    _included_leagues_cf = sorted(pd.Series(df.get('League', pd.Series([]))).dropna().unique().tolist())

if 'PRESET_LEAGUES' in globals():
    _PRESETS_CF = {
        "All listed leagues": _included_leagues_cf,
        "Top 5 Europe": sorted(list(PRESET_LEAGUES.get("Top 5 Europe", []))),
        "Top 20 Europe": sorted(list(PRESET_LEAGUES.get("Top 20 Europe", []))),
        "EFL (England 2–4)": sorted(list(PRESET_LEAGUES.get("EFL (England 2–4)", []))),
        "Custom": None,
    }
else:
    _PRESETS_CF = {
        "All listed leagues": _included_leagues_cf,
        "Top 5 Europe": [], "Top 20 Europe": [], "EFL (England 2–4)": [], "Custom": None,
    }

_DEFAULT_W_CF = {
    'Passes per 90': 3,'Passes to penalty area per 90': 2,'Dribbles per 90': 2,'xA per 90': 2,
    'Progressive passes per 90': 3,'Defensive duels per 90': 2,'Forward passes per 90': 3,
    'PAdj Interceptions': 2,'Aerial duels won, %': 2,'Touches in box per 90': 2,
}

_LS_CF = dict(LEAGUE_STRENGTHS) if 'LEAGUE_STRENGTHS' in globals() else {lg: 50.0 for lg in _included_leagues_cf}
DEFAULT_LEAGUE_WEIGHT = 0.5
DEFAULT_MARKET_WEIGHT = 0.2

CF_FEATURES = [
    'Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
    'Aerial duels won, %','Shots blocked per 90','PAdj Interceptions','Non-penalty goals per 90','xG per 90',
    'Shots per 90','Dribbles per 90','Successful dribbles, %','Offensive duels per 90','Offensive duels won, %',
    'Touches in box per 90','Progressive runs per 90','Accelerations per 90','Passes per 90','Accurate passes, %',
    'Forward passes per 90','Accurate forward passes, %','Long passes per 90','Accurate long passes, %','xA per 90',
    'Smart passes per 90','Key passes per 90','Passes to final third per 90','Accurate passes to final third, %',
    'Passes to penalty area per 90','Accurate passes to penalty area, %','Deep completions per 90',
    'Progressive passes per 90',
]

required_cols_cf = {'Player','Team','League','Age','Position','Minutes played','Market value', *CF_FEATURES}
missing_cf = [c for c in required_cols_cf if c not in df.columns]
if missing_cf:
    st.error(f"Club Fit: dataset missing required columns: {missing_cf}")
else:
    # -------------------- Controls --------------------
    with st.expander("Club-fit settings", expanded=False):
        leagues_available_cf = sorted(set(_included_leagues_cf) | set(df.get('League', pd.Series([])).dropna().unique()))

        target_leagues_cf = st.multiselect(
            "Target leagues (choose target from here)",
            leagues_available_cf,
            default=leagues_available_cf,
            key="cf_target_leagues"
        )

        if 'candidate_leagues_cf' not in st.session_state:
            st.session_state.candidate_leagues_cf = list(_included_leagues_cf)

        preset_name_cf = st.selectbox("Candidate pool preset", list(_PRESETS_CF.keys()), index=0, key="cf_preset_name")
        c1a, c1b = st.columns([1,2])
        if c1a.button("Apply preset", key="cf_apply_preset"):
            if _PRESETS_CF.get(preset_name_cf) is not None:
                st.session_state.candidate_leagues_cf = list(_PRESETS_CF[preset_name_cf])

        extra_candidate_leagues_cf = c1b.multiselect(
            "Extra leagues to add", leagues_available_cf, default=[], key="cf_extra_leagues"
        )
        leagues_selected_cf = sorted(set(st.session_state.candidate_leagues_cf) | set(extra_candidate_leagues_cf))
        st.caption(f"Candidate pool leagues: **{len(leagues_selected_cf)}** selected.")

        # Target pool: universal position_filter (not exact position string)
        target_pool_cf = df[df['League'].isin(target_leagues_cf)].copy()
        target_pool_cf = target_pool_cf[target_pool_cf['Position'].astype(str).apply(position_filter)]
        target_options_cf = sorted(target_pool_cf['Player'].dropna().unique().tolist())

        # -------- SYNC THE SELECTED PLAYER INTO THIS WIDGET --------
        # Keep a canonical "selected_player" around
        st.session_state["selected_player"] = player_name
        sp = st.session_state["selected_player"]

        # Make sure the selected player is present in options (even if filtered out)
        if sp and sp not in target_options_cf and sp in df['Player'].values:
            target_options_cf = [sp] + target_options_cf
            seen = set(); target_options_cf = [x for x in target_options_cf if not (x in seen or seen.add(x))]

        # If widget holds a stale value or a different profile is selected, force it to the new one
        if (
            "cf_target_player" not in st.session_state
            or st.session_state["cf_target_player"] not in target_options_cf
            or st.session_state.get("cf_bound_to") != sp
        ):
            st.session_state["cf_target_player"] = sp if sp in target_options_cf else (target_options_cf[0] if target_options_cf else None)
            st.session_state["cf_bound_to"] = sp  # remember which profile we synced from

        # Now render the selectbox (it will show the synced value)
        target_player_cf = st.selectbox(
            "Target player",
            target_options_cf,
            index=target_options_cf.index(st.session_state["cf_target_player"]) if target_options_cf and st.session_state["cf_target_player"] in target_options_cf else 0,
            key="cf_target_player"
        )

        # Filters
        df["Minutes played"] = pd.to_numeric(df.get("Minutes played"), errors="coerce")
        df["Age"] = pd.to_numeric(df.get("Age"), errors="coerce")
        max_minutes_in_data_cf = int(df["Minutes played"].fillna(0).max())
        slider_max_minutes_cf = int(max(1000, max_minutes_in_data_cf))

        min_minutes_cf, max_minutes_cf = st.slider(
            "Minutes filter (candidates)", 0, slider_max_minutes_cf,
            (500, slider_max_minutes_cf), key="cf_minutes_slider"
        )

        age_series_cf = df["Age"]
        age_min_data_cf = int(np.nanmin(age_series_cf)) if age_series_cf.notna().any() else 14
        age_max_data_cf = int(np.nanmax(age_series_cf)) if age_series_cf.notna().any() else 45
        min_age_cf, max_age_cf = st.slider(
            "Age filter (candidates)", age_min_data_cf, age_max_data_cf, (16, 40), key="cf_age_slider"
        )

        min_strength_cf, max_strength_cf = st.slider("League quality (strength)", 0, 101, (0, 101), key="cf_strength")

        league_weight_cf = st.slider("League weight", 0.0, 1.0, DEFAULT_LEAGUE_WEIGHT, 0.05, key="cf_league_w")
        market_value_weight_cf = st.slider("Market value weight", 0.0, 1.0, DEFAULT_MARKET_WEIGHT, 0.05, key="cf_market_w")
        manual_override_cf = st.number_input("Target market value override (€)", min_value=0, value=0, step=100_000, key="cf_mv_override")

        st.subheader("Advanced feature weights")
        st.caption("Unlisted features default to weight = 1.")
        weights_ui_cf = {f: st.slider(f"• {f}", 0, 5, int(_DEFAULT_W_CF.get(f, 1)), key=f"cf_w_{f}") for f in CF_FEATURES}

        top_n_cf = st.number_input("Show top N teams", 5, 100, 20, 5, key="cf_topn")

    # -------------------- Compute --------------------
    target_player_val = st.session_state.get("cf_target_player")
    if target_player_val and (target_player_val in df['Player'].values):
        # Candidate player pool (universal position_filter)
        df_candidates_cf = df[df['League'].isin(leagues_selected_cf)].copy()
        df_candidates_cf = df_candidates_cf[df_candidates_cf['Position'].astype(str).apply(position_filter)]

        # Numerics + filters
        df_candidates_cf['Minutes played'] = pd.to_numeric(df_candidates_cf['Minutes played'], errors='coerce')
        df_candidates_cf['Age'] = pd.to_numeric(df_candidates_cf['Age'], errors='coerce')
        df_candidates_cf['Market value'] = pd.to_numeric(df_candidates_cf['Market value'], errors='coerce')

        df_candidates_cf = df_candidates_cf[
            df_candidates_cf['Minutes played'].between(min_minutes_cf, max_minutes_cf, inclusive='both')
        ]
        df_candidates_cf = df_candidates_cf[
            df_candidates_cf['Age'].between(min_age_cf, max_age_cf, inclusive='both')
        ]
        df_candidates_cf = df_candidates_cf.dropna(subset=CF_FEATURES)

        if df_candidates_cf.empty:
            st.info("No candidate players after filters. Widen candidate leagues or relax filters.")
        else:
            # Target row from full df (never disappears)
            target_all_rows = df[df['Player'] == target_player_val].copy()
            if target_all_rows.empty:
                st.info("Target player not found in dataset.")
            else:
                target_row_cf = target_all_rows.sort_values('Minutes played', ascending=False).iloc[0]
                target_vector_cf = target_row_cf[CF_FEATURES].astype(float).values
                target_ls_cf = float(_LS_CF.get(target_row_cf['League'], 50.0))

                tv = pd.to_numeric(target_row_cf.get('Market value'), errors='coerce')
                target_market_value_cf = float(manual_override_cf) if manual_override_cf and manual_override_cf > 0 \
                    else (float(tv) if pd.notna(tv) and tv > 0 else 2_000_000.0)

                club_profiles_cf = df_candidates_cf.groupby('Team')[CF_FEATURES].mean().reset_index()

                team_league_cf = df_candidates_cf.groupby('Team')['League'].agg(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
                )
                team_market_cf = df_candidates_cf.groupby('Team')['Market value'].mean()
                club_profiles_cf['League'] = club_profiles_cf['Team'].map(team_league_cf)
                club_profiles_cf['Avg Team Market Value'] = club_profiles_cf['Team'].map(team_market_cf)
                club_profiles_cf = club_profiles_cf.dropna(subset=['Avg Team Market Value'])

                from sklearn.preprocessing import StandardScaler
                scaler_cf = StandardScaler()
                X_team = scaler_cf.fit_transform(club_profiles_cf[CF_FEATURES])
                x_tgt = scaler_cf.transform([target_vector_cf])[0]
                weights_vec_cf = np.array([weights_ui_cf.get(f, 1) for f in CF_FEATURES], dtype=float)

                dist_cf = np.linalg.norm((X_team - x_tgt) * weights_vec_cf, axis=1)
                rng = float(dist_cf.max() - dist_cf.min())
                club_fit_base = (1 - (dist_cf - float(dist_cf.min())) / (rng if rng > 0 else 1.0)) * 100.0
                club_profiles_cf['Club Fit %'] = club_fit_base.round(2)

                club_profiles_cf['League strength'] = club_profiles_cf['League'].map(_LS_CF).fillna(50.0)
                club_profiles_cf = club_profiles_cf[
                    (club_profiles_cf['League strength'] >= float(min_strength_cf)) &
                    (club_profiles_cf['League strength'] <= float(max_strength_cf))
                ]

                if club_profiles_cf.empty:
                    st.info("No teams remain after league-strength filter.")
                else:
                    ratio_cf = (club_profiles_cf['League strength'] / target_ls_cf).clip(0.5, 1.2)
                    club_profiles_cf['Adjusted Fit %'] = (
                        club_profiles_cf['Club Fit %'] * (1 - league_weight_cf) +
                        club_profiles_cf['Club Fit %'] * ratio_cf * league_weight_cf
                    )
                    league_gap_cf = (club_profiles_cf['League strength'] - target_ls_cf).clip(lower=0)
                    penalty_cf = (1 - (league_gap_cf / 100)).clip(lower=0.7)
                    club_profiles_cf['Adjusted Fit %'] *= penalty_cf

                    value_fit_ratio_cf = (club_profiles_cf['Avg Team Market Value'] / target_market_value_cf).clip(0.5, 1.5)
                    value_fit_score_cf = (1 - abs(1 - value_fit_ratio_cf)) * 100.0

                    club_profiles_cf['Final Fit %'] = (
                        club_profiles_cf['Adjusted Fit %'] * (1 - market_value_weight_cf) +
                        value_fit_score_cf * market_value_weight_cf
                    )

                    results_cf = club_profiles_cf[
                        ['Team','League','League strength','Club Fit %','Adjusted Fit %','Final Fit %']
                    ].copy().sort_values('Final Fit %', ascending=False).reset_index(drop=True)
                    results_cf.insert(0, 'Rank', np.arange(1, len(results_cf) + 1))

                    st.caption(
                        f"Target: {target_player_val} — {target_row_cf.get('Team','Unknown')} ({target_row_cf['League']}) • "
                        f"Target MV used: €{target_market_value_cf:,.0f} • Target LS {target_ls_cf:.2f} • "
                        f"Candidates: {len(leagues_selected_cf)} leagues (preset: {preset_name_cf})"
                    )
                    st.dataframe(results_cf.head(int(top_n_cf)), use_container_width=True)

                    csv_cf = results_cf.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download all results (CSV)", data=csv_cf, file_name="club_fit_results.csv", mime="text/csv")

    else:
        st.info("Pick a player to run Club Fit.")
# ---------------------------- END Club Fit ----------------------------

# ----------------- GBE CALCULATOR (FA 2025/26) -----------------
# Relies on df_f, player_name, player_row, extract_country already defined above

st.subheader("🧮 GBE Calculator (FA 2025/26 snapshot)")

if player_row.empty:
    st.info("Select a player in the Single Player Role Profile above to see their GBE snapshot.")
else:
    # ========= Helper: league → FA Band (1–6) =========
    LEAGUE_TO_GBE_BAND = {
        # Band 1 – Big 5
        "England 1.": 1,
        "Germany 1.": 1,
        "Spain 1.": 1,
        "Italy 1.": 1,
        "France 1.": 1,

        # Band 2
        "Portugal 1.": 2,
        "Netherlands 1.": 2,
        "Belgium 1.": 2,
        "Turkey 1.": 2,
        "England 2.": 2,

        # Band 3
        "USA 1.": 3,
        "Brazil 1.": 3,
        "Argentina 1.": 3,
        "Mexico 1.": 3,
        "Scotland 1.": 3,

        # Band 4
        "Czech 1.": 4,
        "Croatia 1.": 4,
        "Switzerland 1.": 4,
        "Spain 2.": 4,
        "Germany 2.": 4,
        "Ukraine 1.": 4,
        "Greece 1.": 4,
        "Colombia 1.": 4,
        "Austria 1.": 4,
        "Denmark 1.": 4,
        "France 2.": 4,
        "Russia 1.": 4,

        # Band 5
        "Serbia 1.": 5,
        "Poland 1.": 5,
        "Slovenia 1.": 5,
        "Chile 1.": 5,
        "Uruguay 1.": 5,
        "Sweden 1.": 5,
        "Norway 1.": 5,
        "Italy 2.": 5,
        "Hungary 1.": 5,
        "Japan 1.": 5,
        "Korea 1.": 5,
        "Australia 1.": 5,
        "England 3.": 5,
        # all others => Band 6
    }

    def gbe_band_for_league(league_name: str) -> int:
        return int(LEAGUE_TO_GBE_BAND.get(str(league_name).strip(), 6))

    # ========= Helper: International appearances (Table 1) =========
    def intl_points_and_auto(fifa_rank: int, pct: int) -> tuple[int, bool]:
        """
        Returns (points, auto_pass) according to Table 1
        (FIFA ranking band × % of competitive senior internationals).
        """
        if fifa_rank <= 10:
            band = "1-10"
        elif fifa_rank <= 20:
            band = "11-20"
        elif fifa_rank <= 30:
            band = "21-30"
        elif fifa_rank <= 50:
            band = "31-50"
        else:
            band = "51+"

        auto = False
        pts = 0
        p = int(pct)

        if band in ("1-10", "11-20", "21-30", "31-50"):
            if p >= 90:
                auto = True
            elif p >= 80:
                auto = True
            elif p >= 70:
                auto = True
            elif p >= 60:
                if band in ("1-10", "11-20", "21-30"):
                    auto = True
                else:  # 31–50
                    pts = 10
            elif p >= 50:
                if band in ("1-10", "11-20"):
                    auto = True
                elif band == "21-30":
                    pts = 10
                elif band == "31-50":
                    pts = 8
            elif p >= 40:
                if band in ("1-10", "11-20"):
                    auto = True
                elif band == "21-30":
                    pts = 9
                elif band == "31-50":
                    pts = 7
            elif p >= 30:
                if band == "1-10":
                    auto = True
                elif band == "11-20":
                    pts = 10
                elif band == "21-30":
                    pts = 8
                elif band == "31-50":
                    pts = 6
            elif p >= 20:
                if band == "1-10":
                    pts = 10
                elif band == "11-20":
                    pts = 9
                elif band == "21-30":
                    pts = 7
            elif p >= 10:
                if band == "1-10":
                    pts = 9
                elif band == "11-20":
                    pts = 8
            elif p >= 1:
                if band == "1-10":
                    pts = 8
                elif band == "11-20":
                    pts = 7
        else:
            # 51+ column
            if p >= 90:
                pts = 2
            elif p >= 80:
                pts = 1

        return int(pts), bool(auto)

    # ========= Helper: Domestic league minutes (Table 2) =========
    def table2_minutes_points(band: int, pct: int, youth_debut: bool) -> int:
        """
        Domestic league minutes points, including Youth Player debut override
        per para 35 (final row of Table 2). :contentReference[oaicite:1]{index=1}
        """
        band = int(band)
        p = int(pct)
        # rows: 90–100, 80–89, 70–79, 60–69, 50–59, 40–49, 30–39
        row_90 = [12, 10, 8, 6, 4, 2]
        row_80 = [11,  9, 7, 5, 3, 1]
        row_70 = [10,  8, 6, 4, 2, 0]
        row_60 = [ 9,  7, 5, 3, 1, 0]
        row_50 = [ 8,  6, 4, 2, 0, 0]
        row_40 = [ 7,  5, 3, 1, 0, 0]
        row_30 = [ 6,  4, 2, 0, 0, 0]
        debut  = [ 6,  5, 4, 3, 2, 1]

        idx = max(0, min(5, band - 1))

        base = 0
        if p >= 90:
            base = row_90[idx]
        elif p >= 80:
            base = row_80[idx]
        elif p >= 70:
            base = row_70[idx]
        elif p >= 60:
            base = row_60[idx]
        elif p >= 50:
            base = row_50[idx]
        elif p >= 40:
            base = row_40[idx]
        elif p >= 30:
            base = row_30[idx]

        debut_pts = debut[idx] if youth_debut else 0
        # Para 36: if eligible in multiple columns, grant the higher value. :contentReference[oaicite:2]{index=2}
        return int(max(base, debut_pts))

    # ========= Helper: Continental minutes (Table 3) =========
    def table3_continental_points(comp_band: int, pct: int) -> int:
        comp_band = int(comp_band)
        p = int(pct)
        band1 = [10, 9, 8, 7, 6, 5, 4]
        band2 = [ 5, 4, 3, 2, 1, 0, 0]
        band3 = [ 2, 1, 0, 0, 0, 0, 0]

        if comp_band == 1:
            row = band1
        elif comp_band == 2:
            row = band2
        else:
            row = band3

        if p >= 90:
            val = row[0]
        elif p >= 80:
            val = row[1]
        elif p >= 70:
            val = row[2]
        elif p >= 60:
            val = row[3]
        elif p >= 50:
            val = row[4]
        elif p >= 40:
            val = row[5]
        elif p >= 30:
            val = row[6]
        else:
            val = 0
        return int(val)

    # ========= Helper: Final league position (Table 4) =========
    FINAL_POS_ROWS = {
        "Title winner":                [6, 5, 4, 3, 2, 1],
        "Band1 group / conf winner":   [5, 4, 3, 2, 1, 0],
        "Band1 qualifiers":            [4, 3, 2, 1, 0, 0],
        "Band2 group":                 [3, 2, 1, 0, 0, 0],
        "Band2 qualifiers":            [2, 1, 0, 0, 0, 0],
        "Mid-table":                   [1, 0, 0, 0, 0, 0],
        "Relegation":                  [0, 0, 0, 0, 0, 0],
        "Promotion":                   [0, 1, 1, 1, 1, 1],  # Band1 => 0
    }

    def final_position_points(band: int, category: str) -> int:
        band = int(band)
        idx = max(0, min(5, band - 1))
        row = FINAL_POS_ROWS.get(category, [0, 0, 0, 0, 0, 0])
        return int(row[idx])

    # ========= Helper: Continental progression (Table 5) =========
    CONT_PROG_ROWS = {
        "Final":                     [10, 7, 2],
        "Semi-final":                [ 9, 6, 1],
        "Quarter-final":             [ 8, 5, 0],
        "Round of 16":               [ 7, 4, 0],
        "Round of 32 / KO PO":       [ 6, 3, 0],
        "Group / league phase":      [ 5, 2, 0],
        "Other":                     [ 0, 0, 0],
    }

    def continental_progression_points(comp_band: int, stage: str) -> int:
        comp_band = max(1, min(3, int(comp_band)))
        idx = comp_band - 1
        row = CONT_PROG_ROWS.get(stage, [0, 0, 0])
        return int(row[idx])

    # ========= Helper: League band points (Table 6) =========
    LEAGUE_BAND_POINTS = [12, 10, 8, 6, 4, 2]

    def league_quality_points(band: int) -> int:
        idx = max(0, min(5, int(band) - 1))
        return int(LEAGUE_BAND_POINTS[idx])

    # ========= Quick reference (helpers back in) =========
    with st.expander("📚 GBE helper – what gives points? (short version)", expanded=False):
        st.markdown(
            """
- **Senior international (Table 1)** – up to **Auto Pass** depending on:
  - FIFA World Ranking band of the nation (1–10, 11–20, 21–30, 31–50, 51+); and  
  - % of **available senior competitive internationals** played.
- **Domestic league minutes (Table 2)** – up to **12 pts** from:
  - **League band (1–6)** × % of **available league minutes** played.  
- **Continental minutes (Table 3)** – up to **10 pts**:
  - **Band 1**: Major top-tier cups (e.g. UEFA Champions League, Copa Libertadores).  
  - **Band 2**: Second-tier cups (e.g. UEFA Europa League, UEFA Conference League, Copa Sudamericana, major confederation CLs, Club World Cup).  
  - **Band 3**: All other recognised senior continental club competitions.
- **Final league position (Table 4)** – up to **6 pts** for title / continental places / promotions.
- **Continental progression (Table 5)** – up to **10 pts** for stage reached (Final → Group).  
- **League quality – current club (Table 6)** – up to **12 pts** from league band.  
- **Youth internationals** – used for ESC / Youth routes, **no points** in the standard 15-point total.
            """
        )

    # ========= Pull player info =========
    pr = player_row.iloc[0]
    player_team   = str(pr.get("Team", ""))
    player_league = str(pr.get("League", ""))
    player_minutes = float(pr.get("Minutes played", 0) or 0)

    birth_country = str(pr.get("Birth country", "") or "").strip()
    home_nation_names = {
        "england",
        "scotland",
        "wales",
        "northern ireland",
        "ireland",
        "republic of ireland",
    }
    birth_norm = birth_country.lower()
    is_home_nation_player = birth_norm in home_nation_names

    # League country via extract_country helper
    try:
        league_country = extract_country(player_league)
    except Exception:
        league_country = str(player_league).split(" ")[0]
    league_country_norm = league_country.strip().lower()
    is_home_nation_league = league_country_norm in home_nation_names

    # ========= 1) Auto domestic minutes % from dataset =========
    same_league = df_f[df_f["League"] == player_league]
    max_minutes_league = float(same_league["Minutes played"].max() or 0)

    if max_minutes_league > 0:
        raw_pct = 100.0 * player_minutes / max_minutes_league
        domestic_minutes_pct = int(math.floor(raw_pct + 0.5))
        domestic_minutes_pct = max(0, min(100, domestic_minutes_pct))
    else:
        domestic_minutes_pct = 0

    player_band = gbe_band_for_league(player_league)

    st.markdown(
        f"**League / Band:** {player_league}  →  **Band {player_band}**  "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; **Minutes:** {int(player_minutes)} "
        f"({domestic_minutes_pct}% of max minutes in this league sample)"
    )

    # ========= 2) Inputs =========
    st.markdown("### Inputs")

    col_intl, col_dom, col_other = st.columns([1.2, 1.0, 1.2])

    # ---- Senior international (Table 1) ----
    with col_intl:
        st.markdown("**Senior International (Table 1)**")
        use_intl = st.checkbox(
            "Include senior international appearances",
            value=False,
            key="gbe_use_intl",
        )
        intl_points = 0
        intl_auto = False

        if use_intl:
            fifa_rank = st.number_input(
                "Aggregated FIFA ranking of national team (1–200)",
                min_value=1, max_value=200, value=50, step=1,
                key="gbe_fifa_rank",
            )
            intl_pct = st.slider(
                "Player's appearances (% of available senior competitive internationals)",
                0, 100, 0, step=5, key="gbe_intl_pct",
            )
            intl_points, intl_auto = intl_points_and_auto(int(fifa_rank), int(intl_pct))

        st.write(f"International points (Table 1): **{intl_points}** · Auto pass: **{intl_auto}**")

    # ---- Domestic minutes (Table 2) ----
    with col_dom:
        st.markdown("**Domestic League Minutes (Table 2)**")

        # NEW: explicit Youth Player toggle (U21) and debut override (para 35). :contentReference[oaicite:3]{index=3}
        is_youth_player = st.checkbox(
            "Player is a Youth Player (U21 – born on or after 1 Jan 2004)",
            value=False,
            key="gbe_is_youth_player",
            help="Only Youth Players (U21) can receive the special debut points in the final row of Table 2.",
        )

        youth_debut = False
        if is_youth_player:
            youth_debut = st.checkbox(
                "Made first senior league appearance during the Reference Period (apply Youth debut points)",
                value=False,
                key="gbe_youth_debut",
                help="If ticked, the Table 2 'Debut for Youth Player' row is applied and can override minutes-based points.",
            )

        domestic_points = table2_minutes_points(player_band, domestic_minutes_pct, youth_debut)

        st.write(f"Domestic minutes % (auto): **{domestic_minutes_pct}%**")
        st.write(f"Domestic minutes points (Table 2): **{domestic_points}**")

    # ---- Other criteria (Tables 3–6) ----
    with col_other:
        st.markdown("**Other criteria (Tables 3–6)**")

        use_cont = st.checkbox("Add continental minutes (Table 3)", value=False, key="gbe_use_cont")
        cont_points = 0
        if use_cont:
            cont_band = st.selectbox(
                "Continental competition band",
                options=[1, 2, 3],
                format_func=lambda x: f"Band {x}",
                key="gbe_cont_band",
            )
            cont_pct = st.slider(
                "Player's continental minutes (% of available)",
                0, 100, 0, step=5, key="gbe_cont_pct",
            )
            cont_points = table3_continental_points(cont_band, cont_pct)

        use_finish = st.checkbox("Add final league position (Table 4)", value=False, key="gbe_use_finish")
        finish_points = 0
        if use_finish:
            finish_cat = st.selectbox(
                "Final league position category",
                options=[
                    "Title winner",
                    "Band1 group / conf winner",
                    "Band1 qualifiers",
                    "Band2 group",
                    "Band2 qualifiers",
                    "Mid-table",
                    "Relegation",
                    "Promotion",
                ],
                key="gbe_finish_cat",
            )
            finish_points = final_position_points(player_band, finish_cat)

        use_cprog = st.checkbox("Add continental progression (Table 5)", value=False, key="gbe_use_cprog")
        cprog_points = 0
        if use_cprog:
            cprog_band = st.selectbox(
                "Continental competition band (for progression)",
                options=[1, 2, 3],
                format_func=lambda x: f"Band {x}",
                key="gbe_cprog_band",
            )
            cprog_stage = st.selectbox(
                "Stage reached",
                options=[
                    "Final",
                    "Semi-final",
                    "Quarter-final",
                    "Round of 16",
                    "Round of 32 / KO PO",
                    "Group / league phase",
                    "Other",
                ],
                key="gbe_cprog_stage",
            )
            cprog_points = continental_progression_points(cprog_band, cprog_stage)

        # --- League band override (Table 6 points only) ---
        band_override_label = st.selectbox(
            "League band override (Table 6 only)",
            options=["Use mapped league band", "Band 1", "Band 2", "Band 3", "Band 4", "Band 5", "Band 6"],
            key="gbe_band_override",
            help="Use this if you want to credit points using a parent club's league band. "
                 "Domestic minutes still use the original band shown above.",
        )

        if band_override_label == "Use mapped league band":
            band_for_league_points = player_band
        else:
            band_for_league_points = int(band_override_label.split()[-1])

        use_lq = st.checkbox(
            "Add league band points – current club (Table 6)",
            value=True,
            key="gbe_use_lq",
        )
        lq_points = league_quality_points(band_for_league_points) if use_lq else 0

    # ==== Youth / extra notes – info only ====
    st.markdown("**Youth competitive internationals / extra notes (info only)**")
    st.caption("Used for ESC / Youth Player routes – **no points** in the standard 15-point total.")
    youth_int_caps = st.number_input(
        "Number of youth competitive international matches in reference period",
        min_value=0, max_value=100, value=0, step=1,
        key="gbe_youth_caps",
    )
    youth_caption = st.text_input(
        "Additional notes (optional)",
        value="",
        key="gbe_youth_caption",
    )

    # ========= Total & classification =========
    base_total_points = int(
        intl_points
        + domestic_points
        + cont_points
        + finish_points
        + cprog_points
        + lq_points
    )

    # ---- ESC eligibility (checkboxes, Band 6 focus) ----
    esc_eligible = False
    esc_reasons = []

    if player_band == 6:
        st.markdown("### ESC eligibility checks (Band 6 league)")
        st.caption(
            "Band 6 leagues do not count as Domestic Senior Competition Matches under the FA definitions. "
            "Players here usually proceed via an ESC slot if they are exceptional. "
            "Tick any ESC criteria this player actually meets in the reference period."
        )

        esc_youth_top50 = st.checkbox(
            "≥1 youth competitive international (Top-50 nation)",
            key="esc_youth_top50",
        )
        esc_youth_outside = st.checkbox(
            "≥5 youth competitive internationals (outside Top-50)",
            key="esc_youth_outside",
        )
        esc_youth_cont = st.checkbox(
            "≥1 youth continental club match",
            key="esc_youth_cont",
        )
        esc_youth_dom = st.checkbox(
            "≥5 domestic youth competition matches (played)",
            key="esc_youth_dom",
        )
        esc_senior_top50 = st.checkbox(
            "≥1 senior competitive international (Top-50 nation)",
            key="esc_senior_top50",
        )
        esc_senior_outside = st.checkbox(
            "≥5 senior competitive internationals (outside Top-50)",
            key="esc_senior_outside",
        )
        esc_senior_cont = st.checkbox(
            "≥1 senior continental club match",
            key="esc_senior_cont",
        )
        esc_senior_dom = st.checkbox(
            "≥5 domestic senior league matches in Band 1–5 (played, any club in reference period)",
            key="esc_senior_dom",
        )

        if esc_youth_top50:
            esc_eligible = True
            esc_reasons.append("Youth intl (Top-50)")
        if esc_youth_outside:
            esc_eligible = True
            esc_reasons.append("Youth intl (outside Top-50)")
        if esc_youth_cont:
            esc_eligible = True
            esc_reasons.append("Youth continental")
        if esc_youth_dom:
            esc_eligible = True
            esc_reasons.append("Domestic youth matches")
        if esc_senior_top50:
            esc_eligible = True
            esc_reasons.append("Senior intl (Top-50)")
        if esc_senior_outside:
            esc_eligible = True
            esc_reasons.append("Senior intl (outside Top-50)")
        if esc_senior_cont:
            esc_eligible = True
            esc_reasons.append("Senior continental")
        if esc_senior_dom:
            esc_eligible = True
            esc_reasons.append("Domestic senior (Band 1–5)")

    # Home-nation / Ireland auto-pass
    auto_reason = ""
    auto_source = None  # "home" or "intl"

    if is_home_nation_player or is_home_nation_league:
        auto_source = "home"
        bits = []
        if is_home_nation_player and birth_country:
            bits.append(f"{birth_country} national")
        if is_home_nation_league and league_country:
            bits.append(f"plays in {league_country} league")
        auto_reason = " · ".join(bits)
    elif intl_auto and use_intl:
        auto_source = "intl"
        auto_reason = "Meets automatic pass threshold via senior international appearances."

    display_points = base_total_points
    if auto_source is not None and display_points < 15:
        display_points = 15

    # ========= Status label logic (Band 1–5 vs Band 6; ESC) =========
    if auto_source == "home":
        status = "Automatic Pass – UK / Irish route"
        status_color = "#16a34a"
    elif auto_source == "intl":
        status = "Automatic Pass – Senior international"
        status_color = "#16a34a"
    else:
        if player_band == 6:
            # Band 6: ESC vs straight Fail
            if esc_eligible:
                status = "Fail / ESC"
                status_color = "#ea580c"
            else:
                status = "Fail"
                status_color = "#b91c1c"
        else:
            # Normal band 1–5 logic
            if display_points >= 15:
                status = "Pass"
                status_color = "#16a34a"
            elif display_points >= 10:
                status = "Exceptions Panel"
                status_color = "#fbbf24"  # yellow background for Exceptions Panel
            else:
                status = "Fail"
                status_color = "#b91c1c"

    bg_color = "#0b1220"
    breakdown_str = (
        f"Domestic minutes: {domestic_points} pts; "
        f"International: {intl_points} pts; "
        f"Continental minutes: {cont_points} pts; "
        f"League position: {finish_points} pts; "
        f"Continental progression: {cprog_points} pts; "
        f"League band: {lq_points} pts."
    )
    points_band_str = "0–9 = Fail/ESC, 10–14 = Exceptions Panel, 15+ = Pass."

    auto_reason_html = ""
    if auto_reason:
        auto_reason_html = (
            f"<div style='margin-top:0.35rem; font-size:0.8rem; color:#e5e7eb;'>"
            f"<strong>Auto-pass reason:</strong> {auto_reason}"
            f"</div>"
        )

    youth_html = ""
    if youth_caption.strip():
        youth_html = (
            f"<div style='margin-top:0.3rem; font-size:0.8rem; color:#cbd5f5;'>"
            f"{youth_caption.strip()}"
            f"</div>"
        )

    # ESC label at bottom
    esc_reason_html = ""
    if auto_source is None:
        # Band 6 players
        if player_band == 6:
            if esc_eligible and esc_reasons:
                esc_reason_html = (
                    f"<div style='margin-top:0.3rem; font-size:0.8rem; color:#fbbf24;'>"
                    f"<strong>ESC criteria met</strong> {', '.join(esc_reasons)}"
                    f"</div>"
                )
            else:
                esc_reason_html = (
                    f"<div style='margin-top:0.3rem; font-size:0.8rem; color:#f97373;'>"
                    f"<strong>ESC criteria not met</strong>"
                    f"</div>"
                )
        # Band 1–5 players who are in Exceptions Panel range (10–14 pts)
        elif 10 <= display_points < 15:
            esc_reason_html = (
                f"<div style='margin-top:0.3rem; font-size:0.8rem; color:#fbbf24;'>"
                f"<strong>ESC criteria met:</strong> Domestic senior (Band 1–5)"
                f"</div>"
            )

    # ========= Flag helper – Twemoji SVG for ALL flags (aligned) =========
    import unicodedata

    LOCAL_TWEMOJI_SPECIAL = {
        "eng": "1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",  # England
        "sct": "1f3f4-e0067-e0062-e0073-e0063-e0074-e007f",  # Scotland
        "wls": "1f3f4-e0067-e0062-e006c-e0073-e007f",        # Wales
        "nir": "1f3f4-e0067-e0062-e006e-e0069-e0072-e007f",  # Northern Ireland
    }

    LOCAL_COUNTRY_TO_CC = {
        # Home nations + Ireland
        "england": "eng",
        "scotland": "sct",
        "wales": "wls",
        "northern ireland": "nir",
        "ireland": "ie",
        "republic of ireland": "ie",

        # Europe
        "spain": "es",
        "germany": "de",
        "italy": "it",
        "france": "fr",
        "belgium": "be",
        "denmark": "dk",
        "poland": "pl",
        "turkey": "tr",
        "netherlands": "nl",
        "croatia": "hr",
        "switzerland": "ch",
        "norway": "no",
        "sweden": "se",
        "cyprus": "cy",
        "czech": "cz",
        "czech republic": "cz",
        "greece": "gr",
        "austria": "at",
        "romania": "ro",
        "albania": "al",
        "bosnia": "ba",
        "bosnia and herzegovina": "ba",
        "kosovo": "xk",
        "slovenia": "si",
        "slovakia": "sk",
        "bulgaria": "bg",
        "finland": "fi",
        "armenia": "am",
        "georgia": "ge",
        "iceland": "is",
        "north macedonia": "mk",
        "lithuania": "lt",
        "malta": "mt",
        "moldova": "md",
        "latvia": "lv",
        "montenegro": "me",
        "estonia": "ee",
        "russia": "ru",
         "portugal": "pt",
                    "hungary": "hu",
                    "ukraine": "ua",
                    "serbia": "rs",
                    "azerbaijan": "az",

        # Middle East & Asia
        "saudi": "sa",
        "saudi arabia": "sa",
        "uae": "ae",
        "united arab emirates": "ae",
        "qatar": "qa",
        "uzbekistan": "uz",
        "kazakhstan": "kz",
        "israel": "il",
        "japan": "jp",
        "korea": "kr",
        "south korea": "kr",
        "china": "cn",

        # Africa
        "morocco": "ma",
        "algeria": "dz",
        "egypt": "eg",
        "south africa": "za",
        "nigeria": "ng",
        "tunisia": "tn",

        # North & South America
        "brazil": "br",
        "argentina": "ar",
        "mexico": "mx",
        "colombia": "co",
        "ecuador": "ec",
        "paraguay": "py",
        "uruguay": "uy",
        "chile": "cl",
        "bolivia": "bo",
        "peru": "pe",
        "venezuela": "ve",
        "costa rica": "cr",
        "canada": "ca",
        "usa": "us",
        "united states": "us",

        # Oceania
        "australia": "au",
    }

    def _norm_local(s: str) -> str:
        if not s:
            return ""
        return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").strip().lower()

    def _iso_to_twemoji_hex(cc: str) -> str:
        # Turn "cz" → "1f1e8-1f1ff" style hex for Twemoji flag SVGs
        base = 0x1F1E6
        c1 = base + (ord(cc[0].upper()) - ord("A"))
        c2 = base + (ord(cc[1].upper()) - ord("A"))
        return f"{c1:x}-{c2:x}"

    def league_flag_html(league_name: str) -> str:
        country = extract_country(league_name)
        n = _norm_local(country)
        cc = LOCAL_COUNTRY_TO_CC.get(n, "")
        if not cc:
            return ""

        # Home nations with special TAG-sequence flags
        if cc in LOCAL_TWEMOJI_SPECIAL:
            code = LOCAL_TWEMOJI_SPECIAL[cc]
        else:
            # All normal ISO country codes use flag regional indicators → Twemoji
            if len(cc) != 2:
                return ""
            code = _iso_to_twemoji_hex(cc)

        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
        return (
            f"<span style='display:inline-block;vertical-align:middle;margin-left:0.35rem;'>"
            f"<img src='{src}' style='height:1.05rem;display:block;' alt='{country}'>"
            f"</span>"
        )

    league_flag_snippet = league_flag_html(player_league)

    # ========= Card layout (tweaked for mobile) =========
    st.markdown(
        f"""
<div style="
    border-radius: 0.9rem;
    padding: 1.0rem 1.25rem 0.95rem 1.25rem;
    margin-top: 0.9rem;
    background: {bg_color};
    border: 1px solid rgba(148, 163, 184, 0.5);
    color: #e5e7eb;
    font-size: 0.94rem;
">
  <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:0.75rem; margin-bottom:0.45rem;">
    <div style="font-size:1.05rem; color:#cbd5f5;">
      <div style="font-weight:600;">GBE / Visa points</div>
      <div style="white-space:nowrap;">
        <span style="font-weight:800; color:#f9fafb; font-size:1.05rem;">
          {player_name}
        </span>
        <span style="opacity:0.85; font-size:1.05rem; margin-left:0.15rem;">
          ({player_team})
        </span>
        {league_flag_snippet}
      </div>
    </div>
    <div style="
        font-size:0.82rem;
        color:#9ca3af;
        text-align:right;
        max-width:40%;
        line-height:1.3;
        word-wrap:break-word;
    ">
      {player_league}<br>
      Band {player_band}
    </div>
  </div>

  <div style="display:flex; align-items:center; gap:1.0rem; margin:0.2rem 0 0.6rem 0;">
    <div>
      <div style="font-size:2.35rem; font-weight:800; line-height:1; letter-spacing:0.02em;">
        {display_points}
      </div>
      <div style="font-size:0.8rem; color:#9ca3af; margin-top:0.18rem; white-space:nowrap;">
        Est. points
      </div>
    </div>
    <div style="
        padding:0.45rem 0.95rem;
        border-radius:999px;
        background:{status_color};
        color:#ffffff;
        font-weight:700;
        font-size:0.9rem;
        white-space:nowrap;
    ">
      {status}
    </div>
  </div>

  <div style="font-size:0.82rem; margin-bottom:0.45rem; color:#cbd5f5;">
    {points_band_str}
  </div>

  <!-- subtle divider line above Breakdown -->
  <div style="height:1px; background:rgba(148,163,184,0.35); margin:0.4rem 0 0.35rem 0;"></div>

  <div style="margin-top:0.05rem; font-size:0.82rem; color:#cbd5f5;">
    <span style="font-weight:700; color:#f9fafb;">Breakdown</span> – {breakdown_str}
  </div>

  {auto_reason_html}
  {youth_html}
  {esc_reason_html}
</div>
""",
        unsafe_allow_html=True,
    )

    # ========= Download snapshot AS IMAGE (PNG) =========
    import io

    fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=150)
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.axis("off")

    # Left: big score
    ax.text(
        0.05, 0.7, str(display_points),
        transform=ax.transAxes,
        fontsize=36,
        fontweight="bold",
        color="white",
        va="center",
    )
    ax.text(
        0.05, 0.5, "Est. points",
        transform=ax.transAxes,
        fontsize=10,
        color="#9ca3af",
        va="center",
    )

    # Title & player
    ax.text(
        0.25, 0.85, "GBE / Visa points",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="semibold",
        color="#cbd5f5",
        va="center",
    )
    ax.text(
        0.25, 0.7, f"{player_name} ({player_team})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="white",
        va="center",
    )
    ax.text(
        0.25, 0.55, f"{player_league} • Band {player_band}",
        transform=ax.transAxes,
        fontsize=9,
        color="#9ca3af",
        va="center",
    )

    # Status bar
    ax.text(
        0.25, 0.35, status,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="white",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=status_color,
            edgecolor="none",
        ),
    )

    # Points band line
    ax.text(
        0.05, 0.25, points_band_str,
        transform=ax.transAxes,
        fontsize=9,
        color="#cbd5f5",
        va="center",
    )

    # Breakdown (single wrapped line)
    ax.text(
        0.05, 0.11, f"Breakdown – {breakdown_str}",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#cbd5f5",
        va="center",
        wrap=True,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    img_bytes = buf.getvalue()

    st.download_button(
        "⬇️ Download GBE / Visa snapshot (.png)",
        data=img_bytes,
        file_name=f"gbe_snapshot_{player_name.replace(' ', '_')}.png",
        mime="image/png",
        key="gbe_download_image_btn",
    )


