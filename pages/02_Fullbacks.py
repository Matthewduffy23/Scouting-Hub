# app.py — Advanced Scouting + Notes + Comparison Radar + Similar Players + Club Fit
# Single file, drop-in. Requires: streamlit, pandas, numpy, matplotlib.
# scikit-learn is optional; a tiny StandardScaler fallback is included.

import os
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

# ----------------- PAGE -----------------
st.set_page_config(page_title="Advanced Fullback Scouting System", layout="wide")
st.title("🔎 Advanced Fullback Scouting System")
st.caption("Use the sidebar to shape your pool. Each section explains what you’re seeing and why.")

# ----------------- CONFIG -----------------
INCLUDED_LEAGUES = [
    'England 1.', 'England 2.', 'England 3.', 'England 4.', 'England 5.',
    'England 6.', 'England 7.', 'England 8.', 'England 9.', 'England 10.',
    'Albania 1.', 'Algeria 1.', 'Andorra 1.', 'Argentina 1.', 'Armenia 1.',
    'Australia 1.', 'Austria 1.', 'Austria 2.', 'Azerbaijan 1.', 'Belgium 1.',
    'Belgium 2.', 'Bolivia 1.', 'Bosnia 1.', 'Brazil 1.', 'Brazil 2.', 'Brazil 3.',
    'Bulgaria 1.', 'Canada 1.', 'Chile 1.', 'Colombia 1.', 'Costa Rica 1.',
    'Croatia 1.', 'Cyprus 1.', 'Czech 1.', 'Czech 2.', 'Denmark 1.', 'Denmark 2.',
    'Ecuador 1.', 'Egypt 1.', 'Estonia 1.', 'Finland 1.', 'France 1.', 'France 2.',
    'France 3.', 'Georgia 1.', 'Germany 1.', 'Germany 2.', 'Germany 3.', 'Germany 4.',
    'Greece 1.', 'Hungary 1.', 'Iceland 1.', 'Israel 1.', 'Israel 2.', 'Italy 1.',
    'Italy 2.', 'Italy 3.', 'Japan 1.', 'Japan 2.', 'Kazakhstan 1.', 'Korea 1.',
    'Latvia 1.', 'Lithuania 1.', 'Malta 1.', 'Mexico 1.', 'Moldova 1.', 'Morocco 1.',
    'Netherlands 1.', 'Netherlands 2.', 'North Macedonia 1.', 'Northern Ireland 1.',
    'Norway 1.', 'Norway 2.', 'Paraguay 1.', 'Peru 1.', 'Poland 1.', 'Poland 2.',
    'Portugal 1.', 'Portugal 2.', 'Portugal 3.', 'Qatar 1.', 'Ireland 1.', 'Romania 1.',
    'Russia 1.', 'Saudi 1.', 'Scotland 1.', 'Scotland 2.', 'Scotland 3.', 'Serbia 1.',
    'Serbia 2.', 'Slovakia 1.', 'Slovakia 2.', 'Slovenia 1.', 'Slovenia 2.', 'South Africa 1.',
    'Spain 1.', 'Spain 2.', 'Spain 3.', 'Sweden 1.', 'Sweden 2.', 'Switzerland 1.',
    'Switzerland 2.', 'Tunisia 1.', 'Turkey 1.', 'Turkey 2.', 'Ukraine 1.', 'UAE 1.',
    'USA 1.', 'USA 2.', 'Uruguay 1.', 'Uzbekistan 1.', 'Venezuela 1.', 'Wales 1.'
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

FEATURES = [
    'Successful defensive actions per 90', 'Defensive duels per 90', 'Defensive duels won, %',
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
    'Progressive passes per 90', 'Accurate progressive passes, %' ]

POLAR_METRICS = [
    "Defensive duels per 90","Defensive duels won, %","PAdj Interceptions", "Aerial duels won, %",
    "Passes per 90","Accurate passes, %","Progressive passes per 90",
    "Progressive runs per 90","Dribbles per 90",
    "xA per 90","Passes to penalty area per 90"
]

# -------- Position filter (central midfielders) --------
CM_PREFIXES = ('LB', 'LWB', 'RB', 'RWB')

def position_filter(pos):
    return str(pos).strip().upper().startswith(CM_PREFIXES)

# -------------------------------------------

# Role buckets
ROLES = {
    'Build Up FB': {
        'metrics': {
            'Passes per 90': 2, 'Accurate passes, %': 1.5, 'Forward passes per 90': 2, 
            'Accurate forward passes, %': 2, 'Progressive passes per 90': 2.5, 'Progressive runs per 90': 2, 
            'Dribbles per 90': 2, 'Deep completions per 90': 1, 'Passes to final third per 90': 2, 
            'Smart passes per 90': 1, 'xA per 90': 1, 
        }
    },
    'Attacking FB': {
        'metrics': {
            'Crosses per 90': 2, 'Accurate crosses, %': 1, 'Dribbles per 90': 3.5, 
            'Accelerations per 90': 2, 'Successful dribbles, %': 1,
            'Touches in box per 90': 2,  'Progressive runs per 90': 3, 
            'Passes to penalty area per 90': 2.5, 'xA per 90': 3,
        }
    },
    'Defensive FB': {
        'metrics': {
            'Aerial duels per 90': 1, 'Aerial duels won, %': 1.5, 'Defensive duels per 90': 2,
            'PAdj Interceptions': 3, 'Shots blocked per 90': 1, 'Defensive duels won, %': 3.5, 
        }
    },
    'All In': {
        'metrics': {
            'Progressive passes per 90': 2, 'xA per 90': 2, 'PAdj Interceptions': 2,
            'Progressive runs per 90': 2, 'Defensive duels won, %': 2, 'Dribbles per 90': 1.5,
        }
    }
}


LEAGUE_STRENGTHS = {
    'England 1.':100.00,'Italy 1.':97.14,'Spain 1.':94.29,'Germany 1.':94.29,'France 1.':91.43,
    'Brazil 1.':82.86,'England 2.':71.43,'Portugal 1.':71.43,'Argentina 1.':71.43,
    'Belgium 1.':68.57,'Mexico 1.':68.57,'Turkey 1.':65.71,'Germany 2.':65.71,'Spain 2.':65.71,
    'France 2.':65.71,'USA 1.':65.71,'Russia 1.':65.71,'Colombia 1.':62.86,'Netherlands 1.':62.86,
    'Austria 1.':62.86,'Switzerland 1.':62.86,'Denmark 1.':62.86,'Croatia 1.':62.86,
    'Japan 1.':62.86,'Korea 1.':62.86,'Italy 2.':62.86,'Czech 1.':57.14,'Norway 1.':57.14,
    'Poland 1.':57.14,'Romania 1.':57.14,'Israel 1.':57.14,'Algeria 1.':57.14,'Paraguay 1.':57.14,
    'Saudi 1.':57.14,'Uruguay 1.':57.14,'Morocco 1.':57.00,'Brazil 2.':56.00,'Ukraine 1.':55.00,
    'Ecuador 1.':54.29,'Spain 3.':54.29,'Scotland 1.':58.00,'Chile 1.':51.43,'Cyprus 1.':51.43,
    'Portugal 2.':51.43,'Slovakia 1.':51.43,'Australia 1.':51.43,'Hungary 1.':51.43,'Egypt 1.':51.43,
    'England 3.':51.43,'France 3.':48.00,'Japan 2.':48.00,'Bulgaria 1.':48.57,'Slovenia 1.':48.57,
    'Venezuela 1.':48.00,'Germany 3.':45.71,'Albania 1.':44.00,'Serbia 1.':42.86,'Belgium 2.':42.86,
    'Bosnia 1.':42.86,'Kosovo 1.':42.86,'Nigeria 1.':42.86,'Azerbaijan 1.':50.00,'Bolivia 1.':50.00,
    'Costa Rica 1.':50.00,'South Africa 1.':50.00,'UAE 1.':50.00,'Georgia 1.':40.00,'Finland 1.':40.00,
    'Italy 3.':40.00,'Peru 1.':40.00,'Tunisia 1.':40.00,'USA 2.':40.00,'Armenia 1.':40.00,
    'North Macedonia 1.':40.00,'Qatar 1.':40.00,'Uzbekistan 1.':42.00,'Norway 2.':42.00,
    'Kazakhstan 1.':42.00,'Poland 2.':38.00,'Denmark 2.':37.00,'Czech 2.':37.14,'Israel 2.':37.14,
    'Netherlands 2.':37.14,'Switzerland 2.':37.14,'Iceland 1.':34.29,'Ireland 1.':34.29,'Sweden 2.':34.29,
    'Germany 4.':34.29,'Malta 1.':30.00,'Turkey 2.':31.43,'Canada 1.':28.57,'England 4.':28.57,
    'Scotland 2.':28.57,'Moldova 1.':28.57,'Austria 2.':25.71,'Lithuania 1.':25.71,'Brazil 3.':25.00,
    'England 7.':25.00,'Slovenia 2.':22.00,'Latvia 1.':22.86,'Serbia 2.':20.00,'Slovakia 2.':20.00,
    'England 9.':20.00,'England 8.':15.00,'Montenegro 1.':14.29,'Wales 1.':12.00,'Portugal 3.':11.43,
    'Northern Ireland 1.':11.43,'England 10.':10.00,'Scotland 3.':10.00,'England 6.':10.00
}
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}

# ----------------- DATA LOADER -----------------
from pathlib import Path
import io
import pandas as pd
import streamlit as st

# ---------- CACHED READERS (no widgets here) ----------
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    """
    Tries several locations for the CSV:
    1) Current working dir (repo root on Streamlit Cloud)
    2) Parent of this file (..), then this file's folder
    Falls back to a file uploader (widget OUTSIDE cache).
    """
    # 1) repo root / working directory
    candidates = [
        Path.cwd() / csv_name,
        Path(__file__).resolve().parent.parent / csv_name,  # ../WORLDJUNE25.csv
        Path(__file__).resolve().parent / csv_name,         # ./WORLDJUNE25.csv (same folder as the page)
    ]

    for p in candidates:
        if p.exists():
            return _read_csv_from_path(str(p))

    # ---------- Fallback: let the user upload (widget OUTSIDE cache) ----------
    st.warning(
        f"Could not find **{csv_name}** in expected locations.\n\n"
        "Please upload the CSV file below."
    )
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None:
        st.stop()
    return _read_csv_from_bytes(up.getvalue())


df = load_df()


# ----------------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.header("Filters")
    c1, c2, c3 = st.columns([1,1,1])
    use_top5  = c1.checkbox("Top-5 EU", value=False)
    use_top20 = c2.checkbox("Top-20 EU", value=False)
    use_efl   = c3.checkbox("EFL", value=False)

    seed = set()
    if use_top5:  seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top20: seed |= PRESET_LEAGUES["Top 20 Europe"]
    if use_efl:   seed |= PRESET_LEAGUES["EFL (England 2–4)"]

    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df.get("League", pd.Series([])).dropna().unique()))
    default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES
    leagues_sel = st.multiselect("Leagues (add or prune the presets)", leagues_avail, default=default_leagues)

    # numeric coercions
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (1000, 5000))
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 40))

    pos_text = st.text_input("Position startswith", "CF")

    # Defaults OFF; league beta default shown as 0.40 but toggle unticked
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    use_league_weighting = st.checkbox("Use league weighting in role score", value=False)
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.40, 0.05,
                     help="0 = ignore league strength; 1 = only league strength")

    # Market value
    df["Market value"] = pd.to_numeric(df["Market value"], errors="coerce")
    mv_col = "Market value"
    mv_max_raw = int(np.nanmax(df[mv_col])) if df[mv_col].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    st.markdown("**Market value (€)**")
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, max_m))
        min_value = mv_min_m * 1_000_000
        max_value = mv_max_m * 1_000_000
    else:
        min_value, max_value = st.slider("Range (€)", 0, mv_cap, (0, mv_cap), step=100_000)
    value_band_max = st.number_input("Value band (tab 4 max €)", min_value=0,
                                     value=min_value if min_value>0 else 5_000_000, step=250_000)

    st.subheader("Minimum performance thresholds")
    enable_min_perf = st.checkbox("Require minimum percentile on selected metrics", value=False)
    sel_metrics = st.multiselect("Metrics to threshold", FEATURES[:],
                                 default=['Non-penalty goals per 90','xG per 90'] if enable_min_perf else [])
    min_pct = st.slider("Minimum percentile (0–100)", 0, 100, 60)

    top_n = st.number_input("Top N per table", 5, 200, 50, 5)
    round_to = st.selectbox("Round output percentiles to", [0, 1], index=0)

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
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Position"].astype(str).apply(position_filter)]
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
df_f = df_f.dropna(subset=FEATURES)

df_f["Contract expires"] = pd.to_datetime(df_f["Contract expires"], errors="coerce")
if apply_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]

df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
df_f = df_f[(df_f["League Strength"] >= float(min_strength)) & (df_f["League Strength"] <= float(max_strength))]
df_f = df_f[(df_f["Market value"] >= min_value) & (df_f["Market value"] <= max_value)]
if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# ----------------- PERCENTILES FOR TABLES (per league) -----------------
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)
for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

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
        league_scaled = (df_in["League Strength"].fillna(50) / 100.0) * 100.0
        return (1 - beta) * player_score + beta * league_scaled
    return player_score

for role_name, role_def in ROLES.items():
    df_f[f"{role_name} Score"] = compute_weighted_role_score(df_f, role_def["metrics"], beta=beta, league_weighting=use_league_weighting)

# ----------------- THRESHOLDS -----------------
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
def fmt_cols(df_in: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df_in.copy()
    out[score_col] = out[score_col].round(round_to).astype(int if round_to == 0 else float)
    cols = ["Player","Team","League","Position", "Age","Contract expires","League Strength", score_col]
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
        t = t[t["Contract expires"].dt.year <= contract_year]
    if value_max is not None:
        t = t[t["Market value"] <= value_max]
    return t

# ----------------- TABS (tables) -----------------
tabs = st.tabs(["Overall Top N", "U23 Top N", "Expiring Contracts", "Value Band (≤ max €)"])
for role, role_def in ROLES.items():
    with tabs[0]:
        st.subheader(f"{role} — Overall Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(df_f, role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[1]:
        u23_cutoff = st.number_input(f"{role} — U23 cutoff", min_value=16, max_value=30, value=23, step=1, key=f"u23_{role}")
        st.subheader(f"{role} — U{u23_cutoff} Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, age_max=u23_cutoff), role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[2]:
        exp_year = st.number_input(f"{role} — Expiring by year", min_value=2024, max_value=2030, value=cutoff_year, step=1, key=f"exp_{role}")
        st.subheader(f"{role} — Contracts expiring ≤ {exp_year}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, contract_year=exp_year), role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[3]:
        v_max = st.number_input(f"{role} — Max value (€)", min_value=0, value=value_band_max, step=100_000, key=f"val_{role}")
        st.subheader(f"{role} — Value band ≤ €{v_max:,.0f}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, value_max=v_max), role, int(top_n)), use_container_width=True)
        st.divider()

# ----------------- METRIC LEADERBOARD (9/10 polished) -----------------
import re, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams, font_manager as fm

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
    default_metric = "Progressive runs per 90" if "Progressive runs per 90" in FEATURES else FEATURES[0]
    metric_pick   = st.selectbox("Metric", FEATURES, index=FEATURES.index(default_metric))
    top_n         = st.slider("Top N", 5, 40, 20, 5)

# --- Data
val_col = metric_pick
plot_df = df_f[["Player","Team",val_col]].dropna(subset=[val_col]).copy()
plot_df = plot_df.sort_values(val_col, ascending=False).head(int(top_n)).reset_index(drop=True)

# Label formatter "M.Grimes, Coventry"
def label_name_team(player, team):
    tokens = re.split(r"\s+", str(player).strip())
    if tokens:
        initial = tokens[0][0]
        last = re.sub(r"[^\w\-’']", "", tokens[-1])
        name = f"{initial}.{last}"
    else:
        name = str(player)
    return f"{name}, {team}"

y_labels = [label_name_team(r.Player, r.Team) for r in plot_df.itertuples(index=False)]
vals = plot_df[val_col].astype(float).values

# --- Medium→Dark Blue palette (subtle; bottom still medium blue)
# higher = darker, lower = medium (not pale)
cmap = LinearSegmentedColormap.from_list(
    "medium_to_dark_blue",
    ["#5F8EF1",  # medium blue (min)
     "#2F63D4",  # mid blue
     "#0A2A66"]  # deep navy (max)
)
norm   = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
colors = [cmap(norm(v)) for v in vals]

# --- Figure
fig, ax = plt.subplots(figsize=(11.5, 6.2))
page_grey = "#f3f4f6"
fig.patch.set_facecolor(page_grey)
ax.set_facecolor(page_grey)

# Title
fig.suptitle(f"Top {len(plot_df)} – {metric_pick}",
             fontsize=16, fontweight="bold", color="#111827", y=0.985)
plt.subplots_adjust(top=0.90, left=0.27, right=0.965, bottom=0.14)

# Bars
bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="none", zorder=2)

# Axes & labels
ax.invert_yaxis()
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(y_labels, fontsize=10.5, color="#0f172a")
ax.set_ylabel("")
ax.set_xlabel(val_col, color="#111827", labelpad=6, fontsize=9.5)  # smaller label

# Gridlines
ax.grid(axis="x", color="#e7e9ec", linewidth=0.7, zorder=1)

# Spines cleanup
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#d1d5db")
ax.tick_params(axis="x", labelsize=9, colors="#374151")

# X ticks
def fmt(x, _): return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"
ax.xaxis.set_major_formatter(FuncFormatter(fmt))
xmax = float(vals.max()) if len(vals) else 1.0
ax.set_xlim(0, xmax * 1.1)

# Value labels: small, plain, aligned neatly
pad = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
for rect, v in zip(bars, vals):
    ax.text(rect.get_width() + pad,
            rect.get_y() + rect.get_height()/2,
            fmt(v, None),
            va="center", ha="left", fontsize=8.5, color="#111827")

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
    min_minutes_pool, max_minutes_pool = c2.slider("Pool minutes", 0, 5000, (1000, 5000))
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

EXTRA_METRICS = [
    'Successful defensive actions per 90', 'Defensive duels per 90', 'Defensive duels won, %',
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
    'Long Passes per 90': {
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
        ("Succ. def acts", "Successful defensive actions per 90"),
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


# ----------------- (A) SCATTERPLOT — Goals vs xG -----------------
st.markdown("---")
st.header("📈 Scatterplot")

with st.expander("Scatter settings", expanded=False):
    # Axis metric picks (defaults as requested)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    x_default = "Progressive passes per 90"
    y_default = "Progressive runs per 90"
    x_metric = st.selectbox(
        "X-axis metric",
        [c for c in FEATURES if c in numeric_cols],
        index=(FEATURES.index(x_default) if x_default in FEATURES else 0),
        key="sc_x",
    )
    y_metric = st.selectbox(
        "Y-axis metric",
        [c for c in FEATURES if c in numeric_cols],
        index=(FEATURES.index(y_default) if y_default in FEATURES else 1),
        key="sc_y",
    )

    # Pool: default = player's league; presets + custom add-ons
    leagues_available_sc = sorted(df["League"].dropna().unique().tolist())
    player_league = player_row.iloc[0]["League"] if not player_row.empty else None

    preset_choices_sc = [
        "Player's league",
        "Top 5 Europe",
        "Top 20 Europe",
        "EFL (England 2–4)",
        "Custom",
    ]
    preset_sc = st.selectbox("League preset", preset_choices_sc, index=0, key="sc_preset")

    preset_map_sc = {
        "Player's league": {player_league} if player_league else set(),
        "Top 5 Europe": set(PRESET_LEAGUES.get("Top 5 Europe", [])),
        "Top 20 Europe": set(PRESET_LEAGUES.get("Top 20 Europe", [])),
        "EFL (England 2–4)": set(PRESET_LEAGUES.get("EFL (England 2–4)", [])),
        "Custom": set(),
    }
    preset_set = preset_map_sc.get(preset_sc, set())
    add_leagues_sc = st.multiselect("Add leagues", leagues_available_sc, default=[], key="sc_add_leagues")
    leagues_scatter = sorted(set(add_leagues_sc) | preset_set)

    # If user left it empty, fall back to player's league so the plot always works
    if not leagues_scatter and player_league:
        leagues_scatter = [player_league]

    same_pos_scatter = st.checkbox("Limit pool to current position prefix", value=True, key="sc_pos")

    # Filters: minutes, age, league strength (quality)
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes_s, max_minutes_s = st.slider("Minutes filter", 0, 5000, (1000, 5000), key="sc_min")
    age_min_bound = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_bound = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age_s, max_age_s = st.slider("Age filter", age_min_bound, age_max_bound, (16, 40), key="sc_age")

    min_strength_s, max_strength_s = st.slider("League quality (strength)", 0, 101, (0, 101), key="sc_ls")

    # Label & inclusion toggles
    include_selected = st.toggle("Include selected player", value=True, key="sc_include")
    label_all = st.toggle("Label ALL players in chart", value=False, key="sc_labels_all")
    allow_overlap = st.toggle("Allow overlapping labels", value=False, key="sc_overlap")

    # Visual improvements
    show_medians = st.checkbox("Show median reference lines", value=True, key="sc_medians")
    shade_iqr = st.checkbox("Shade interquartile range (25–75%)", value=True, key="sc_iqr")
    point_alpha = st.slider("Point opacity", 0.2, 1.0, 0.85, 0.05, key="sc_alpha")

# ---- Build scatter pool ----
try:
    pool_sc = df[df["League"].isin(leagues_scatter)].copy()
    if same_pos_scatter and not player_row.empty:
        pool_sc = pool_sc[pool_sc["Position"].astype(str).apply(position_filter)]

    # numeric + filters
    pool_sc["Minutes played"] = pd.to_numeric(pool_sc["Minutes played"], errors="coerce")
    pool_sc["Age"] = pd.to_numeric(pool_sc["Age"], errors="coerce")
    pool_sc = pool_sc[pool_sc["Minutes played"].between(min_minutes_s, max_minutes_s)]
    pool_sc = pool_sc[pool_sc["Age"].between(min_age_s, max_age_s)]

    # league quality filter
    pool_sc["League Strength"] = pool_sc["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
    pool_sc = pool_sc[
        (pool_sc["League Strength"] >= float(min_strength_s))
        & (pool_sc["League Strength"] <= float(max_strength_s))
    ]

    # Ensure metrics are numeric and present
    if x_metric not in pool_sc.columns or y_metric not in pool_sc.columns:
        st.info("Selected axis metrics are missing from the dataset.")
    else:
        pool_sc[x_metric] = pd.to_numeric(pool_sc[x_metric], errors="coerce")
        pool_sc[y_metric] = pd.to_numeric(pool_sc[y_metric], errors="coerce")
        pool_sc = pool_sc.dropna(subset=[x_metric, y_metric, "Player", "Team", "League"])

        # Selected player's name (regardless of inclusion toggle)
        selected_player_name = player_row.iloc[0]["Player"] if not player_row.empty else None

        # If excluded, make sure the selected player is NOT in the pool
        if not include_selected and selected_player_name is not None and not pool_sc.empty:
            pool_sc = pool_sc[pool_sc["Player"] != selected_player_name]

        # If included, ensure we add them even if filtered out above
        if include_selected and selected_player_name is not None:
            need_insert = True
            if not pool_sc.empty:
                need_insert = not (pool_sc["Player"] == selected_player_name).any()
            if need_insert:
                insertable = df[df["Player"] == selected_player_name].head(1).copy()
                if not insertable.empty:
                    insertable["League Strength"] = insertable["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
                    insertable[x_metric] = pd.to_numeric(insertable[x_metric], errors="coerce")
                    insertable[y_metric] = pd.to_numeric(insertable[y_metric], errors="coerce")
                    pool_sc = pd.concat([pool_sc, insertable], ignore_index=True, sort=False)

        # ----- Plot -----
        if pool_sc.empty:
            st.info("No players in scatter pool after filters.")
        else:
            fig, ax = plt.subplots(figsize=(9.4, 6.6), dpi=200)
            # Grey page & axis backgrounds
            fig.patch.set_facecolor("#f3f4f6")     # page bg
            ax.set_facecolor("#eeeeee")            # plot bg

            # Compute limits with a small padding so labels/points don't clip
            x_vals = pool_sc[x_metric].values
            y_vals = pool_sc[y_metric].values
            def padded_limits(arr, pad_frac=0.06):
                a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
                if a_min == a_max:
                    a_min -= 1e-6; a_max += 1e-6
                pad = (a_max - a_min) * pad_frac
                return a_min - pad, a_max + pad
            xlim = padded_limits(x_vals); ylim = padded_limits(y_vals)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

            # Determine whether we have a selected player to highlight
            sel_name = selected_player_name if include_selected else None

            # Others (black)
            others = pool_sc[pool_sc["Player"] != sel_name] if sel_name is not None else pool_sc
            ax.scatter(
                others[x_metric], others[y_metric],
                s=30, c="black", alpha=float(point_alpha), linewidths=0.4, edgecolors="white", zorder=2
            )

            # Selected player (red) + label (only once) if included
            already_labeled = set()
            if sel_name is not None:
                sel = pool_sc[pool_sc["Player"] == sel_name]
                ax.scatter(
                    sel[x_metric], sel[y_metric],
                    s=90, c="#C81E1E", edgecolors="white", linewidths=1.0, zorder=4
                )
                for _, r in sel.iterrows():
                    ax.annotate(
                        r["Player"], (r[x_metric], r[y_metric]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#C81E1E", zorder=5
                    )
                    already_labeled.add(r["Player"])

            # ----- Visual improvements -----
            # 1) Optional IQR shading (helps quick orientation)
            if shade_iqr:
                x_q1, x_q3 = np.nanpercentile(x_vals, [25, 75])
                y_q1, y_q3 = np.nanpercentile(y_vals, [25, 75])
                ax.axvspan(x_q1, x_q3, color="#d1d5db", alpha=0.25, zorder=1)
                ax.axhspan(y_q1, y_q3, color="#d1d5db", alpha=0.25, zorder=1)

            # 2) Median reference lines (dashed), with unified label "Median"
            if show_medians:
                med_x = float(np.nanmedian(x_vals)); med_y = float(np.nanmedian(y_vals))
                ax.axvline(med_x, color="#6b7280", ls="--", lw=1.25, zorder=1.5)
                ax.axhline(med_y, color="#6b7280", ls="--", lw=1.25, zorder=1.5)
                ax.text(med_x, ylim[1], "Median", ha="right", va="bottom",
                        fontsize=8, color="#374151", backgroundcolor="white", zorder=3, clip_on=True)
                ax.text(xlim[0], med_y, "Median", ha="left", va="top",
                        fontsize=8, color="#374151", backgroundcolor="white", zorder=3, clip_on=True)

            # 3) Optional labeling of ALL players (without duplication)
            if label_all:
                label_df = pool_sc
                # Simple overlap-avoidance: skip labels too close to an already-labeled point
                x_tol = (xlim[1] - xlim[0]) * 0.02
                y_tol = (ylim[1] - ylim[0]) * 0.02
                placed_pts = []

                # seed with the selected player's position(s) if present & included
                if sel_name is not None:
                    sel_seed = pool_sc[pool_sc["Player"] == sel_name]
                    for _, r in sel_seed.iterrows():
                        placed_pts.append((float(r[x_metric]), float(r[y_metric])))

                offsets = [(6,6), (-6,6), (6,-6), (-6,-6)]
                for i, (_, r) in enumerate(label_df.iterrows()):
                    pname = r["Player"]
                    # don't duplicate the selected player's label
                    if pname in already_labeled:
                        continue

                    px, py = float(r[x_metric]), float(r[y_metric])
                    if not allow_overlap:
                        too_close = any((abs(px - qx) < x_tol and abs(py - qy) < y_tol) for (qx, qy) in placed_pts)
                        if too_close:
                            continue
                        placed_pts.append((px, py))

                    dx, dy = offsets[i % len(offsets)]
                    ax.annotate(
                        pname, (px, py),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color="#111827", zorder=3
                    )

            # Styling: bold axis labels, light grid, subtle spines
            ax.set_xlabel(x_metric, fontweight="bold")
            ax.set_ylabel(y_metric, fontweight="bold")
            ax.grid(True, which="major", linewidth=0.7, color="#d1d5db")
            ax.grid(True, which="minor", linewidth=0.45, color="#e5e7eb", alpha=0.7)
            ax.minorticks_on()
            for spine in ax.spines.values():
                spine.set_edgecolor("#9ca3af")

            # Caption with pool size & leagues
            leagues_shown = ", ".join(sorted(set(pool_sc["League"])))
            st.caption(f"Pool size: {len(pool_sc):,} • Leagues: {leagues_shown}")
            st.pyplot(fig, use_container_width=True)
except Exception as e:
    st.info(f"Scatter could not be drawn: {e}")
# ----------------------------------------------------------------------
# ----------------- (B) COMPARISON RADAR — universal position_filter, A fixed, B any league -----------------
st.markdown("---")
st.header("📊 Player Comparison Radar")

DEFAULT_RADAR_METRICS = [
    "Defensive duels per 90","Defensive duels won, %","PAdj Interceptions","Aerial duels won, %",
    "Passes per 90","Accurate passes, %","Progressive passes per 90",
    "Progressive runs per 90","Dribbles per 90",
    "xA per 90","Passes to penalty area per 90"
]

def _clean_radar_label(s: str) -> str:
    s = s.replace("Aerial duels won, %", "Aerial %")
    s = s.replace("xA per 90", "xA")
    s = s.replace("Defensive duels won, %", "Def Duel %")
    s = s.replace("Defensive duels per 90", "Defensive duels").replace("Passes per 90", "Passes")
    s = s.replace("Progressive runs per 90", "Progressive Runs").replace("Progressive passes per 90", "Progressive Passes")
    s = s.replace("Passes to penalty area per 90", "Passes to Pen Area").replace("Accurate passes, %", "Pass %")
    return re.sub(r"\s*per\s*90", "", s, flags=re.I)

if player_row.empty:
    st.info("Pick a player above to draw the radar.")
else:
    # Player A is the selected player
    pA = player_name
    rowA_all = df[df["Player"] == pA]
    if rowA_all.empty:
        st.info("Selected player not found in dataset.")
    else:
        # Use the first occurrence if duplicates exist
        rowA = rowA_all.iloc[0]

        # Build Player B list using the SAME universal position_filter used elsewhere
        pool_pos = df[df["Position"].astype(str).apply(position_filter)].copy()
        players_b = sorted(pool_pos["Player"].dropna().unique().tolist())
        players_b = [p for p in players_b if p != pA]  # exclude A

        if not players_b:
            st.info("No comparison players available for the current universal position filter.")
        else:
            # Default B = first option (stable); user can change via dropdown
            pB = st.selectbox("Player B (blue)", players_b, index=0, key="radar_pb")

            rowB_all = df[df["Player"] == pB]
            if rowB_all.empty:
                st.info("Comparison player not found in dataset.")
            else:
                rowB = rowB_all.iloc[0]

                # Radar metrics that actually exist and are numeric
                numeric_cols = set(df.select_dtypes(include="number").columns.tolist())
                radar_metrics = [m for m in DEFAULT_RADAR_METRICS if m in df.columns and m in numeric_cols]
                if not radar_metrics:
                    st.info("No numeric radar metrics available in dataset.")
                else:
                    # Pooled comparison set: A∪B leagues, filtered by the same universal position_filter
                    union_leagues = {rowA["League"], rowB["League"]}
                    pool = df[
                        (df["League"].isin(union_leagues)) &
                        (df["Position"].astype(str).apply(position_filter))
                    ].copy()

                    # Numeric conversion + drop rows missing any radar metric
                    for m in radar_metrics:
                        pool[m] = pd.to_numeric(pool[m], errors="coerce")
                    pool = pool.dropna(subset=radar_metrics + ["Player"])

                    if pool.empty:
                        st.info("No players in the combined A∪B league pool after applying the universal position filter.")
                    else:
                        # Percentiles vs the A∪B pool
                        pool_pct = pool[radar_metrics].rank(pct=True) * 100.0

                        def pct_for(name: str) -> np.ndarray:
                            idx = pool[pool["Player"] == name].index
                            if len(idx) == 0:
                                return np.full(len(radar_metrics), np.nan)
                            # Average if multiple rows for a player exist
                            return pool_pct.loc[idx, :].mean(axis=0).values

                        A_r = pct_for(pA)
                        B_r = pct_for(pB)

                        # Labels & tick scaffolding (visual only)
                        labels = [_clean_radar_label(m) for m in radar_metrics]
                        axis_min = pool[radar_metrics].min().values
                        axis_max = pool[radar_metrics].max().values
                        pad = (axis_max - axis_min) * 0.07
                        axis_min = axis_min - pad
                        axis_max = axis_max + pad
                        axis_ticks = [np.linspace(axis_min[i], axis_max[i], 11) for i in range(len(labels))]

                        # ---- draw radar ----
                        COL_A = "#C81E1E"; COL_B = "#1D4ED8"
                        FILL_A = (200/255, 30/255, 30/255, 0.60)
                        FILL_B = (29/255, 78/255, 216/255, 0.60)
                        PAGE_BG = AX_BG = "#FFFFFF"
                        GRID_BAND_A = "#FFFFFF"; GRID_BAND_B = "#E5E7EB"
                        RING_COLOR = "#D1D5DB"; RING_LW = 1.0
                        LABEL_COLOR = "#0F172A"; TITLE_FS = 26; SUB_FS = 12; AXIS_FS = 10
                        TICK_FS = 7; TICK_COLOR = "#9CA3AF"; INNER_HOLE = 10

                        def draw_radar(labels, A_r, B_r, ticks, headerA, subA, headerB, subB):
                            N = len(labels)
                            theta = np.linspace(0, 2*np.pi, N, endpoint=False)
                            theta_c = np.concatenate([theta, theta[:1]])
                            Ar = np.concatenate([A_r, A_r[:1]])
                            Br = np.concatenate([B_r, B_r[:1]])

                            fig = plt.figure(figsize=(13.2, 8.0), dpi=260)
                            fig.patch.set_facecolor(PAGE_BG)
                            ax = plt.subplot(111, polar=True); ax.set_facecolor(AX_BG)
                            ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
                            ax.set_xticks(theta); ax.set_xticklabels(labels, fontsize=AXIS_FS, color=LABEL_COLOR, fontweight=600)
                            ax.set_yticks([]); ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]

                            # radial bands
                            for i in range(10):
                                r0, r1 = np.linspace(INNER_HOLE,100,11)[i], np.linspace(INNER_HOLE,100,11)[i+1]
                                band = GRID_BAND_A if i % 2 == 0 else GRID_BAND_B
                                ax.add_artist(Wedge((0,0), r1, 0, 360, width=(r1-r0),
                                                    transform=ax.transData._b, facecolor=band,
                                                    edgecolor="none", zorder=0.8))
                            ring_t = np.linspace(0, 2*np.pi, 361)
                            for r in np.linspace(INNER_HOLE,100,11):
                                ax.plot(ring_t, np.full_like(ring_t, r), color=RING_COLOR, lw=RING_LW, zorder=0.9)

                            # numeric axis tick labels around rings — light & small
                            start_idx = 2
                            for i, ang in enumerate(theta):
                                vals = ticks[i][start_idx:]
                                for rr, v in zip(np.linspace(INNER_HOLE,100,11)[start_idx:], vals):
                                    ax.text(ang, rr-1.8, f"{v:.1f}", ha="center", va="center",
                                            fontsize=TICK_FS, color=TICK_COLOR, zorder=1.1)
                            ax.add_artist(Circle((0,0), radius=INNER_HOLE-0.6, transform=ax.transData._b,
                                                 color=PAGE_BG, zorder=1.2, ec="none"))

                            # A & B polygons
                            ax.plot(theta_c, Ar, color=COL_A, lw=2.2, zorder=3)
                            ax.fill(theta_c, Ar, color=FILL_A, zorder=2.5)
                            ax.plot(theta_c, Br, color=COL_B, lw=2.2, zorder=3)
                            ax.fill(theta_c, Br, color=FILL_B, zorder=2.5)
                            ax.set_rlim(0, 105)

                            # headers (teams / leagues)
                            minsA = f"{int(pd.to_numeric(rowA.get('Minutes played',0))):,} mins" if pd.notna(rowA.get('Minutes played')) else "Minutes: N/A"
                            minsB = f"{int(pd.to_numeric(rowB.get('Minutes played',0))):,} mins" if pd.notna(rowB.get('Minutes played')) else "Minutes: N/A"

                            fig.text(0.12, 0.96,  headerA, color=COL_A, fontsize=TITLE_FS, fontweight="bold", ha="left")
                            fig.text(0.12, 0.935, subA, color=COL_A, fontsize=SUB_FS, ha="left")
                            fig.text(0.12, 0.915, minsA, color="#374151", fontsize=10, ha="left")

                            fig.text(0.88, 0.96,  headerB, color=COL_B, fontsize=TITLE_FS, fontweight="bold", ha="right")
                            fig.text(0.88, 0.935, subB, color=COL_B, fontsize=SUB_FS, ha="right")
                            fig.text(0.88, 0.915, minsB, color="#374151", fontsize=10, ha="right")
                            return fig

                        fig_r = draw_radar(
                            labels, A_r, B_r, axis_ticks,
                            headerA=pA, subA=f"{rowA['Team']} — {rowA['League']}",
                            headerB=pB, subB=f"{rowB['Team']} — {rowB['League']}",
                        )
                        st.caption(
                            "Percentiles are computed against the **combined A∪B league pool**, "
                            "filtered by your universal **position_filter** (e.g., LB/LWB/RWB/RB families)."
                        )
                        st.pyplot(fig_r, use_container_width=True)
# ----------------- END Radar -----------------








# ----------------- (C) SIMILAR PLAYERS (adjustable pool — FIXED PRESET UI) -----------------
st.markdown("---")
st.header("🧭 Similar players (within adjustable pool)")

# --- Feature basket declared FIRST so UI can use it ---
SIM_FEATURES = [
    'Successful defensive actions per 90','Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
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
    'PAdj Interceptions': 2, 'Aeriel duels won, %': 2, 'Touches in box per 90': 2,
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
    sim_min_minutes, sim_max_minutes = st.slider("Minutes played (candidates)", 0, 5000, (1000, 5000), key="sim_min")
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
    'PAdj Interceptions': 2,'Aeriel duels won, %': 2,'Touches in box per 90': 2,
}

_LS_CF = dict(LEAGUE_STRENGTHS) if 'LEAGUE_STRENGTHS' in globals() else {lg: 50.0 for lg in _included_leagues_cf}
DEFAULT_LEAGUE_WEIGHT = 0.5
DEFAULT_MARKET_WEIGHT = 0.2

CF_FEATURES = [
    'Successful defensive actions per 90','Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
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

