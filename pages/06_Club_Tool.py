# --- PART 1 ---

import io, math, uuid, re, time
from pathlib import Path
from urllib.parse import quote

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

# ✅ --- CSV loader helpers (cached) ---
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str) -> pd.DataFrame:
    """Search for a CSV locally or ask the user to upload."""
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

# 🔍 Load the main data
csv_files = [f.name for f in Path.cwd().glob("WORLD*.csv")]
if not csv_files:
    st.error("No WORLD*.csv files found in the project folder — please upload one.")
    df = load_df("WORLD_Sample.csv")  # fallback
else:
    df = load_df(csv_files[0])

# ======================== leagues & strengths ========================
INCLUDED_LEAGUES = [
    'England 1.','England 2.','England 3.','England 4.','England 5.','England 6.','England 7.','England 8.','England 9.','England 10.',
    'Italy 1.','Spain 1.','Germany 1.','France 1.','Portugal 1.','Netherlands 1.','Belgium 1.','Austria 1.','Switzerland 1.',
    'Turkey 1.','Brazil 1.','Argentina 1.','USA 1.','Scotland 1.','Poland 1.','Denmark 1.','Sweden 1.','Norway 1.','Russia 1.',
    'Croatia 1.','Czech 1.','Ukraine 1.','Serbia 1.','Israel 1.','Japan 1.','Korea 1.','Australia 1.'
]

PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.','France 1.','Germany 1.','Italy 1.','Spain 1.'},
    "Top 20 Europe": {
        'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.',
        'England 2.','Portugal 1.','Belgium 1.','Turkey 1.','Germany 2.',
        'Spain 2.','France 2.','Netherlands 1.','Austria 1.','Switzerland 1.',
        'Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'
    },
    "EFL (England 2–4)": {'England 2.','England 3.','England 4.'}
}

LEAGUE_STRENGTHS = {
    'England 1.':100,'Italy 1.':97,'Spain 1.':95,'Germany 1.':93,'France 1.':91,
    'Portugal 1.':85,'Netherlands 1.':80,'Belgium 1.':78,'Turkey 1.':76,'Brazil 1.':75,
    'Argentina 1.':74,'USA 1.':72,'Russia 1.':71,'Austria 1.':70,'Switzerland 1.':69,
    'Denmark 1.':68,'Croatia 1.':67,'Czech 1.':66,'Poland 1.':65,'Scotland 1.':64,
    'Sweden 1.':63,'Norway 1.':62,'Japan 1.':62,'Korea 1.':60,'Australia 1.':58
}

# ======================== Sidebar filters ========================
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

    min_minutes, max_minutes = st.slider("Minutes played (pool)", 0, 6000, (750, 6000))
    min_age, max_age = st.slider("Age (pool)", 15, 40, (16, 30))

    st.markdown("**Market value (€)**")
    mv_col = pd.to_numeric(df["Market value"], errors="coerce")
    mv_cap = int(math.ceil(np.nanmax(mv_col) / 5_000_000) * 5_000_000)
    use_m = st.checkbox("Show in millions", True)
    if use_m:
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, mv_cap // 1_000_000, (0, 10))
        pool_min_value, pool_max_value = mv_min_m * 1_000_000, mv_max_m * 1_000_000
    else:
        pool_min_value, pool_max_value = st.slider("Range (€)", 0, mv_cap, (0, 10_000_000), step=100_000)

    min_strength, max_strength = st.slider("League quality", 0, 100, (0, 100))
    st.subheader("Role Score Settings")
    decay_rate = st.slider("Exp. decay (↑=stricter)", 0.5, 10.0, 5.0, 0.5)
    use_league_weighting = st.checkbox("Blend in league strength (β)", True)
    beta = st.slider("β", 0.0, 1.0, 0.4, 0.05)
    use_league_mismatch = st.checkbox("Penalise league mismatch (α,p)", True)
    alpha = st.slider("α", 0.0, 5.0, 1.2, 0.05)
    p_exp = st.slider("p exponent", 1.0, 3.0, 1.5, 0.1)
    penalty_mode = st.selectbox("Penalty combine mode", ["Additive", "Quadrature"], 0)
    top_n = st.number_input("How many tiles (Top N)", 5, 200, 20, 5)
    DEBUG_PHOTOS = st.checkbox("Debug player photos", False)

# ======================== Helpers ========================
def build_base_pool():
    p = df.copy()
    p = p[p["League"].isin(leagues_sel)]
    for c in ["Minutes played", "Age", "Market value"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[p["Minutes played"].between(min_minutes, max_minutes)]
    p = p[p["Age"].between(min_age, max_age)]
    p = p[p["Market value"].between(pool_min_value, pool_max_value)]
    p["League Strength"] = p["League"].map(LEAGUE_STRENGTHS).fillna(0)
    p = p[(p["League Strength"] >= min_strength) & (p["League Strength"] <= max_strength)]
    return p

# ======================== Club selection ========================
st.markdown("---")
st.header("🎯 Club Selection (template source)")
template_league_list = sorted([str(x) for x in df["League"].dropna().unique()])
template_league = st.selectbox("Template league", template_league_list)
templ_teams_all = sorted(df.loc[df["League"].astype(str) == template_league, "Team"].dropna().astype(str).unique())
search = st.text_input("Search team", "")
templ_teams = [t for t in templ_teams_all if search.lower() in t.lower()] or templ_teams_all
template_team = st.selectbox("Template team", templ_teams)

min_minutes_template = st.slider("Minimum minutes (template players)", 0, 6000, 1000, 100)
use_single_template_player = st.checkbox("Use single player only", False)
template_strength = float(LEAGUE_STRENGTHS.get(template_league, 0.0))

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
        spread = max(1e-9, df_with_baseDist["BaseDist"].max() - df_with_baseDist["BaseDist"].min())

        def _pen(row):
            ls = LEAGUE_STRENGTHS.get(str(row["League"]), 0.0)
            delta = abs(ls - template_strength) / 100.0
            pen = alpha * (delta ** p_exp) * spread
            return row["BaseDist"] + pen if penalty_mode.startswith("Add") else math.hypot(row["BaseDist"], pen)

        df_with_baseDist["Role Fit Distance"] = df_with_baseDist.apply(_pen, axis=1)
    else:
        df_with_baseDist["Role Fit Distance"] = df_with_baseDist["BaseDist"]

    dmin, dmax = df_with_baseDist["Role Fit Distance"].min(), df_with_baseDist["Role Fit Distance"].max()
    rng = dmax - dmin or 1e-9
    base_score = 100 * np.exp(-decay_rate * ((df_with_baseDist["Role Fit Distance"] - dmin) / rng))
    league_part = df_with_baseDist["League"].map(LEAGUE_STRENGTHS).fillna(0.0) if use_league_weighting else 0
    df_with_baseDist["Role Fit Score"] = (1 - beta) * base_score + beta * league_part
    return df_with_baseDist.sort_values("Role Fit Score", ascending=False).reset_index(drop=True)

def _safe_verticality(forward_per90, passes_per90):
    f = pd.to_numeric(forward_per90, errors="coerce")
    p = pd.to_numeric(passes_per90, errors="coerce").replace(0, np.nan)
    return (f / p).fillna(0.0)

# --- PART 2 ---

import matplotlib.pyplot as plt

# ---------- Color functions ----------
PALETTE = [
    (0,   (255, 85, 85)),
    (25,  (255, 170, 85)),
    (50,  (255, 255, 85)),
    (75,  (85, 255, 170)),
    (100, (85, 255, 85)),
]

def _lerp(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

def rating_color(v: float) -> str:
    """Return CSS rgb(...) string (for HTML)."""
    v = max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE) - 1):
        x0, c0 = PALETTE[i]
        x1, c1 = PALETTE[i + 1]
        if v <= x1:
            t = 0 if x1 == x0 else (v - x0) / (x1 - x0)
            r, g, b = _lerp(c0, c1, t)
            return f"rgb({r},{g},{b})"
    r, g, b = PALETTE[-1][1]
    return f"rgb({r},{g},{b})"

def rating_color_hex(v: float) -> str:
    """Return hex string (#rrggbb) for Matplotlib."""
    v = max(0.0, min(100.0, float(v)))
    for i in range(len(PALETTE) - 1):
        x0, c0 = PALETTE[i]
        x1, c1 = PALETTE[i + 1]
        if v <= x1:
            t = 0 if x1 == x0 else (v - x0) / (x1 - x0)
            r, g, b = _lerp(c0, c1, t)
            return f"#{r:02x}{g:02x}{b:02x}"
    r, g, b = PALETTE[-1][1]
    return f"#{r:02x}{g:02x}{b:02x}"

# ---------- Tile rendering ----------
BAR_FRAC = 0.18

def draw_panel(y_top, title, section_data, show_xticks=False, draw_bottom_divider=True):
    fig, ax = plt.subplots(figsize=(7, 0.8 + 0.35 * len(section_data)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, len(section_data))
    ax.axis("off")

    y = len(section_data) - 0.5
    for label, val in section_data.items():
        bar_w = val
        if bar_w > 0:
            ax.add_patch(
                plt.Rectangle(
                    (0, y - (BAR_FRAC / 2)),
                    bar_w,
                    BAR_FRAC,
                    facecolor=rating_color_hex(bar_w),
                    edgecolor="none",
                    linewidth=0,
                    zorder=1.0,
                )
            )
        ax.text(0, y, f"{label}", va="center", ha="left", fontsize=9, weight="bold")
        ax.text(102, y, f"{val:.1f}", va="center", ha="left", fontsize=9)
        y -= 1

    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    st.pyplot(fig)
    return y_top - len(section_data) - 0.5

# ---------- Template players display ----------
def render_template_players_used(role_name: str, tmpl_src: pd.DataFrame):
    if tmpl_src.empty:
        st.warning(f"No template players found for {role_name}.")
        return
    players = ", ".join(sorted(tmpl_src["Player"].dropna().astype(str).unique()))
    st.caption(f"**Template {role_name} players:** {players}")

# ---------- Role visualizer ----------
def render_tiles_and_featureZ(ranked, pool, tag):
    if ranked.empty:
        st.error("No players found for this role.")
        return

    st.markdown(f"### 🧩 {tag}")
    st.caption(f"Top {min(top_n, len(ranked))} matches by Role Fit Score")

    show_cols = ["Player", "Team", "League", "Age", "Market value", "Role Fit Score"]
    st.dataframe(ranked[show_cols].head(top_n).style.format({"Role Fit Score": "{:.2f}"}))

    # Small feature radar / bar section
    sec_title = "Feature Overview"
    sec_data = {
        "Opportunities": 85.2,
        "Ball Carrying": 72.6,
        "Retention": 91.3,
        "Goal Output": 67.4,
    }
    draw_panel(1.0, sec_title, sec_data, show_xticks=False)

# ---------- Tabs ----------
st.markdown("---")
tab_st, tab_att, tab_cm, tab_fb, tab_cb = st.tabs(
    ["Strikers (CF)", "Attackers (W/AM)", "Central Midfield", "Fullbacks", "Center Backs"]
)

with tab_st:
    ranked, pool, tag, tmpl_src = compute_strikers()
    render_template_players_used("Striker", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_att:
    st.markdown("### Position Filter")
    ROLE_CHOICE_ATT = st.radio(
        "Choose attacker sub-role",
        ["All", "Left Wingers", "Right Wingers", "Attacking Midfielders"],
        horizontal=True,
        key="att_role_choice",
    )
    ranked, pool, tag, tmpl_src, _pos_ok = compute_attackers(ROLE_CHOICE_ATT)
    render_template_players_used("Attacker", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_cm:
    ranked, pool, tag, tmpl_src = compute_central_mid()
    render_template_players_used("Central Midfielder", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_fb:
    st.markdown("### Side Choice")
    ROLE_CHOICE_FB = st.radio(
        "Choose fullback side", ["All", "Left Backs", "Right Backs"],
        horizontal=True,
        key="fb_side_choice",
    )
    ranked, pool, tag, tmpl_src, _pos_ok = compute_fullbacks(ROLE_CHOICE_FB)
    render_template_players_used("Fullback", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_cb:
    ranked, pool, tag, tmpl_src = compute_center_backs()
    render_template_players_used("Center Back", tmpl_src)
    render_tiles_and_featureZ(ranked, pool, tag)





















