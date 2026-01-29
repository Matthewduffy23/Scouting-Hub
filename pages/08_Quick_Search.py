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

# --- session state (required for photos/crests + caching) ---
st.session_state.setdefault("photo_map", {})  # per-player photo override
st.session_state.setdefault("crest_map", {})  # per-team crest override
st.session_state.setdefault("_fotmob_team_squad_cache", {})  # fotmob API cache


import re, unicodedata, requests
from difflib import SequenceMatcher

PLACEHOLDER_IMG = "https://i.redd.it/43axcjdu59nd1.jpeg"

def _fotmob_team_id_from_url(team_url: str) -> str:
    m = re.search(r"/teams/(\d+)/", str(team_url or ""))
    return m.group(1) if m else ""

def _fotmob_crest_url(team_url: str) -> str:
    tid = _fotmob_team_id_from_url(team_url)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png" if tid else ""

def _fotmob_team_squad(team_id: str):
    cache = st.session_state.setdefault("_fotmob_team_squad_cache", {})
    if team_id in cache:
        return cache[team_id] or []

    squad = []
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
        "ø":"o","œ":"oe","æ":"ae","å":"a","ä":"a","ö":"o","ü":"u",
        "ß":"ss","ł":"l","đ":"d","ð":"d","þ":"th","ç":"c","ş":"s",
        "ğ":"g","ı":"i",
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

def resolve_player_photo(player: str, team: str, league: str) -> str:
    # 1) override
    key_id = f"{player}|||{team}|||{league}"
    override = st.session_state.get("photo_map", {}).get(key_id, "")
    if override:
        return override

    # 2) needs team_fotmob_urls.py mapping
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

    # surname match
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

    # full match fallback
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
    # 1) override
    crest_key = f"{team}|||{league}"
    override = st.session_state.get("crest_map", {}).get(crest_key, "")
    if override:
        return override

    # 2) needs team_fotmob_urls.py mapping
    team_url = ""
    try:
        from team_fotmob_urls import FOTMOB_TEAM_URLS
        team_url = (FOTMOB_TEAM_URLS.get(team) or "").strip()
    except Exception:
        team_url = ""

    return _fotmob_crest_url(team_url) if team_url else ""



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




# ============================ CLUB TOOL — ONE-PAGER (FULL) + STYLES / STRENGTHS / WEAKNESSES ============================
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

def _format_value_gbp_from_m(mv_m) -> str:
    """mv_m is 'millions' (e.g., 30.0). Return £30m / £3.25m / £750k."""
    v = pd.to_numeric(mv_m, errors="coerce")
    if not np.isfinite(v):
        return "—"

    pounds = float(v) * 1_000_000.0

    if pounds >= 1_000_000:
        s = f"{pounds/1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"£{s}m"
    if pounds >= 1_000:
        return f"£{pounds/1_000:.0f}k"
    return f"£{int(pounds):,}"


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
    use_mv_filter = st.toggle("Filter by market value", value=False, key="t20_use_mv_filter_v33")

    if use_mv_filter:
        mv_range = st.slider(
            "Market value (£m)",
            0.0, 100.0, (0.0, 100.0),
            0.5,
            key="t20_mv_v33",
        )
    else:
        # If the slider existed before, delete it and hard-rerun so NOTHING can read it this run.
        if "t20_mv_v33" in st.session_state:
            st.session_state.pop("t20_mv_v33", None)
            st.rerun()
        mv_range = None




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

# Market value (£m) display filter (optional; keeps NaNs visible)
if ("_mv_m" in df_disp.columns) and use_mv_filter and (mv_range is not None):
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
    value_txt = _esc(_format_value_gbp_from_m(r.get("_mv_m", np.nan)))



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
    <td><span class="t20-muted">{value_txt}</span></td>
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
        <th>Value</th>
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