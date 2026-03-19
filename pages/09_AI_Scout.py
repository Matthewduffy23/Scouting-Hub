# 06_AI_Scout.py — AI Scout Assistant
# Original dark card layout for Top 3 · Simple table for candidates 4-10
# New features vs original:
#   - Season detection (WORLDJUNE25 = 2024/25, else 2025/26)
#   - Role scores calculated and shown per candidate
#   - League realism bands (no PL suggestions for League Two players)
#   - Rate-limit safe: sequential calls with small delays
#   - Bio search (Sonnet + web) for top 3 only
#   - Transfermarkt + FMInside for top 3 only

import os
import re
import io
import json
import time
import glob
import unicodedata
import requests

import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from difflib import SequenceMatcher

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AI Scout", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# CSS — original dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
:root{--bg:#0b0f1f;--card:#111827;--stroke:#1f2937;--text:#f1f5f9;--muted:#9fb0c8;--accent:#7c3aed;}
.stApp{background:var(--bg);}
.block-container{max-width:1100px;padding-top:40px;}

.scout-title{font-weight:900;font-size:clamp(30px,4vw,46px);color:var(--text);margin:0;}
.scout-sub{color:var(--muted);margin:4px 0 28px 0;font-size:15px;}
.search-label{color:var(--muted);font-size:13px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px;}

/* club card */
.club-card{background:var(--card);border:1px solid var(--stroke);border-radius:16px;
  padding:20px 24px;margin:0 0 28px 0;}
.club-card h3{color:var(--text);font-size:18px;font-weight:800;margin:0 0 4px 0;}
.club-card .league-badge{color:var(--muted);font-size:13px;margin:0 0 16px 0;}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;}
.stat-box{background:#1a2236;border-radius:10px;padding:10px 14px;}
.stat-box .label{color:var(--muted);font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:.05em;}
.stat-box .value{color:var(--text);font-size:20px;font-weight:800;margin-top:2px;}
.stat-box .rank{font-size:11px;margin-top:2px;}

/* candidate card */
.cand-card{background:var(--card);border:1px solid var(--stroke);border-radius:16px;
  padding:20px 24px;margin-bottom:16px;}
.cand-rank{display:inline-block;background:var(--accent);color:#fff;
  font-weight:900;font-size:13px;padding:3px 10px;border-radius:20px;margin-bottom:10px;}
.cand-name{color:var(--text);font-size:22px;font-weight:900;margin:0;}
.cand-meta{color:var(--muted);font-size:13px;margin:4px 0 14px 0;}
.stat-pills{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;}
.pill{background:#1a2236;border:1px solid var(--stroke);border-radius:8px;
  padding:5px 12px;font-size:12px;color:var(--text);}
.pill .plab{color:var(--muted);font-size:10px;display:block;margin-bottom:1px;}
.pill .pval{font-weight:700;}
.report-text{color:#cbd5e1;font-size:14px;line-height:1.7;
  border-left:3px solid var(--accent);padding-left:14px;margin-top:10px;}

/* role score pills */
.role-row{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 4px 0;}
.role-pill{border-radius:8px;padding:4px 10px;font-size:12px;font-weight:700;
  color:#0b0d12;display:inline-block;}

/* fm badge */
.fm-row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;}
.fm-pill{background:#0f3460;border:1px solid #1e4d8c;border-radius:6px;
  padding:4px 10px;font-size:12px;color:#93c5fd;}
.fm-pill span{font-weight:700;}

/* season tag */
.season-tag{display:inline-block;background:#1a2236;border:1px solid #2d3f5e;
  border-radius:6px;padding:2px 8px;font-size:11px;color:#60a5fa;
  font-weight:700;margin-left:8px;vertical-align:middle;}

/* warning / info */
.warn-box{background:#1c1208;border:1px solid #92400e;border-radius:10px;
  padding:12px 16px;color:#fbbf24;font-size:13px;margin-bottom:16px;}
.info-box{background:#0c1a2e;border:1px solid #1e3a5f;border-radius:10px;
  padding:12px 16px;color:#93c5fd;font-size:13px;margin-bottom:16px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEASON DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_season(filename: str) -> str:
    fn = str(filename).upper()
    if "JUNE25" in fn or "JUN25" in fn:
        return "2024/25"
    return "2025/26"

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
LEAGUE_STRENGTHS = {
    'England 1.':100.00,'Spain 1.':87.84,'Germany 1.':87.45,'Italy 1.':85.88,'France 1.':83.14,
    'England 2.':75.10,'Belgium 1.':74.51,'Brazil 1.':74.31,'Portugal 1.':72.94,'Argentina 1.':71.37,
    'USA 1.':70.00,'Denmark 1.':70.78,'Poland 1.':69.61,'Turkey 1.':69.02,'Netherlands 1.':69.02,
    'Croatia 1.':68.43,'Germany 2.':68.04,'Japan 1.':67.84,'Switzerland 1.':67.45,'Spain 2.':67.06,
    'Norway 1.':66.67,'Mexico 1.':66.47,'Sweden 1.':66.27,'Colombia 1.':65.88,'Czech 1.':65.29,
    'Ecuador 1.':65.29,'Greece 1.':64.12,'Saudi 1.':64.12,'Italy 2.':63.53,'Hungary 1.':63.53,
    'Austria 1.':63.33,'Morocco 1.':63.14,'Korea 1.':62.75,'France 2.':64.00,
    'England 3.':61.96,'Romania 1.':61.76,'Scotland 1.':61.76,'Uruguay 1.':60.39,
    'Chile 1.':59.80,'Israel 1.':58.43,'Brazil 2.':58.04,'Slovenia 1.':57.45,
    'Slovakia 1.':56.47,'Germany 3.':54.51,'Ukraine 1.':54.31,'Portugal 2.':53.14,
    'Serbia 1.':52.16,'Japan 2.':50.98,'England 4.':50.78,'Ireland 1.':50.59,
    'Russia 1.':62.41,'France 3.':49.61,'Belgium 2.':48.43,'Finland 1.':48.43,
    'Switzerland 2.':46.47,'Norway 2.':45.88,'Sweden 2.':45.69,'Turkey 2.':44.51,
    'Czech 2.':43.33,'Netherlands 2.':42.16,'Italy 3.':45.00,'Denmark 2.':40.39,
    'Scotland 2.':38.63,'England 7.':37.25,'Germany 4.':35.29,'Portugal 3.':35.29,
    'England 5.':33.33,'England 9.':31.37,'Northern Ireland 1.':30.98,'Denmark 3.':29.41,
    'Wales 1.':26.67,'Scotland 3.':20.00,'England 6.':16.08,'England 8.':15.69,
    'England 10.':3.92,'Cyprus 1.':60.00,
}

# League band: used for realism checks
def get_league_band(league: str) -> int:
    s = LEAGUE_STRENGTHS.get(str(league).strip(), 40.0)
    if s >= 83:  return 1   # Big 5 (PL, La Liga, Bundesliga, Serie A, Ligue 1)
    if s >= 70:  return 2   # Championship, strong European (Eredivisie, Pro League, Primeira Liga)
    if s >= 60:  return 3   # League One, mid-European
    if s >= 50:  return 4   # League Two, lower European
    if s >= 33:  return 5   # National League, England 5-7
    return 6                # Semi-pro / amateur

BAND_LABELS = {
    1: "Top 5 European",
    2: "Championship / Strong European",
    3: "League One / Mid-European",
    4: "League Two / Lower European",
    5: "National League / Non-League",
    6: "Amateur / Youth",
}

# Realistic next-step: player can move up 1 band max
def realistic_destination_bands(player_league: str) -> list:
    b = get_league_band(player_league)
    return [max(1, b - 1), b]  # one step up or same level

POSITION_METRICS = {
    "CF": ["Non-penalty goals per 90","xG per 90","Shots per 90","Touches in box per 90",
           "Dribbles per 90","Progressive runs per 90","Aerial duels per 90",
           "Aerial duels won, %","Passes per 90","xA per 90"],
    "LW": ["Non-penalty goals per 90","xG per 90","xA per 90","Dribbles per 90",
           "Successful dribbles, %","Crosses per 90","Progressive runs per 90",
           "Key passes per 90","Touches in box per 90","Passes per 90"],
    "RW": ["Non-penalty goals per 90","xG per 90","xA per 90","Dribbles per 90",
           "Successful dribbles, %","Crosses per 90","Progressive runs per 90",
           "Key passes per 90","Touches in box per 90","Passes per 90"],
    "AMF": ["xA per 90","Key passes per 90","Passes to penalty area per 90","xG per 90",
            "Dribbles per 90","Smart passes per 90","Progressive passes per 90",
            "Touches in box per 90","Passes per 90","Accurate passes, %"],
    "CMF": ["Passes per 90","Accurate passes, %","Progressive passes per 90",
            "Progressive runs per 90","xA per 90","Defensive duels per 90",
            "PAdj Interceptions","Touches in box per 90","Dribbles per 90","xG per 90"],
    "DMF": ["Passes per 90","Accurate passes, %","PAdj Interceptions",
            "Defensive duels per 90","Defensive duels won, %","Aerial duels per 90",
            "Aerial duels won, %","Progressive passes per 90","Long passes per 90","Shots blocked per 90"],
    "CB":  ["Aerial duels per 90","Aerial duels won, %","Defensive duels per 90",
            "Defensive duels won, %","PAdj Interceptions","Shots blocked per 90",
            "Passes per 90","Accurate passes, %","Progressive passes per 90","Long passes per 90"],
    "RB":  ["Crosses per 90","xA per 90","Progressive runs per 90","Dribbles per 90",
            "Defensive duels per 90","Aerial duels won, %","Passes per 90",
            "Accurate passes, %","Touches in box per 90","PAdj Interceptions"],
    "LB":  ["Crosses per 90","xA per 90","Progressive runs per 90","Dribbles per 90",
            "Defensive duels per 90","Aerial duels won, %","Passes per 90",
            "Accurate passes, %","Touches in box per 90","PAdj Interceptions"],
    "GK":  ["Save rate, %","Prevented goals per 90","Exits per 90",
            "Accurate long passes, %","Passes per 90"],
}

# Role buckets for scoring
ROLE_BUCKETS = {
    "CB": {
        "Ball Playing CB": {"Passes per 90":2,"Accurate passes, %":2,"Forward passes per 90":2,
            "Accurate forward passes, %":2,"Progressive passes per 90":2,
            "Progressive runs per 90":1.5,"Dribbles per 90":1.5,"Accurate long passes, %":1},
        "Wide CB":         {"Defensive duels per 90":1.5,"Defensive duels won, %":2,
            "Dribbles per 90":2,"Progressive passes per 90":1,"Progressive runs per 90":2},
        "Box Defender":    {"Aerial duels per 90":1,"Aerial duels won, %":3,
            "PAdj Interceptions":2,"Shots blocked per 90":1,"Defensive duels won, %":4},
    },
    "FB": {
        "Build Up FB":   {"Passes per 90":2,"Accurate passes, %":1.5,"Progressive passes per 90":2.5,
            "Progressive runs per 90":2,"Dribbles per 90":2,"xA per 90":1},
        "Attacking FB":  {"Crosses per 90":2,"Dribbles per 90":3.5,"Touches in box per 90":2,
            "Progressive runs per 90":3,"xA per 90":3},
        "Defensive FB":  {"Aerial duels won, %":1.5,"Defensive duels per 90":2,
            "PAdj Interceptions":3,"Defensive duels won, %":3.5},
    },
    "CM": {
        "Deep Playmaker":     {"Passes per 90":1,"Accurate passes, %":1,"Progressive passes per 90":3,
            "Passes to final third per 90":2.5,"Accurate long passes, %":1},
        "Advanced Playmaker": {"xA per 90":4,"Key passes per 90":1,"Smart passes per 90":2,
            "Passes to penalty area per 90":2},
        "Defensive CM":       {"Defensive duels per 90":4,"Defensive duels won, %":4,
            "PAdj Interceptions":3,"Aerial duels won, %":1},
        "Ball Carrying CM":   {"Dribbles per 90":4,"Successful dribbles, %":2,
            "Progressive runs per 90":3,"Accelerations per 90":3},
    },
    "ATT": {
        "Goal Threat":   {"xG per 90":3,"Non-penalty goals per 90":3,"Shots per 90":2,
            "Touches in box per 90":2},
        "Playmaker ATT": {"xA per 90":3,"Key passes per 90":1,"Passes to penalty area per 90":2},
        "Ball Carrier":  {"Dribbles per 90":4,"Successful dribbles, %":2,"Progressive runs per 90":3},
    },
    "CF": {
        "Target Man":    {"Aerial duels per 90":3,"Aerial duels won, %":5},
        "Goal Threat CF":{"Non-penalty goals per 90":3,"Shots per 90":1.5,"xG per 90":3,
            "Touches in box per 90":1},
        "Link Up CF":    {"Passes per 90":2,"xA per 90":3,"Dribbles per 90":2,
            "Progressive runs per 90":2,"Accurate passes, %":1.5},
    },
    "GK": {
        "Shot Stopper":   {"Prevented goals per 90":3,"Save rate, %":1},
        "Ball Playing GK":{"Passes per 90":1,"Accurate passes, %":3,"Accurate long passes, %":2},
        "Sweeper GK":     {"Exits per 90":1},
    },
}

POS_TO_ROLE_KEY = {
    "GK":"GK","CB":"CB","LCB":"CB","RCB":"CB",
    "LB":"FB","RB":"FB","LWB":"FB","RWB":"FB",
    "DMF":"CM","LDMF":"CM","RDMF":"CM","LCMF":"CM","RCMF":"CM",
    "AMF":"ATT","LAMF":"ATT","RAMF":"ATT","LW":"ATT","LWF":"ATT","RW":"ATT","RWF":"ATT",
    "CF":"CF",
}

# Equivalence groups — searching for any member expands to the full group
# LW, LAMF, LWF are functionally the same attacking role
POSITION_EQUIVALENCE = {
    "LW":   ["LW","LAMF","LWF"],
    "LAMF": ["LW","LAMF","LWF"],
    "LWF":  ["LW","LAMF","LWF"],
    "RW":   ["RW","RAMF","RWF"],
    "RAMF": ["RW","RAMF","RWF"],
    "RWF":  ["RW","RAMF","RWF"],
    "AMF":  ["AMF","LAMF","RAMF"],
    "CMF":  ["LCMF","RCMF","CMF"],
    "LCMF": ["LCMF","RCMF","CMF"],
    "RCMF": ["LCMF","RCMF","CMF"],
    "DMF":  ["DMF","LDMF","RDMF"],
    "LDMF": ["DMF","LDMF","RDMF"],
    "RDMF": ["DMF","LDMF","RDMF"],
    "CB":   ["CB","LCB","RCB"],
    "LCB":  ["CB","LCB","RCB"],
    "RCB":  ["CB","LCB","RCB"],
    "LB":   ["LB","LWB"],
    "LWB":  ["LB","LWB"],
    "RB":   ["RB","RWB"],
    "RWB":  ["RB","RWB"],
}

def expand_positions(wanted: list) -> list:
    """Expand each requested position to its equivalence group."""
    expanded = []
    for p in wanted:
        expanded.extend(POSITION_EQUIVALENCE.get(p.upper(), [p.upper()]))
    return list(dict.fromkeys(expanded))  # deduplicate, preserve order

def get_role_key(position: str) -> str:
    tok = str(position).split(",")[0].strip().upper()
    return POS_TO_ROLE_KEY.get(tok, "ATT")

def role_score_color(v: float) -> str:
    if v >= 85: return "#2E6114"
    if v >= 75: return "#5C9E2E"
    if v >= 66: return "#7FBC41"
    if v >= 55: return "#A7D763"
    if v >= 41: return "#F6D645"
    if v >= 25: return "#D77A2E"
    return "#C63733"

def compute_role_scores(player_row: pd.Series, pool_df: pd.DataFrame, position: str) -> dict:
    rk = get_role_key(position)
    roles = ROLE_BUCKETS.get(rk, {})
    scores = {}
    for role_name, metrics in roles.items():
        total_w, wsum = 0.0, 0.0
        for met, w in metrics.items():
            if met not in pool_df.columns:
                continue
            pool_vals = pd.to_numeric(pool_df[met], errors="coerce").dropna()
            player_val = pd.to_numeric(player_row.get(met, np.nan), errors="coerce")
            if pd.isna(player_val) or pool_vals.empty:
                continue
            pct = float((pool_vals < player_val).mean() * 100 + (pool_vals == player_val).mean() * 50)
            wsum += pct * w
            total_w += w
        if total_w > 0:
            scores[role_name] = round(wsum / total_w, 1)
    return scores

def render_role_pills(role_scores: dict) -> str:
    if not role_scores:
        return ""
    pills = []
    for role, score in sorted(role_scores.items(), key=lambda x: -x[1]):
        color = role_score_color(score)
        fg = "#000" if score >= 54 else "#fff"
        pills.append(
            f"<div class='role-pill' style='background:{color};color:{fg};'>"
            f"{role}: {score:.0f}</div>"
        )
    return "<div class='role-row'>" + "".join(pills) + "</div>"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_mv(v) -> str:
    try:
        v = float(v)
        if v >= 1_000_000:
            return f"£{v/1_000_000:.1f}m"
        if v >= 1_000:
            return f"£{int(v/1_000)}k"
        return f"£{int(v):,}"
    except Exception:
        return "—"

def _slug(s: str) -> str:
    s = str(s).strip().lower()
    repl = {"ø":"o","œ":"oe","æ":"ae","å":"a","ä":"a","ö":"o","ü":"u",
            "ß":"ss","ł":"l","đ":"d","ç":"c","ş":"s","ğ":"g","ı":"i"}
    for k, v in repl.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", "", s).strip()

def _surname(name: str) -> str:
    p = name.strip()
    if "," in p:
        return p.split(",")[0].strip()
    parts = p.split()
    return parts[-1] if parts else p

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def describe_ppda(ppda: float) -> str:
    if ppda < 7:   return "Very High Press"
    if ppda < 9:   return "High Press"
    if ppda < 11:  return "Moderate Press"
    if ppda < 14:  return "Low Block"
    return "Deep Block"

def describe_possession(poss: float) -> str:
    if poss >= 58: return "Dominant Possession"
    if poss >= 53: return "Possession-Based"
    if poss >= 47: return "Balanced"
    if poss >= 42: return "Transitional"
    return "Direct / Counter"

def describe_directness(long_p90: float) -> str:
    if long_p90 >= 55: return "Very Direct"
    if long_p90 >= 45: return "Direct"
    if long_p90 >= 35: return "Mixed"
    return "Short / Build-Up"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_csv_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_season"] = detect_season(Path(path).name)
    return df

@st.cache_data(show_spinner=False)
def load_csv_bytes(data: bytes, filename: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    df["_season"] = detect_season(filename)
    return df

@st.cache_data(show_spinner=False)
def load_team_stats(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_and_merge(csv_paths: tuple, upload_bytes: tuple) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        try:
            frames.append(load_csv_path(path))
        except Exception:
            pass
    try:
        names, data_list = upload_bytes
        if names and data_list:
            for name, data in zip(names, data_list):
                try:
                    frames.append(load_csv_bytes(data, name))
                except Exception:
                    pass
    except Exception:
        pass
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "Player" in merged.columns and "Team" in merged.columns:
        merged["Minutes played"] = pd.to_numeric(merged.get("Minutes played", 0), errors="coerce").fillna(0)
        merged = merged.sort_values("Minutes played", ascending=False)
        merged = merged.drop_duplicates(subset=["Player","Team"], keep="first")
    return merged.reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# TEAM PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def build_team_profile(team_name: str, team_df: pd.DataFrame):
    if team_df is None or team_df.empty:
        return None
    mask = team_df["Team"].astype(str).str.lower() == team_name.lower().strip()
    if not mask.any():
        mask = team_df["Team"].astype(str).str.lower().str.contains(
            team_name.lower().strip()[:8], na=False)
    if not mask.any():
        return None
    row = team_df[mask].iloc[0]
    def safe(col, default=0.0):
        try: return float(row[col])
        except: return default
    profile = {
        "team": str(row.get("Team", team_name)),
        "league": str(row.get("League", "—")),
        "matches": int(safe("Matches")),
        "wins": int(safe("Wins")),
        "draws": int(safe("Draws")),
        "losses": int(safe("Losses")),
        "points": int(safe("Points")),
        "xpoints": round(safe("Expected Points"), 1),
        "goals_for": int(safe("Goals For")),
        "goals_against": int(safe("Goals Against")),
        "avg_age": round(safe("Avg Age"), 1),
        "possession": round(safe("Possession %"), 1),
        "ppda": round(safe("PPDA"), 2),
        "xg_p90": round(safe("xG p90"), 2),
        "xga_p90": round(safe("xG Against p90"), 2),
        "passes_p90": round(safe("Passes p90"), 1),
        "long_passes_p90": round(safe("Long Passes p90"), 1),
        "prog_passes_p90": round(safe("Progressive Passes p90"), 1),
        "prog_runs_p90": round(safe("Progressive Runs p90"), 1),
        "crosses_p90": round(safe("Crosses p90"), 1),
        "aerial_p90": round(safe("Aerial Duels p90"), 1),
        "aerial_won_pct": round(safe("Aerial Duels Won %"), 1),
        "shots_p90": round(safe("Shots p90"), 1),
        "press_style": describe_ppda(safe("PPDA")),
        "poss_style": describe_possession(safe("Possession %")),
        "directness": describe_directness(safe("Long Passes p90")),
    }
    league_teams = team_df[team_df["League"] == row["League"]]
    def pct_rank(col):
        try:
            vals = pd.to_numeric(league_teams[col], errors="coerce").dropna()
            v = float(row[col])
            if col == "PPDA": return int((vals >= v).mean() * 100)
            return int((vals <= v).mean() * 100)
        except: return 50
    profile["ppda_pct"] = pct_rank("PPDA")
    profile["xg_pct"] = pct_rank("xG p90")
    profile["aerial_pct"] = pct_rank("Aerial Duels p90")
    profile["possession_pct"] = pct_rank("Possession %")
    profile["crosses_pct"] = pct_rank("Crosses p90")
    profile["prog_runs_pct"] = pct_rank("Progressive Runs p90")
    return profile

# ══════════════════════════════════════════════════════════════════════════════
# FMINSIDE FETCH
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fminside_player(player_name: str, team_name: str):
    try:
        surname = _slug(_surname(player_name))
        full_slug = _slug(player_name)
        search_url = f"https://fminside.net/players?search={requests.utils.quote(player_name)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(search_url, headers=headers, timeout=8)
        if r.status_code != 200: return None
        links = re.findall(r'href="(/players/\d+/[^"]+)"', r.text)
        if not links: return None
        best_link, best_score = None, 0.0
        for lnk in links[:8]:
            slug_part = lnk.split("/")[-1].replace("-", " ")
            sc = max(_similar(_slug(slug_part), full_slug),
                     _similar(_slug(slug_part.split()[-1] if slug_part.split() else ""), surname))
            if sc > best_score:
                best_score = sc
                best_link = lnk
        if best_score < 0.45 or not best_link: return None
        rp = requests.get(f"https://fminside.net{best_link}", headers=headers, timeout=8)
        if rp.status_code != 200: return None
        html = rp.text
        attrs = {}
        fm_fields = {
            "pace": r'Pace[^<]*<[^>]+>\s*(\d+)',
            "acceleration": r'Acceleration[^<]*<[^>]+>\s*(\d+)',
            "strength": r'Strength[^<]*<[^>]+>\s*(\d+)',
            "jumping_reach": r'Jumping Reach[^<]*<[^>]+>\s*(\d+)',
            "stamina": r'Stamina[^<]*<[^>]+>\s*(\d+)',
            "height": r'(\d{3})\s*cm',
        }
        for attr, pattern in fm_fields.items():
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                attrs[attr] = int(val) if val.isdigit() else val
        attrs["_url"] = f"https://fminside.net{best_link}"
        return attrs if len(attrs) > 2 else None
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFERMARKT FETCH
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_transfermarkt_value(player_name: str, team_name: str):
    try:
        search_url = f"https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query={requests.utils.quote(player_name)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-GB,en;q=0.9",
        }
        r = requests.get(search_url, headers=headers, timeout=10)
        if r.status_code != 200: return None
        player_links = re.findall(
            r'href="(/[^/]+/profil/spieler/\d+)"[^>]*>([^<]+)</a>', r.text)
        if not player_links: return None
        best_link, best_score = None, 0.0
        full_slug = _slug(player_name)
        for lnk, name_text in player_links[:6]:
            sc = _similar(_slug(name_text), full_slug)
            if sc > best_score:
                best_score = sc
                best_link = lnk
        if best_score < 0.45 or not best_link: return None
        rp = requests.get(f"https://www.transfermarkt.co.uk{best_link}", headers=headers, timeout=10)
        if rp.status_code != 200: return None
        phtml = rp.text
        value_eur, value_str = None, "—"
        mv2 = re.search(
            r'class="[^"]*marketValue[^"]*"[^>]*>\s*€\s*([\d,\.]+)\s*(k|m|Th\.?|Mill\.?)',
            phtml, re.IGNORECASE)
        if mv2:
            raw = mv2.group(1).replace(",", ".").strip()
            unit = mv2.group(2).lower()
            try:
                num = float(raw)
                if "m" in unit or "mill" in unit:
                    value_eur = num * 1_000_000
                    value_str = f"€{num:.1f}m"
                else:
                    value_eur = num * 1_000
                    value_str = f"€{int(num)}k"
            except Exception:
                pass
        contract_match = re.search(
            r'Contract expires[^:]*:\s*([A-Za-z]+ \d{4}|\d{2}/\d{2}/\d{4})', phtml, re.IGNORECASE)
        contract = contract_match.group(1).strip() if contract_match else "—"
        return {"value_str": value_str, "value_eur": value_eur,
                "contract": contract, "_url": f"https://www.transfermarkt.co.uk{best_link}"}
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE API WRAPPER — rate-limit safe
# ══════════════════════════════════════════════════════════════════════════════
def claude_call(client, model: str, messages: list, system: str = "",
                max_tokens: int = 400, retry: int = 2) -> str:
    """Wrapper with retry on 429 rate limit."""
    for attempt in range(retry + 1):
        try:
            kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
            if system:
                kwargs["system"] = system
            r = client.messages.create(**kwargs)
            return r.content[0].text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 15 * (attempt + 1)
                st.warning(f"Rate limit hit — waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                return f"[Error: {err[:120]}]"
    return "[Rate limit exceeded — try again in a moment]"

def claude_call_with_search(client, prompt: str, system: str = "",
                             max_tokens: int = 500) -> str:
    """Sonnet + web search for player bio — with retry."""
    for attempt in range(2):
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=system or "Football analyst. Factual only. Be concise.",
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
            )
            text_parts = [b.text for b in r.content if hasattr(b, "text")]
            return " ".join(text_parts).strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                time.sleep(20)
            else:
                return ""
    return ""

# ══════════════════════════════════════════════════════════════════════════════
# REGION → LEAGUE MAPPING
# ══════════════════════════════════════════════════════════════════════════════
REGION_LEAGUES = {
    "europe": [
        "England 1.","England 2.","England 3.","England 4.","England 5.",
        "Spain 1.","Spain 2.","Germany 1.","Germany 2.","Germany 3.",
        "Italy 1.","Italy 2.","France 1.","France 2.","Portugal 1.","Portugal 2.",
        "Netherlands 1.","Belgium 1.","Turkey 1.","Scotland 1.","Russia 1.",
        "Greece 1.","Switzerland 1.","Austria 1.","Denmark 1.","Norway 1.",
        "Sweden 1.","Czech 1.","Poland 1.","Croatia 1.","Serbia 1.",
        "Ukraine 1.","Hungary 1.","Romania 1.","Slovakia 1.","Slovenia 1.",
        "Bulgaria 1.","Finland 1.","Ireland 1.","Wales 1.","Northern Ireland 1.",
        "Israel 1.","Cyprus 1.","Iceland 1.","Luxembourg 1.",
    ],
    "south america": [
        "Brazil 1.","Brazil 2.","Argentina 1.","Colombia 1.","Chile 1.",
        "Uruguay 1.","Ecuador 1.","Paraguay 1.","Peru 1.","Bolivia 1.","Venezuela 1.",
    ],
    "north america": [
        "USA 1.","USA 2.","Mexico 1.","Canada 1.","Costa Rica 1.",
    ],
    "asia": [
        "Japan 1.","Japan 2.","Korea 1.","Korea 2.","Saudi 1.","China 1.",
        "UAE 1.","Qatar 1.","Uzbekistan 1.",
    ],
    "africa": [
        "Morocco 1.","Egypt 1.","Algeria 1.","Tunisia 1.","Nigeria 1.",
        "South Africa 1.",
    ],
    "top 5": [
        "England 1.","Spain 1.","Germany 1.","Italy 1.","France 1.",
    ],
    "efl": ["England 2.","England 3.","England 4."],
    "championship": ["England 2."],
    "league one": ["England 3."],
    "league two": ["England 4."],
}

def resolve_leagues(raw_leagues, regions, available_in_csv):
    """
    Convert region names and league names to actual Wyscout league strings.
    Returns None (= no filter) if nothing specified.
    """
    if not raw_leagues and not regions:
        return None

    result = set()

    for item in (raw_leagues or []):
        item_lower = item.lower().strip()
        # Check if it's a region keyword
        if item_lower in REGION_LEAGUES:
            result.update(REGION_LEAGUES[item_lower])
        else:
            # Treat as literal league name
            result.add(item)

    for region in (regions or []):
        region_lower = region.lower().strip()
        if region_lower in REGION_LEAGUES:
            result.update(REGION_LEAGUES[region_lower])

    if not result:
        return None

    # Only keep leagues that actually exist in the loaded CSVs
    available_set = set(available_in_csv)
    filtered = [lg for lg in result if lg in available_set]
    return filtered if filtered else None

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_parameters(client, query: str, reference_player_block: str = "") -> dict:
    ref_section = ""
    if reference_player_block:
        ref_section = f"""
REFERENCE PLAYER DATA (found in CSV — use this to understand what style/metrics to search for):
{reference_player_block}
Use the reference player's strongest metrics (80th percentile+) to infer key_style_traits and priority_metrics.
"""
    response = claude_call(
        client,
        model="claude-haiku-4-5-20251001",
        max_tokens=700,
        system=f"""Extract search parameters from a scout query and return ONLY valid JSON.

IMPORTANT LEAGUE RULES:
- For specific leagues use Wyscout format with trailing dot: "England 2.", "Spain 1." etc.
- For regions use these EXACT keywords in the "regions" field: "europe", "south america", "north america", "asia", "top 5", "efl", "championship", "league one", "league two"
- "European leagues" → regions: ["europe"]
- "South America" → regions: ["south america"]
- "Europe or South America" → regions: ["europe", "south america"]
- If no leagues/regions mentioned, set both to null
{ref_section}
Return this exact JSON structure:
{{
  "club": "Leicester City",
  "club_league": "England 2.",
  "position_prefixes": ["RW"],
  "max_age": 23,
  "min_age": null,
  "min_minutes": 500,
  "leagues": null,
  "regions": ["europe", "south america"],
  "max_market_value_m": 5.0,
  "foot": "left",
  "key_style_traits": ["creative","playmaker","dribbler"],
  "physical_traits": ["fast"],
  "priority_metrics": ["xA per 90","Dribbles per 90","Key passes per 90"],
  "fetch_fminside": false,
  "fetch_transfermarkt": false
}}
Return ONLY the JSON, no markdown, no explanation.""",
        messages=[{"role": "user", "content": query}]
    )
    raw = re.sub(r"^```[a-z]*\n?", "", response.strip())
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# FILTER + SCORE
# ══════════════════════════════════════════════════════════════════════════════
def _pos_tokens(pos_str: str) -> list:
    """Split 'LW, LAMF, AMF' into ['LW','LAMF','AMF'] — primary token first."""
    return [t.strip().upper() for t in re.split(r"[,/\s]+", str(pos_str)) if t.strip()]

def _pos_matches(pos_str: str, wanted: list) -> bool:
    """
    Exact token match only — 'LW' will NOT match 'LWB'.
    A player qualifies if ANY of their listed position tokens is in the wanted set.
    """
    player_tokens = _pos_tokens(pos_str)
    return any(tok in wanted for tok in player_tokens)

def find_player_in_csv(name: str, df: pd.DataFrame):
    """
    Fuzzy-match a player name in the dataframe.
    Returns the best-matching row or None.
    """
    if df.empty or "Player" not in df.columns:
        return None
    name_slug = _slug(name)
    best_score, best_idx = 0.0, None
    for idx, row in df.iterrows():
        candidate = _slug(str(row.get("Player", "")))
        score = _similar(name_slug, candidate)
        # Also try surname-only match
        surname_score = _similar(_slug(_surname(name)), _slug(_surname(str(row.get("Player","")))))
        combined = max(score, surname_score * 0.9)
        if combined > best_score:
            best_score = combined
            best_idx = idx
    if best_score >= 0.55 and best_idx is not None:
        return df.loc[best_idx]
    return None

def build_player_profile_block(player_row: pd.Series, full_df: pd.DataFrame,
                                position_metrics: list) -> str:
    """
    Build a compact stat profile string for a named player,
    showing their key metrics with percentile context.
    Used to give Claude context when searching for 'similar to X'.
    """
    lines = [
        f"Player: {player_row.get('Player','?')} | "
        f"Team: {player_row.get('Team','?')} | "
        f"League: {player_row.get('League','?')} | "
        f"Position: {player_row.get('Position','?')} | "
        f"Age: {player_row.get('Age','?')} | "
        f"Foot: {player_row.get('Foot','?')}",
        "Key stats (percentile vs full database):"
    ]
    for m in position_metrics:
        if m not in full_df.columns:
            continue
        val = pd.to_numeric(player_row.get(m, np.nan), errors="coerce")
        if pd.isna(val):
            continue
        pool = pd.to_numeric(full_df[m], errors="coerce").dropna()
        pct = int((pool <= val).mean() * 100) if not pool.empty else 50
        lines.append(f"  {m}: {val:.2f} ({pct}th pct)")
    return "\n".join(lines)


def filter_candidates(df, params):
    pool = df.copy()
    for col in ["Minutes played", "Age", "Market value"]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce")

    # Position — exact token match with equivalence expansion
    wanted_raw = [p.upper().strip() for p in (params.get("position_prefixes") or [])]
    wanted = expand_positions(wanted_raw) if wanted_raw else []
    if wanted:
        pool = pool[pool["Position"].astype(str).apply(
            lambda p: _pos_matches(p, wanted))]

    # Age
    if params.get("max_age"):
        pool = pool[pool["Age"] <= float(params["max_age"])]
    if params.get("min_age"):
        pool = pool[pool["Age"] >= float(params["min_age"])]

    # Minutes
    min_mins = float(params.get("min_minutes") or 500)
    pool = pool[pool["Minutes played"] >= min_mins]

    # Leagues — resolve regions + explicit leagues against what's in the CSV
    available_leagues = pool["League"].dropna().unique().tolist() if "League" in pool.columns else []
    resolved = resolve_leagues(
        params.get("leagues"),
        params.get("regions"),
        available_leagues,
    )
    if resolved:
        pool = pool[pool["League"].isin(resolved)]

    # Foot
    if params.get("foot") and "Foot" in pool.columns:
        pool = pool[pool["Foot"].astype(str).str.lower().str.startswith(
            params["foot"][0].lower(), na=False)]

    # Market value
    if params.get("max_market_value_m") and "Market value" in pool.columns:
        pool = pool[pool["Market value"] <= float(params["max_market_value_m"]) * 1_000_000]

    return pool

def _similarity_score(pool, ref_row, all_metrics, full_pool):
    """Player profile similarity — distance from reference player's stat vector."""
    if ref_row is None:
        return pd.Series(50.0, index=pool.index)

    scores = []
    for _, row in pool.iterrows():
        diffs = []
        for m in all_metrics:
            if m not in full_pool.columns: continue
            pool_vals = pd.to_numeric(full_pool[m], errors="coerce").dropna()
            ref_val   = pd.to_numeric(ref_row.get(m, np.nan), errors="coerce")
            cand_val  = pd.to_numeric(row.get(m, np.nan), errors="coerce")
            if pd.isna(ref_val) or pd.isna(cand_val) or pool_vals.empty: continue
            spread = pool_vals.std()
            if spread < 1e-9: continue
            diffs.append(abs(float(cand_val) - float(ref_val)) / spread)
        if diffs:
            dist = float(np.mean(diffs))
            scores.append(max(0.0, 100.0 - dist * 40.0))
        else:
            scores.append(50.0)
    return pd.Series(scores, index=pool.index)

def _team_style_score(pool, team_profile, all_metrics, weights):
    """Team style fit — upweight metrics that match the club's tactical profile."""
    if not team_profile:
        return pd.Series(50.0, index=pool.index)
    # Already handled by weights boosting in main scorer — return pure weighted percentile
    return pd.Series(50.0, index=pool.index)  # placeholder, blended via weights

def apply_realism_filter(scored, requesting_league, realism_level):
    """
    Penalise or remove candidates from leagues too far above the requesting club.
    Players at Man City / Real Madrid shouldn't top a League Two search.
    """
    if realism_level == "Off" or not requesting_league:
        return scored

    req_strength = LEAGUE_STRENGTHS.get(str(requesting_league).strip(), 50.0)
    req_band     = get_league_band(requesting_league)

    def _pen(row):
        player_league   = str(row.get("League", ""))
        player_strength = LEAGUE_STRENGTHS.get(player_league.strip(), 50.0)
        player_band     = get_league_band(player_league)
        band_gap        = player_band - req_band   # negative = player is ABOVE requesting club

        # How many bands above the requesting club is this player?
        bands_above = max(0, -band_gap)

        if realism_level == "Strict":
            # Must be within ±1 band
            if bands_above > 1:
                return 0.0
        elif realism_level == "Medium":
            # Up to 2 bands above is ok; beyond that heavy penalty
            if bands_above > 2:
                return 0.0
            elif bands_above == 2:
                return 0.5   # 50% penalty
        elif realism_level == "Soft":
            # Gentle decay — each band above adds 20% penalty
            if bands_above > 0:
                return max(0.1, 1.0 - bands_above * 0.2)

        # Also penalise players from elite clubs unless budget allows
        # (proxy: if player_strength > 85 and req_strength < 65 → top-club penalty)
        if player_strength > 85 and req_strength < 65:
            elite_penalty = (player_strength - 85) / 15.0 * 0.4  # up to 40%
            return max(0.1, 1.0 - elite_penalty)

        return 1.0

    multipliers = scored.apply(_pen, axis=1)
    scored = scored.copy()
    scored["_scout_score"] = scored["_scout_score"] * multipliers
    return scored.sort_values("_scout_score", ascending=False)

def score_candidates(pool, params, team_profile, full_pool,
                     scoring_modes=None, reference_player_row=None):
    """
    Multi-mode scorer. scoring_modes = dict with keys:
      top_performers, role_fit, similar_player, team_style
    Each enabled mode contributes equally to the final score unless only one is on.
    """
    if pool.empty:
        return pool

    if scoring_modes is None:
        scoring_modes = {"top_performers": True, "role_fit": True,
                         "similar_player": False, "team_style": False}

    prefixes = [p.upper().strip() for p in (params.get("position_prefixes") or ["CF"])]
    primary_pos = expand_positions(prefixes)[0] if prefixes else "CF"
    role_key = POS_TO_ROLE_KEY.get(primary_pos, "ATT")

    default_metrics = POSITION_METRICS.get(primary_pos,
                      POSITION_METRICS.get(role_key, POSITION_METRICS["CF"]))
    priority = params.get("priority_metrics") or []
    all_metrics = list(dict.fromkeys(priority + default_metrics))
    all_metrics = [m for m in all_metrics if m in full_pool.columns]

    weights = {m: 1.0 for m in all_metrics}

    # ── Role bucket matching from team profile ────────────────────────────────
    matched_role_metrics = {}
    if team_profile and role_key in ROLE_BUCKETS:
        ppda      = team_profile.get("ppda", 10)
        poss      = team_profile.get("possession", 50)
        long_p90  = team_profile.get("long_passes_p90", 40)
        aerial    = team_profile.get("aerial_p90", 40)
        crosses   = team_profile.get("crosses_p90", 20)
        prog_runs = team_profile.get("prog_runs_p90", 20)

        role_fit_scores = {}
        for role_name, role_metrics in ROLE_BUCKETS[role_key].items():
            fit = 0.0
            if ppda < 9 and any("Defensive" in m or "Interception" in m for m in role_metrics):
                fit += 2.0
            if long_p90 > 45 and any("Aerial" in m for m in role_metrics):
                fit += 2.5
            if poss > 53 and any("passes" in m.lower() for m in role_metrics):
                fit += 2.0
            if crosses > 25 and any("Cross" in m or "box" in m.lower() for m in role_metrics):
                fit += 1.5
            if prog_runs > 25 and any("Progressive runs" in m or "Dribble" in m for m in role_metrics):
                fit += 1.5
            role_fit_scores[role_name] = fit

        if role_fit_scores:
            best_role = max(role_fit_scores, key=role_fit_scores.get)
            matched_role_metrics = ROLE_BUCKETS[role_key][best_role]

    # ── Style trait → metric inference ───────────────────────────────────────
    TRAIT_METRIC_MAP = {
        "target man":    {"Aerial duels per 90": 2.5, "Aerial duels won, %": 2.5},
        "aerial":        {"Aerial duels per 90": 2.0, "Aerial duels won, %": 2.0},
        "header":        {"Aerial duels per 90": 2.0, "Aerial duels won, %": 2.5},
        "goalscorer":    {"Non-penalty goals per 90": 2.5, "xG per 90": 2.0, "Shots per 90": 1.5},
        "finisher":      {"Non-penalty goals per 90": 2.5, "xG per 90": 2.0},
        "poacher":       {"Touches in box per 90": 2.0, "Non-penalty goals per 90": 2.5},
        "creative":      {"xA per 90": 2.5, "Key passes per 90": 2.0, "Smart passes per 90": 1.5},
        "playmaker":     {"xA per 90": 2.5, "Progressive passes per 90": 2.0, "Key passes per 90": 1.5},
        "chance creation": {"xA per 90": 2.5, "Key passes per 90": 2.0, "Passes to penalty area per 90": 2.0},
        "assist":        {"xA per 90": 2.5, "Key passes per 90": 1.5},
        "progressive":   {"Progressive passes per 90": 2.0, "Progressive runs per 90": 2.0},
        "vision":        {"Smart passes per 90": 2.0, "Key passes per 90": 2.0, "xA per 90": 2.0},
        "passer":        {"Accurate passes, %": 2.0, "Passes per 90": 1.5, "Progressive passes per 90": 1.5},
        "build-up":      {"Accurate passes, %": 2.0, "Progressive passes per 90": 2.0},
        "dribbler":      {"Dribbles per 90": 2.5, "Successful dribbles, %": 2.0},
        "dribble":       {"Dribbles per 90": 2.5, "Successful dribbles, %": 2.0},
        "carries":       {"Progressive runs per 90": 2.0, "Dribbles per 90": 2.0},
        "ball carrier":  {"Dribbles per 90": 2.0, "Progressive runs per 90": 2.0, "Accelerations per 90": 1.5},
        "wide creator":  {"Crosses per 90": 2.0, "xA per 90": 2.0, "Dribbles per 90": 1.5},
        "crossing":      {"Crosses per 90": 2.5, "Accurate crosses, %": 1.5},
        "pressing":      {"Defensive duels per 90": 2.0, "PAdj Interceptions": 2.0},
        "press":         {"Defensive duels per 90": 2.0, "PAdj Interceptions": 2.0},
        "high press":    {"Defensive duels per 90": 2.0, "PAdj Interceptions": 2.5, "Accelerations per 90": 1.5},
        "defensive":     {"Defensive duels won, %": 2.0, "PAdj Interceptions": 2.0},
        "tackler":       {"Defensive duels per 90": 2.0, "Defensive duels won, %": 2.5},
        "interceptor":   {"PAdj Interceptions": 3.0},
        "box to box":    {"Defensive duels per 90": 1.5, "Progressive runs per 90": 1.5, "xG per 90": 1.5},
        "fast":          {"Progressive runs per 90": 1.5, "Accelerations per 90": 2.0},
        "quick":         {"Progressive runs per 90": 1.5, "Accelerations per 90": 2.0},
        "strong":        {"Aerial duels won, %": 1.5, "Defensive duels won, %": 1.5},
        "link-up":       {"Passes per 90": 1.5, "xA per 90": 2.0, "Touches in box per 90": 1.5},
    }

    style_traits = [s.lower() for s in (params.get("key_style_traits") or [])]
    for trait in style_traits:
        for phrase, metric_boosts in TRAIT_METRIC_MAP.items():
            if phrase in trait or trait in phrase:
                for met, boost in metric_boosts.items():
                    if met in weights:
                        weights[met] = max(weights[met], boost)

    # ── Apply role/team style weights ─────────────────────────────────────────
    role_weights = dict(weights)
    team_weights = dict(weights)

    if scoring_modes.get("role_fit") and matched_role_metrics:
        for met, w in matched_role_metrics.items():
            if met in role_weights:
                role_weights[met] = role_weights[met] * (1.0 + w * 0.3)

    if scoring_modes.get("team_style") and team_profile:
        ppda = team_profile.get("ppda", 10)
        poss = team_profile.get("possession", 50)
        long_p90 = team_profile.get("long_passes_p90", 40)
        aerial = team_profile.get("aerial_p90", 40)
        if ppda < 8:
            for m in ["Defensive duels per 90","PAdj Interceptions","Accelerations per 90"]:
                if m in team_weights: team_weights[m] = team_weights[m] * 2.0
        if poss > 57:
            for m in ["Accurate passes, %","Progressive passes per 90"]:
                if m in team_weights: team_weights[m] = team_weights[m] * 1.8
        if long_p90 > 50 and poss < 47:
            for m in ["Aerial duels per 90","Aerial duels won, %"]:
                if m in team_weights: team_weights[m] = team_weights[m] * 2.2

    # ── Compute per-mode scores ───────────────────────────────────────────────
    scored = pool.copy()

    def _percentile_score(df_scored, w_map):
        acc = np.zeros(len(df_scored))
        tot = 0.0
        for m in all_metrics:
            if m not in full_pool.columns: continue
            vals_full = pd.to_numeric(full_pool[m], errors="coerce").dropna()
            if vals_full.empty: continue
            cand_vals = pd.to_numeric(df_scored[m], errors="coerce")
            pcts = cand_vals.apply(
                lambda v: (vals_full <= v).mean() * 100 if pd.notna(v) else 50.0).values
            w = w_map.get(m, 1.0)
            acc += pcts * w
            tot += w
        return acc / tot if tot > 0 else np.full(len(df_scored), 50.0)

    active_modes = []
    mode_scores  = {}

    if scoring_modes.get("top_performers"):
        mode_scores["top"] = _percentile_score(scored, weights)
        active_modes.append("top")

    if scoring_modes.get("role_fit"):
        mode_scores["role"] = _percentile_score(scored, role_weights)
        active_modes.append("role")

    if scoring_modes.get("team_style") and team_profile:
        mode_scores["team"] = _percentile_score(scored, team_weights)
        active_modes.append("team")

    if scoring_modes.get("similar_player") and reference_player_row is not None:
        mode_scores["player"] = _similarity_score(scored, reference_player_row,
                                                   all_metrics, full_pool).values
        active_modes.append("player")

    # ── Blend active modes equally ────────────────────────────────────────────
    if active_modes:
        final_score = np.mean([mode_scores[m] for m in active_modes], axis=0)
    else:
        final_score = _percentile_score(scored, weights)

    scored["_scout_score"] = final_score
    scored["_matched_role"] = (
        max(matched_role_metrics.keys(), default="")
        if isinstance(matched_role_metrics, dict) and matched_role_metrics else ""
    )
    return scored.sort_values("_scout_score", ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE NARRATIVE CALLS
# ══════════════════════════════════════════════════════════════════════════════
def generate_club_narrative(client, team_profile: dict, query: str) -> str:
    prompt = f"""Write a brief tactical profile of {team_profile['team']} in 3-4 sentences.
Data:
- League: {team_profile['league']}
- Record: {team_profile['wins']}W {team_profile['draws']}D {team_profile['losses']}L ({team_profile['points']} pts, xPts: {team_profile['xpoints']})
- Pressing: PPDA {team_profile['ppda']} ({team_profile['press_style']}) — {team_profile['ppda_pct']}th pct in league
- Possession: {team_profile['possession']}% ({team_profile['poss_style']})
- Directness: {team_profile['long_passes_p90']} long passes p90 ({team_profile['directness']})
- xG p90: {team_profile['xg_p90']} ({team_profile['xg_pct']}th pct), xGA p90: {team_profile['xga_p90']}
- Aerial duels p90: {team_profile['aerial_p90']} — {team_profile['aerial_pct']}th pct
- Scout context: {query}
End with one sentence on what type of player suits their style."""
    return claude_call(client, "claude-haiku-4-5-20251001",
                       [{"role": "user", "content": prompt}], max_tokens=300)

def fetch_player_bio(client, player: str, team: str, league: str,
                     season: str, position: str) -> str:
    """Sonnet + web search for career bio. Capped to 3 sentences."""
    prompt = (f"Find factual career information about {player}, who plays for {team} "
              f"in {league} as a {position} ({season} season). "
              f"Include: nationality, age, career clubs and seasons, international caps if any, "
              f"goals/assists record. Write EXACTLY 3 sentences. No speculation.")
    result = claude_call_with_search(
        client, prompt,
        system="Football analyst. Return exactly 3 sentences of factual career info. No more.",
        max_tokens=350,
    )
    # Hard cap: take first 3 sentences only
    if result:
        sentences = re.split(r'(?<=[.!?])\s+', result.strip())
        return " ".join(sentences[:3])
    return ""

def generate_mini_report(client, player, params, team_profile, full_pool, fm_data, tm_data, bio_context, season):
    prefixes = [p.upper().strip() for p in (params.get("position_prefixes") or ["CF"])]
    primary_pos = expand_positions(prefixes)[0] if prefixes else "CF"
    role_key = POS_TO_ROLE_KEY.get(primary_pos, "ATT")
    metrics = POSITION_METRICS.get(primary_pos,
              POSITION_METRICS.get(role_key, POSITION_METRICS["CF"]))
    metrics = [m for m in metrics if m in full_pool.columns]

    stats_lines = []
    for m in metrics:
        val = pd.to_numeric(player.get(m), errors="coerce")
        if pd.isna(val): continue
        peer_vals = pd.to_numeric(full_pool[m], errors="coerce").dropna()
        pct = int((peer_vals <= val).mean() * 100) if not peer_vals.empty else 50
        stats_lines.append(f"  {m}: {val:.2f} [{pct}th pct]")

    fm_block = "Not available."
    if fm_data:
        parts = [f"{a.replace('_',' ').title()}: {fm_data[a]}/20"
                 for a in ["pace","acceleration","strength","jumping_reach","stamina"]
                 if a in fm_data]
        if "height" in fm_data: parts.append(f"Height: {fm_data['height']}cm")
        fm_block = ", ".join(parts) if parts else "Partial only."

    tm_block = "Not fetched."
    if tm_data:
        tm_block = f"Market Value: {tm_data.get('value_str','—')}, Contract: {tm_data.get('contract','—')}"

    # Build role requirements context from team profile
    role_req_block = ""
    if team_profile and role_key in ROLE_BUCKETS:
        role_lines = []
        for role_name, role_metrics in ROLE_BUCKETS[role_key].items():
            top_mets = sorted(role_metrics.items(), key=lambda x: -x[1])[:3]
            role_lines.append(f"  {role_name}: {', '.join(m for m,_ in top_mets)}")
        role_req_block = f"\nAvailable roles for this position:\n" + "\n".join(role_lines)

    matched_role = str(player.get("_matched_role", ""))

    club_ctx = "No club context."
    if team_profile:
        player_band = get_league_band(str(player.get("League", "")))
        club_band   = get_league_band(team_profile["league"])
        band_gap    = club_band - player_band
        realism_note = ""
        if band_gap < -1:
            realism_note = (f" LEVEL GAP: Player is {BAND_LABELS.get(player_band,'?')} "
                           f"→ club is {BAND_LABELS.get(club_band,'?')} — significant step up, flag this.")
        club_ctx = (f"{team_profile['team']} ({team_profile['league']}) — "
                    f"{team_profile['press_style']}, {team_profile['poss_style']}, "
                    f"{team_profile['directness']}.{realism_note}")
        if matched_role:
            club_ctx += f" Best role fit for this club's style: {matched_role}."

    bio_section = f"\nCareer context: {bio_context}" if bio_context else ""

    prompt = f"""Write a professional scouting report. STRICT RULES:
- EXACTLY 5 sentences. No more, no less.
- Each sentence on its own line.
- Sentence 1: Best statistical standout with exact value and percentile ({season} data).
- Sentence 2: Second key strength or tactical fit to the club's style.
- Sentence 3: Career context if available, otherwise physical/technical profile.
- Sentence 4: ONE genuine risk or weakness — must be backed by a low stat value.
- Sentence 5: SIGN / MONITOR / PASS with one reason. Flag level gap if significant.
No headers, no bullet points, no markdown.

PERCENTILE CONTEXT — USE THIS SCALE:
90th+ = Elite for this level. Exceptional strength.
80-89th = Strong. Clear asset.
70-79th = Above average. Good for level.
50-69th = Average to solid. Functional.
30-49th = Below average. Area of concern.
<30th = Weakness. Flag as risk.
DO NOT describe 79th percentile or above as low or a concern. 79th = above average.
Only flag stats below 40th percentile as genuine weaknesses.

Player: {player.get('Player','—')}
Club: {player.get('Team','—')} ({player.get('League','—')}) — {season}
Age: {player.get('Age','—')} | Position: {player.get('Position','—')} | Foot: {player.get('Foot','—')}
Minutes: {player.get('Minutes played','—')} | Contract: {player.get('Contract expires','—')} | Value: {fmt_mv(player.get('Market value'))}
{bio_section}

Stats (percentile vs full database):
{chr(10).join(stats_lines) if stats_lines else 'Unavailable.'}

FM Physical: {fm_block}
Transfermarkt: {tm_block}
Club context: {club_ctx}
{role_req_block}
Query: {params.get('_raw_query','—')}

Write 5 sentences now, one per line:"""

    return claude_call(client, "claude-sonnet-4-6",
                       [{"role": "user", "content": prompt}], max_tokens=320)

# ══════════════════════════════════════════════════════════════════════════════
# STAT PILLS + FM PILLS
# ══════════════════════════════════════════════════════════════════════════════
def render_stat_pills(player: pd.Series, params: dict, full_pool: pd.DataFrame) -> str:
    prefixes = [p.upper().strip() for p in (params.get("position_prefixes") or ["CF"])]
    primary_pos = prefixes[0] if prefixes else "CF"
    metrics = POSITION_METRICS.get(primary_pos, POSITION_METRICS["CF"])[:6]
    metrics = [m for m in metrics if m in full_pool.columns]
    pills = []
    for m in metrics:
        val = pd.to_numeric(player.get(m), errors="coerce")
        if pd.isna(val): continue
        peer_vals = pd.to_numeric(full_pool[m], errors="coerce").dropna()
        pct = int((peer_vals <= val).mean() * 100) if not peer_vals.empty else 50
        short = (m.replace(" per 90","p90").replace("Non-penalty goals","NP Goals")
                  .replace("Accurate ","").replace(" won, %"," Win%")
                  .replace("PAdj Interceptions","PAdj Int"))
        pills.append(
            f"<div class='pill'><span class='plab'>{short}</span>"
            f"<span class='pval'>{val:.2f} <span style='color:#6b7280;font-size:10px'>({pct}th)</span></span></div>"
        )
    return "".join(pills)

def render_fm_pills(fm_data: dict) -> str:
    if not fm_data: return ""
    parts = []
    for attr in ["pace","acceleration","strength","jumping_reach","stamina"]:
        if attr not in fm_data: continue
        label = attr.replace("_"," ").title()
        val = fm_data[attr]
        color = "#22c55e" if val >= 14 else ("#f59e0b" if val >= 10 else "#ef4444")
        parts.append(f"<div class='fm-pill'>{label}: <span style='color:{color}'>{val}</span>/20</div>")
    if "height" in fm_data:
        parts.append(f"<div class='fm-pill'>Height: <span>{fm_data['height']}cm</span></div>")
    return "<div class='fm-row'>" + "".join(parts) + "</div>" if parts else ""

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤖 AI Scout")
    st.markdown("---")

    api_key_input = st.text_input(
        "Anthropic API Key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-api03-..."
    )
    st.markdown("---")

    csv_candidates = sorted(
        glob.glob(str(Path.cwd() / "WORLD*.csv")) +
        glob.glob(str(Path.cwd().parent / "WORLD*.csv"))
    )
    csv_labels = [Path(p).name for p in csv_candidates]

    if csv_labels:
        selected_csvs = st.multiselect(
            "Player datasets", options=csv_labels,
            default=csv_labels[:1],
            help="WORLDJUNE25.csv = 2024/25 · Other = 2025/26"
        )
    else:
        selected_csvs = []

    uploaded_files = st.file_uploader("Or upload CSV(s)", type=["csv"],
                                       accept_multiple_files=True)
    st.markdown("---")

    team_stat_candidates = sorted(
        glob.glob(str(Path.cwd() / "*team*stats*.csv")) +
        glob.glob(str(Path.cwd() / "*team*.csv")) +
        glob.glob(str(Path.cwd().parent / "*team*stats*.csv"))
    )
    team_stat_labels = [Path(p).name for p in team_stat_candidates]

    if team_stat_labels:
        selected_team_stats = st.selectbox(
            "Team stats dataset",
            options=["— None —"] + team_stat_labels,
            index=1 if team_stat_labels else 0
        )
    else:
        selected_team_stats = "— None —"
        uploaded_team_stats = st.file_uploader("Upload team stats CSV", type=["csv"])

    st.markdown("---")
    st.caption("Enrichment (top 3 only)")
    do_bio = st.checkbox("🔍 Career bio search (web)", value=False,
                          help="Sonnet + web search. ~4p per player. Adds career history.")
    do_tm  = st.checkbox("💰 Transfermarkt value", value=True,
                          help="Live market value and contract info.")
    do_fm  = st.checkbox("⚽ FMInside attributes", value=False,
                          help="Pace, strength, jumping reach etc from FMInside.")

    st.markdown("---")
    st.caption("⚖️ Scoring modes (combine any)")
    mode_top    = st.checkbox("🏆 Top performers",      value=True,
                               help="Pure percentile — best stats for position vs full database.")
    mode_role   = st.checkbox("🎯 Role fit",            value=True,
                               help="Weighted percentile match to the role buckets (Ball Carrying CM etc).")
    mode_player = st.checkbox("👤 Similar player profile", value=False,
                               help="Similarity distance to a reference player's stat vector. Requires a named player in the query (e.g. 'replace Fatawu').")
    mode_team   = st.checkbox("🏟️ Similar team style",  value=False,
                               help="Weights metrics that match the requesting club's tactical profile (PPDA, possession, aerial stats).")

    if not any([mode_top, mode_role, mode_player, mode_team]):
        st.warning("Select at least one scoring mode.")
        mode_top = True

    st.caption("🌍 Realism")
    realism_level = st.select_slider(
        "League & team realism",
        options=["Off", "Soft", "Medium", "Strict"],
        value="Medium",
        help=(
            "Off = no restriction. "
            "Soft = minor penalty for big gaps. "
            "Medium = realistic step-up (1-2 bands). "
            "Strict = same band only."
        )
    )
    st.markdown("---")
    st.caption("Results")
    top_n = st.slider("Total candidates to return", 5, 15, 10,
                       help="Top 3 get full reports. Rest are listed only.")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
csv_paths_to_load = tuple(
    str(Path.cwd() / name) for name in selected_csvs
    if (Path.cwd() / name).exists()
)
upload_names = tuple(f.name for f in (uploaded_files or []))
upload_data  = tuple(f.getvalue() for f in (uploaded_files or []))

player_df = load_and_merge(csv_paths_to_load, (upload_names, upload_data))

team_df = None
if selected_team_stats and selected_team_stats != "— None —":
    ts_path = Path.cwd() / selected_team_stats
    if ts_path.exists():
        team_df = load_team_stats(str(ts_path))
elif 'uploaded_team_stats' in dir() and uploaded_team_stats is not None:
    team_df = pd.read_csv(io.BytesIO(uploaded_team_stats.getvalue()))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="scout-title">🤖 AI Scout</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="scout-sub">Describe what you need in plain English. '
    'Top 3 get full reports with role scores, career context and FM data. '
    'Remaining candidates are listed for manual review.</div>',
    unsafe_allow_html=True
)

if player_df.empty:
    st.markdown("<div class='warn-box'>⚠️ No player data loaded. Select a WORLD*.csv in the sidebar.</div>",
                unsafe_allow_html=True)
elif not api_key_input:
    st.markdown("<div class='warn-box'>⚠️ Enter your Anthropic API key in the sidebar.</div>",
                unsafe_allow_html=True)
else:
    n_players = len(player_df)
    n_leagues = player_df["League"].nunique() if "League" in player_df.columns else 0
    seasons = player_df["_season"].unique().tolist() if "_season" in player_df.columns else []
    season_str = " · ".join(seasons) if seasons else "—"
    tm_status = f"Team stats: {len(team_df)} teams" if team_df is not None else "⚠️ No team stats"
    st.markdown(
        f"<div class='info-box'>✅ {n_players:,} players · {n_leagues} leagues · "
        f"Season(s): {season_str} · {tm_status}</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH BOX
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='search-label'>Scout Request</div>", unsafe_allow_html=True)

query = st.text_area(
    label="scout_query", label_visibility="collapsed",
    placeholder=(
        "e.g. Salford City in League Two have £1m to spend on a striker. "
        "Find me the best U23 CFs in the EFL — a tall target man with high xG. "
        "Fast, athletically strong, Transfermarkt value under £1m."
    ),
    height=100, key="scout_query_input"
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    run = st.button("🔍 Scout", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear"):
        st.session_state.pop("scout_results", None)
        st.rerun()

with st.expander("💡 Example queries"):
    examples = [
        "Brentford need a left-footed creative winger, U25, max £15m. Dribbles well, creates chances. Top 5 European league.",
        "Find a ball-playing CB for a Championship club. Composed on the ball, good in the air, under 28. Budget £5m.",
        "A defensive midfielder for a mid-table Bundesliga side — high pressing, wins duels. Under 26, max €8m.",
        "Salford City, League Two, £1m budget. U23 CF, tall target man, high xG, fast. EFL leagues only.",
        "Brighton need an attacking midfielder — creative, progressive passer, high xA. U23, check FM attributes.",
    ]
    for ex in examples:
        if st.button(f"→ {ex[:80]}...", key=f"ex_{hash(ex)}"):
            st.session_state["scout_query_prefill"] = ex
            st.rerun()

if "scout_query_prefill" in st.session_state and not query:
    query = st.session_state.pop("scout_query_prefill")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCOUT LOGIC
# ══════════════════════════════════════════════════════════════════════════════
if run and query.strip() and not player_df.empty and api_key_input:

    if not ANTHROPIC_AVAILABLE:
        st.error("anthropic package not installed. Run: pip install anthropic")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key_input)

    # Step 1: Detect reference player in query, look them up in CSV
    reference_player_block = ""
    reference_player_row = None

    # Heuristic: look for patterns like "similar to X", "like X", "replace X", "X at Y"
    ref_patterns = [
        r"similar (?:to|player to)\s+([A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+(?:\s+[A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+)?)",
        r"replace(?:ment for)?\s+([A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+(?:\s+[A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+)?)",
        r"like\s+([A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+(?:\s+[A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+)?)",
        r"([A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+(?:\s+[A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+)?)\s+at\s+[A-Z]",
        r"current\s+\w+\s+([A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+(?:\s+[A-Z][a-zA-Záàâäéèêëíìîïóòôöúùûüñçşğ\-]+)?)",
    ]

    if not player_df.empty:
        for pattern in ref_patterns:
            m = re.search(pattern, query)
            if m:
                candidate_name = m.group(1).strip()
                # Skip common words that match pattern but aren't names
                if candidate_name.lower() in ("the","a","an","their","this","that","my","our"):
                    continue
                found = find_player_in_csv(candidate_name, player_df)
                if found is not None:
                    pos = str(found.get("Position","CF"))
                    rk = get_role_key(pos)
                    ref_metrics = POSITION_METRICS.get(
                        pos.split(",")[0].strip().upper(),
                        POSITION_METRICS.get(rk, POSITION_METRICS["CF"])
                    )
                    reference_player_block = build_player_profile_block(found, player_df, ref_metrics)
                    reference_player_row = found
                    break

    # Step 1b: Extract params — pass reference player stats if found
    with st.spinner("🧠 Reading your request..."):
        params = extract_parameters(client, query, reference_player_block)
        params["_raw_query"] = query

    if not params or not params.get("position_prefixes"):
        # Fallback: try once more without reference block
        with st.spinner("🧠 Re-parsing request..."):
            params = extract_parameters(client, query, "")
            params["_raw_query"] = query

    if not params:
        st.error("Couldn't parse the request. Try being more specific about position and league.")
        st.stop()

    # Show reference player card if found
    if reference_player_row is not None:
        ref_name = str(reference_player_row.get("Player","?"))
        ref_team = str(reference_player_row.get("Team","?"))
        ref_league = str(reference_player_row.get("League","?"))
        ref_pos = str(reference_player_row.get("Position","?"))
        ref_season = str(reference_player_row.get("_season","?"))
        st.markdown(f"""
<div class='info-box' style='border-color:#7c3aed;'>
  🔍 <strong>Reference player found:</strong> {ref_name} · {ref_team} · {ref_league} · {ref_pos} · {ref_season}<br>
  <span style='font-size:12px;color:#9fb0c8;'>Stats used to infer search profile — traits and priority metrics derived from their top percentile metrics.</span>
</div>
""", unsafe_allow_html=True)

    # Step 2: Club profile
    team_profile = None
    club_name = params.get("club")
    if club_name and team_df is not None:
        with st.spinner(f"📊 Loading {club_name} tactical profile..."):
            team_profile = build_team_profile(club_name, team_df)

    # Step 3: Club profile card
    if team_profile:
        with st.spinner(f"✍️ Generating {club_name} profile..."):
            club_narrative_raw = generate_club_narrative(client, team_profile, query)

        # Split into sentences, render as short paragraphs, strip any markdown headers
        club_narrative_raw = re.sub(r"^#+\s.*\n?", "", club_narrative_raw, flags=re.MULTILINE)
        club_sentences = re.split(r'(?<=[.!?])\s+', club_narrative_raw.strip())
        club_sentences = [s.strip() for s in club_sentences if s.strip()][:4]
        club_narrative_html = "".join(f"<p style='margin:0 0 6px 0;'>{s}</p>" for s in club_sentences)

        st.markdown(f"""
<div class='club-card'>
  <h3>📋 {team_profile['team']}</h3>
  <div class='league-badge'>{team_profile['league']} ·
    {team_profile['wins']}W {team_profile['draws']}D {team_profile['losses']}L ·
    {team_profile['points']} pts (xPts: {team_profile['xpoints']})</div>
  <div class='stat-grid'>
    <div class='stat-box'>
      <div class='label'>Pressing (PPDA)</div>
      <div class='value'>{team_profile['ppda']}</div>
      <div class='rank' style='color:#{"22c55e" if team_profile["ppda_pct"]>60 else "f59e0b"}'>{team_profile['press_style']}</div>
    </div>
    <div class='stat-box'>
      <div class='label'>Possession</div>
      <div class='value'>{team_profile['possession']}%</div>
      <div class='rank' style='color:#9fb0c8'>{team_profile['poss_style']}</div>
    </div>
    <div class='stat-box'>
      <div class='label'>Directness</div>
      <div class='value'>{team_profile['long_passes_p90']}</div>
      <div class='rank' style='color:#9fb0c8'>Long p90 · {team_profile['directness']}</div>
    </div>
    <div class='stat-box'>
      <div class='label'>xG p90</div>
      <div class='value'>{team_profile['xg_p90']}</div>
      <div class='rank' style='color:#9fb0c8'>{team_profile['xg_pct']}th pct in league</div>
    </div>
    <div class='stat-box'>
      <div class='label'>Aerial Duels p90</div>
      <div class='value'>{team_profile['aerial_p90']}</div>
      <div class='rank' style='color:#9fb0c8'>{team_profile['aerial_pct']}th pct in league</div>
    </div>
    <div class='stat-box'>
      <div class='label'>Avg Squad Age</div>
      <div class='value'>{team_profile['avg_age']}</div>
      <div class='rank' style='color:#9fb0c8'>years old</div>
    </div>
  </div>
  <div class='report-text' style='margin-top:16px'>{club_narrative_html}</div>
</div>
""", unsafe_allow_html=True)

    elif club_name:
        st.markdown(
            f"<div class='warn-box'>⚠️ '{club_name}' not found in team stats. Results are purely statistical.</div>",
            unsafe_allow_html=True)

    # Step 4: Filter + score
    # Ensure params has minimum required structure
    if not isinstance(params, dict):
        params = {}
    params.setdefault("position_prefixes", [])
    params.setdefault("min_minutes", 500)
    params.setdefault("leagues", None)
    params.setdefault("regions", None)

    # Build scoring modes dict from sidebar toggles
    scoring_modes = {
        "top_performers":  mode_top,
        "role_fit":        mode_role,
        "similar_player":  mode_player,
        "team_style":      mode_team,
    }
    active_mode_labels = []
    if mode_top:    active_mode_labels.append("🏆 Top performers")
    if mode_role:   active_mode_labels.append("🎯 Role fit")
    if mode_player: active_mode_labels.append("👤 Similar player")
    if mode_team:   active_mode_labels.append("🏟️ Team style")

    # Warn if similar_player is on but no reference player found
    if mode_player and reference_player_row is None:
        st.markdown(
            "<div class='warn-box'>⚠️ Similar player mode is on but no reference player was found in your CSVs. "
            "Name a player in your query (e.g. 'replace Fatawu') — falling back to other active modes.</div>",
            unsafe_allow_html=True)
        scoring_modes["similar_player"] = False

    with st.spinner("🔎 Filtering and scoring..."):
        filtered = filter_candidates(player_df, params)
        scored = score_candidates(
            filtered, params, team_profile, player_df,
            scoring_modes=scoring_modes,
            reference_player_row=reference_player_row,
        )
        # Apply realism filter
        requesting_league = params.get("requesting_league") or params.get("club_league") or ""
        if not requesting_league and team_profile:
            requesting_league = team_profile.get("league", "")
        scored = apply_realism_filter(scored, requesting_league, realism_level)

    if scored.empty:
        st.warning("No players found. Try relaxing the filters.")
        st.stop()

    top_candidates = scored.head(top_n).copy()
    leagues_str = ", ".join(params.get("leagues") or ["all loaded leagues"])
    wanted_raw = [p.upper() for p in (params.get("position_prefixes") or [])]
    expanded = expand_positions(wanted_raw)
    pos_str = " + ".join(expanded) if expanded else "all positions"
    modes_str = " · ".join(active_mode_labels) if active_mode_labels else "Top performers"
    realism_str = f"Realism: {realism_level}"
    st.markdown(
        f"<div class='info-box'>Found <strong>{len(scored):,}</strong> candidates · "
        f"Positions: <strong>{pos_str}</strong> · "
        f"Leagues: <strong>{leagues_str}</strong> · "
        f"Scoring: {modes_str} · {realism_str} · "
        f"Top 3 get full reports · #{4}–#{len(top_candidates)} listed below</div>",
        unsafe_allow_html=True)

    # ── TOP 3: Full cards with reports ───────────────────────────────────────
    st.markdown("## 🎯 Top 3 Candidates")

    for rank, (_, player) in enumerate(top_candidates.head(3).iterrows(), 1):
        player_name = str(player.get("Player", "Unknown"))
        team_name   = str(player.get("Team", "—"))
        league      = str(player.get("League", "—"))
        age         = player.get("Age", "—")
        pos         = str(player.get("Position", "—"))
        foot        = str(player.get("Foot", "—"))
        minutes     = player.get("Minutes played", "—")
        contract    = str(player.get("Contract expires", "—"))
        mv          = fmt_mv(player.get("Market value"))
        score       = float(player.get("_scout_score", 0))
        season      = str(player.get("_season", "2025/26"))

        # FM fetch — only if sidebar toggle on
        fm_data = None
        if do_fm:
            with st.spinner(f"⚽ Fetching FM data for {player_name}..."):
                fm_data = fetch_fminside_player(player_name, team_name)
            time.sleep(1)

        # Transfermarkt — only if sidebar toggle on
        tm_data = None
        if do_tm:
            with st.spinner(f"💰 Checking Transfermarkt for {player_name}..."):
                tm_data = fetch_transfermarkt_value(player_name, team_name)
            time.sleep(1)

        # Bio search — only if sidebar toggle on
        bio_context = ""
        if do_bio:
            with st.spinner(f"🔍 Searching career data for {player_name}..."):
                bio_context = fetch_player_bio(client, player_name, team_name, league, season, pos)
            time.sleep(2)

        # Role scores
        pos_pool = player_df[
            (player_df["Position"].astype(str).apply(
                lambda p: get_role_key(p) == get_role_key(pos))) &
            (player_df["League"] == league)
        ]
        if len(pos_pool) < 10:
            pos_pool = player_df[
                player_df["Position"].astype(str).apply(
                    lambda p: get_role_key(p) == get_role_key(pos))
            ]
        role_scores = compute_role_scores(player, pos_pool, pos)

        # Mini report
        with st.spinner(f"✍️ Writing report for {player_name}..."):
            mini_report = generate_mini_report(
                client, player, params, team_profile, player_df,
                fm_data, tm_data, bio_context, season
            )
        time.sleep(2)  # gap before next report

        stat_pills_html = render_stat_pills(player, params, player_df)
        fm_pills_html   = render_fm_pills(fm_data) if fm_data else ""
        role_pills_html = render_role_pills(role_scores)

        tm_display = ""
        if tm_data and tm_data.get("value_str") and tm_data["value_str"] != "—":
            tm_display = f" · TM: {tm_data['value_str']}"
            if tm_data.get("contract") and tm_data["contract"] != "—":
                tm_display += f" · Contract: {tm_data['contract']}"

        # Season tag
        season_color = "#2563eb" if season == "2024/25" else "#059669"
        season_tag = f"<span class='season-tag' style='background:#{season_color}1a;border-color:{season_color};color:{season_color};'>{season}</span>"

        bio_html = ""
        if bio_context:
            bio_html = f"<div style='color:#9fb0c8;font-size:13px;margin:8px 0 0 0;font-style:italic;'>{bio_context}</div>"

        mins_str = int(minutes) if str(minutes).replace('.','').isdigit() else minutes

        # Format report: split sentences onto separate lines for readability
        report_html = ""
        if mini_report and not mini_report.startswith("["):
            sentences = re.split(r'(?<=[.!?])\s+', mini_report.strip())
            sentences = [s.strip() for s in sentences if s.strip()][:5]
            report_html = "".join(
                f"<p style='margin:0 0 8px 0;'>{s}</p>" for s in sentences
            )
        else:
            report_html = f"<p>{mini_report}</p>"

        st.markdown(f"""
<div class='cand-card'>
  <div class='cand-rank' title='Style Match Score: weighted percentile of key position metrics vs full database, boosted by your query traits (aerial, dribbling etc). Higher = better fit for what you searched. Does not adjust for league level — check the report for level context.'>#{rank} · Style Match {score:.0f}/100 ℹ</div>
  <div class='cand-name'>{player_name} {season_tag}</div>
  <div class='cand-meta'>
    {team_name} · {league} · {pos} · Age {age} · {foot} foot ·
    {mins_str} mins · CSV MV: {mv}{tm_display}
  </div>
  <div class='stat-pills'>{stat_pills_html}</div>
  {role_pills_html}
  {fm_pills_html}
  {bio_html}
  <div class='report-text'>{report_html}</div>
</div>
""", unsafe_allow_html=True)

    # ── CANDIDATES 4-N: Simple table ─────────────────────────────────────────
    remaining = top_candidates.iloc[3:]
    if not remaining.empty:
        st.markdown(f"---\n## 📋 Further Candidates (#{4}–#{len(top_candidates)})")
        st.caption("Statistical shortlist only — run a new query to profile these individually.")

        rows = []
        for _, player in remaining.iterrows():
            pos_str = str(player.get("Position", ""))
            rk = get_role_key(pos_str)
            q_pool = player_df[
                (player_df["Position"].astype(str).apply(lambda p: get_role_key(p) == rk)) &
                (player_df["League"] == str(player.get("League", "")))
            ]
            if len(q_pool) < 10:
                q_pool = player_df[player_df["Position"].astype(str).apply(lambda p: get_role_key(p) == rk)]
            qs = compute_role_scores(player, q_pool, pos_str)
            br = max(qs, key=qs.get) if qs else "—"
            bs = qs.get(br, 0)
            age_v = player.get("Age", "")
            mins_v = player.get("Minutes played", "")
            mv_v   = player.get("Market value", "")
            rows.append({
                "Player": str(player.get("Player", "")),
                "Team":   str(player.get("Team", "")),
                "League": str(player.get("League", "")),
                "Season": str(player.get("_season", "2025/26")),
                "Pos":    pos_str.split(",")[0].strip().upper(),
                "Age":    int(float(age_v)) if pd.notna(age_v) and str(age_v) not in ("","nan") else "—",
                "Mins":   int(float(mins_v)) if pd.notna(mins_v) and str(mins_v) not in ("","nan") else "—",
                "MV":     fmt_mv(mv_v),
                "Best Role": f"{br} ({bs:.0f})",
                "Scout Score": f"{player.get('_scout_score', 0):.0f}",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Chief Scout Summary ───────────────────────────────────────────────────
    if len(top_candidates) >= 3:
        with st.spinner("📝 Writing chief scout recommendation..."):
            time.sleep(3)  # let rate limit breathe after 3 reports
            top3_summary = "\n".join([
                f"{i+1}. {str(r.get('Player','?'))} ({str(r.get('Team','?'))}, "
                f"{str(r.get('League','?'))}, age {r.get('Age','?')}, {r.get('_season','?')})"
                for i, (_, r) in enumerate(top_candidates.head(3).iterrows())
            ])
            club_ctx = (
                f"{team_profile['team']} — {team_profile['press_style']}, "
                f"{team_profile['poss_style']}, {team_profile['directness']}."
                if team_profile else "No club context."
            )
            summary = claude_call(
                client, "claude-sonnet-4-6",
                [{"role": "user", "content":
                  f"""You are a chief scout presenting to a director of football.
Summarise your recommendation in 3-4 sentences. Name your top pick and why. Mention one value alternative.
Be realistic about player levels — do not oversell.

Scout query: {query}
Club context: {club_ctx}
Top 3:
{top3_summary}

Write the recommendation:"""}],
                max_tokens=350
            )

        st.markdown(f"""
<div class='club-card' style='margin-top:24px;border-color:#7c3aed;'>
  <h3>🏆 Chief Scout Recommendation</h3>
  <div class='report-text' style='border-color:#7c3aed;margin-top:8px'>{summary}</div>
</div>
""", unsafe_allow_html=True)

elif run and not query.strip():
    st.warning("Please enter a scout request above.")
elif run and player_df.empty:
    st.error("No player data loaded.")
elif run and not api_key_input:
    st.error("Enter your Anthropic API key in the sidebar.")

st.markdown("---")
st.markdown(
    "<div style='color:#4b5563;font-size:12px;text-align:center'>"
    "AI Scout · Claude · Wyscout CSV + FMInside + Transfermarkt · Internal use only"
    "</div>", unsafe_allow_html=True
)