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
    "CF": ["xG per 90","Non-penalty goals per 90","xA per 90",
           "Dribbles per 90","Passes per 90","Shots per 90",
           "Touches in box per 90","Progressive runs per 90"],
    "LW": ["xG per 90","Non-penalty goals per 90","xA per 90",
           "Dribbles per 90","Passes per 90","Shots per 90",
           "Touches in box per 90","Progressive runs per 90"],
    "RW": ["xG per 90","Non-penalty goals per 90","xA per 90",
           "Dribbles per 90","Passes per 90","Shots per 90",
           "Touches in box per 90","Progressive runs per 90"],
    "AMF": ["xG per 90","Non-penalty goals per 90","xA per 90",
            "Dribbles per 90","Passes per 90","Shots per 90",
            "Touches in box per 90","Progressive passes per 90"],
    "CMF": ["Passes per 90","Accurate passes, %","Progressive passes per 90",
            "Touches in box per 90","Defensive duels per 90",
            "PAdj Interceptions","xA per 90","Dribbles per 90"],
    "DMF": ["Passes per 90","Accurate passes, %","Progressive passes per 90",
            "Touches in box per 90","Defensive duels per 90",
            "PAdj Interceptions","Defensive duels won, %","Aerial duels won, %"],
    "CB":  ["Aerial duels won, %","Defensive duels won, %","PAdj Interceptions",
            "Passes per 90","Progressive passes per 90","Aerial duels per 90",
            "Defensive duels per 90","Accurate passes, %"],
    "RB":  ["xA per 90","Dribbles per 90","Crosses per 90","Passes per 90",
            "Defensive duels per 90","Progressive runs per 90",
            "Accurate passes, %","PAdj Interceptions"],
    "LB":  ["xA per 90","Dribbles per 90","Crosses per 90","Passes per 90",
            "Defensive duels per 90","Progressive runs per 90",
            "Accurate passes, %","PAdj Interceptions"],
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
        "Deep Playmaker CM":     {"Passes per 90":1,"Accurate passes, %":1,"Progressive passes per 90":3,
            "Passes to final third per 90":2.5,"Accurate long passes, %":1},
        "Advanced Playmaker CM": {"xA per 90":4,"Key passes per 90":1,"Smart passes per 90":2,
            "Passes to penalty area per 90":2},
        "Defensive CM":          {"Defensive duels per 90":4,"Defensive duels won, %":4,
            "PAdj Interceptions":3,"Aerial duels won, %":1},
        "Ball Carrying CM":      {"Dribbles per 90":4,"Successful dribbles, %":2,
            "Progressive runs per 90":3,"Accelerations per 90":3},
    },
    "ATT": {
        "Goal Threat ATT":   {"xG per 90":3,"Non-penalty goals per 90":3,"Shots per 90":2,
            "Touches in box per 90":2},
        "Playmaker ATT": {"xA per 90":3,"Key passes per 90":1,"Passes to penalty area per 90":2},
        "Ball Carrier":  {"Dribbles per 90":4,"Successful dribbles, %":2,"Progressive runs per 90":3},
        "Wide Creator":  {"Crosses per 90":2,"xA per 90":2,"Dribbles per 90":2,
            "Progressive runs per 90":2},
    },
    "CF": {
        "Target Man CF":    {"Aerial duels per 90":3,"Aerial duels won, %":5},
        "Goal Threat CF":{"Non-penalty goals per 90":3,"Shots per 90":1.5,"xG per 90":3,
            "Touches in box per 90":1},
        "Link Up CF":    {"Passes per 90":2,"xA per 90":3,"Dribbles per 90":2,
            "Progressive runs per 90":2,"Accurate passes, %":1.5},
    },
    "GK": {
        "Shot Stopper GK":   {"Prevented goals per 90":3,"Save rate, %":1},
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

# Strict position groups — used for SCORING only (grouping peers), NOT for filtering expansion
# Filtering is always strict: CF query = CF players only, RW query = RW players only
POS_SCORE_GROUP = {
    "GK":   ["GK"],
    "CB":   ["CB","LCB","RCB"],
    "LCB":  ["CB","LCB","RCB"],
    "RCB":  ["CB","LCB","RCB"],
    "LB":   ["LB","LWB"],
    "LWB":  ["LB","LWB"],
    "RB":   ["RB","RWB"],
    "RWB":  ["RB","RWB"],
    "DMF":  ["DMF","LDMF","RDMF"],
    "LDMF": ["DMF","LDMF","RDMF"],
    "RDMF": ["DMF","LDMF","RDMF"],
    "CMF":  ["CMF","LCMF","RCMF"],
    "LCMF": ["CMF","LCMF","RCMF"],
    "RCMF": ["CMF","LCMF","RCMF"],
    "AMF":  ["AMF","LAMF","RAMF"],
    "LAMF": ["AMF","LAMF","RAMF"],
    "RAMF": ["AMF","LAMF","RAMF"],
    "LW":   ["LW","LWF","LAMF"],
    "LWF":  ["LW","LWF","LAMF"],
    "RW":   ["RW","RWF","RAMF"],
    "RWF":  ["RW","RWF","RAMF"],
    "CF":   ["CF"],
}

def expand_positions(wanted: list) -> list:
    """
    Expand a position list for DISPLAY and peer-group scoring only.
    For filtering, use the wanted list directly — no expansion.
    """
    expanded = []
    for p in wanted:
        expanded.extend(POS_SCORE_GROUP.get(p.upper(), [p.upper()]))
    return list(dict.fromkeys(expanded))

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
    if not team_name or not team_name.strip():
        return None

    name_clean = team_name.lower().strip()
    teams_lower = team_df["Team"].astype(str).str.lower()

    # 1. Exact match
    mask = teams_lower == name_clean
    # 2. Contains full name
    if not mask.any():
        mask = teams_lower.str.contains(re.escape(name_clean), na=False)
    # 3. Name contains any word ≥4 chars from query (e.g. "Leicester" matches "Leicester City")
    if not mask.any():
        words = [w for w in re.split(r"\s+", name_clean) if len(w) >= 4]
        for w in words:
            m = teams_lower.str.contains(re.escape(w), na=False)
            if m.any():
                mask = m
                break
    # 4. Token overlap score (≥0.5)
    if not mask.any():
        query_tokens = set(re.split(r"\s+", name_clean))
        best_score, best_idx = 0.0, None
        for i, t in enumerate(teams_lower):
            t_tokens = set(re.split(r"\s+", t))
            overlap = len(query_tokens & t_tokens) / max(len(query_tokens | t_tokens), 1)
            if overlap > best_score:
                best_score, best_idx = overlap, i
        if best_score >= 0.4 and best_idx is not None:
            mask = pd.Series(False, index=team_df.index)
            mask.iloc[best_idx] = True

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

POSITION CODES — use Wyscout format exactly:
- Striker / Centre Forward → ["CF"]
- Right Winger / Right Wide Attacker → ["RW"] (do NOT use RCMF or RAMF)
- Left Winger / Left Wide Attacker → ["LW"] (do NOT use LCMF or LAMF)
- Attacking Midfielder (10) → ["AMF"]
- Central Midfielder → ["CMF"] or ["LCMF","RCMF"]
- Defensive Midfielder → ["DMF"]
- Right Back / Right Wing Back → ["RB"]
- Left Back / Left Wing Back → ["LB"]
- Centre Back → ["CB"]
- Goalkeeper → ["GK"]
CRITICAL: "winger", "wide forward", "wide attacker" = RW or LW only — NEVER CMF, RCMF, LCMF
CRITICAL: "attacker" without side = ["RW","LW","AMF"] — NEVER CMF variants

LEAGUE RULES:
- Specific leagues: Wyscout format with trailing dot: "England 2.", "Spain 1." etc.
- Regions (use EXACT keywords in "regions" field): "europe", "south america", "north america", "asia", "top 5", "efl", "championship", "league one", "league two"
- No leagues/regions mentioned → set both to null
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

def _pos_matches_strict(pos_str: str, wanted: list) -> bool:
    """
    Strict primary-position match.
    Only the FIRST position token is checked — e.g. a player listed as 'CF, AMF'
    is a CF and will only appear in CF searches.
    No equivalence expansion — CF stays CF, RW stays RW.
    """
    tokens = _pos_tokens(pos_str)
    if not tokens:
        return False
    return tokens[0] in wanted

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

    # Position — strict primary-position match, no expansion
    wanted_raw = [p.upper().strip() for p in (params.get("position_prefixes") or [])]
    if wanted_raw:
        pool = pool[pool["Position"].astype(str).apply(
            lambda p: _pos_matches_strict(p, wanted_raw))]

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

def _percentile_of_value(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return 0.5
    return float((s <= float(value)).mean())

def _proper_similarity_score(pool, ref_row, sim_metrics, full_df, league_weight=0.2, percentile_weight=0.7):
    """
    Proper similarity — mirrors compute_similarity_from_template from streamlit_app.py.
    StandardScaler + per-league percentile distance + league difficulty adjustment.
    """
    if ref_row is None or pool.empty:
        return pd.Series(50.0, index=pool.index)

    feats = [m for m in sim_metrics if m in full_df.columns]
    if not feats:
        return pd.Series(50.0, index=pool.index)

    # Target values (reference player)
    target_vals = np.array([
        pd.to_numeric(ref_row.get(f, np.nan), errors="coerce") for f in feats
    ], dtype=float)
    if np.all(np.isnan(target_vals)):
        return pd.Series(50.0, index=pool.index)

    # Candidate pool — numeric only, drop rows missing all feats
    cand = pool.copy()
    for f in feats:
        cand[f] = pd.to_numeric(cand[f], errors="coerce")
    cand = cand.dropna(subset=feats, how="all")
    if cand.empty:
        return pd.Series(50.0, index=pool.index)

    # Per-league percentile ranks for candidates
    percl = cand.groupby("League")[feats].rank(pct=True).reindex(cand.index).fillna(0.5).values

    # Target percentile vector (vs target player's own league)
    target_league = str(ref_row.get("League", ""))
    league_block = full_df[full_df["League"].astype(str) == target_league][feats].copy()
    for f in feats:
        league_block[f] = pd.to_numeric(league_block[f], errors="coerce")

    target_pct = np.array([
        _percentile_of_value(league_block[f], target_vals[i])
        for i, f in enumerate(feats)
    ], dtype=float)

    # Standardised actual-value distance
    try:
        from sklearn.preprocessing import StandardScaler as _SS
        scaler = _SS()
        # Fit on full_df subset for this position
        all_vals = full_df[feats].copy()
        for f in feats:
            all_vals[f] = pd.to_numeric(all_vals[f], errors="coerce")
        all_vals = all_vals.dropna(how="all")
        scaler.fit(all_vals.fillna(all_vals.mean()))
        cand_std = scaler.transform(cand[feats].fillna(cand[feats].mean()))
        # Replace NaN targets with pool mean
        target_std = scaler.transform(
            [np.where(np.isnan(target_vals), all_vals.mean().values, target_vals)]
        )
    except Exception:
        cand_std   = cand[feats].fillna(0).values
        target_std = np.array([[np.nanmean(target_vals)] * len(feats)])

    pct_dist = np.linalg.norm(percl - target_pct, axis=1)
    act_dist = np.linalg.norm(cand_std - target_std, axis=1)
    combined = pct_dist * float(percentile_weight) + act_dist * (1.0 - float(percentile_weight))

    arr = combined.ravel()
    rng = np.ptp(arr)
    normed = (arr - arr.min()) / (rng if rng > 1e-9 else 1.0)
    sims = (1.0 - normed) * 100.0

    # League difficulty adjustment (symmetric ratio)
    tgt_ls = float(LEAGUE_STRENGTHS.get(target_league.strip(), 50.0))
    cand_ls = cand["League"].map(LEAGUE_STRENGTHS).fillna(50.0).values
    eps = 1e-6
    ratio = np.minimum(cand_ls, tgt_ls) / (np.maximum(cand_ls, tgt_ls) + eps)
    adj_sims = sims * ((1.0 - league_weight) + league_weight * ratio)

    result = pd.Series(50.0, index=pool.index)
    result.loc[cand.index] = np.clip(adj_sims, 0.0, 100.0)
    return result


def _proper_role_fit_score(pool, role_metrics_dict, full_df, beta=0.0):
    """
    Per-league percentile × role metric weights.
    Each player ranked vs ALL players of the same role_key in their own league
    from the full database — not just vs the filtered candidate pool.
    """
    if not role_metrics_dict or pool.empty:
        return pd.Series(50.0, index=pool.index)

    scored = pool.copy()
    total_w = sum(role_metrics_dict.values()) or 1.0
    wsum = np.zeros(len(scored))

    for met, w in role_metrics_dict.items():
        if met not in full_df.columns:
            continue
        met_pcts = np.full(len(scored), 50.0)
        for lg in scored["League"].dropna().astype(str).unique():
            cand_mask = scored["League"].astype(str) == lg
            # Reference: same league in full_df (all positions — role peers)
            ref_vals = pd.to_numeric(
                full_df.loc[full_df["League"].astype(str) == lg, met],
                errors="coerce"
            ).dropna()
            if ref_vals.empty:
                continue
            cand_vals = pd.to_numeric(scored.loc[cand_mask, met], errors="coerce")
            pcts = cand_vals.apply(
                lambda v: float((ref_vals <= v).mean() * 100) if pd.notna(v) else 50.0
            ).values
            met_pcts[np.where(cand_mask.values)[0]] = pcts
        wsum += met_pcts * w

    player_score = wsum / total_w

    if beta > 0:
        ls = scored["League"].map(LEAGUE_STRENGTHS).fillna(50.0).values
        player_score = (1.0 - beta) * player_score + beta * ls

    return pd.Series(np.clip(player_score, 0.0, 100.0), index=pool.index)


def _proper_team_style_score(pool, team_profile, role_key, full_df):
    """
    Team style similarity using team_profile stats directly (PPDA, possession, etc).

    Maps team-level style dimensions to equivalent player-level composite metrics,
    computes the template club's target percentile for each dimension from the
    team stats CSV (already in team_profile as pct_rank fields), then measures
    how close each candidate's composite metrics are to that target profile.
    """
    if not team_profile or pool.empty:
        return pd.Series(50.0, index=pool.index)

    # ── Build composite dimensions for candidates (player-level) ─────────────
    # These mirror the composites from streamlit_app.py per role
    def _n(df, col):
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    COMPOSITES = {
        "ATT": {
            "Goal Threat":       lambda df: 0.4*_n(df,"xG per 90") + 0.4*_n(df,"Non-penalty goals per 90") + 0.2*_n(df,"Touches in box per 90"),
            "Creativity":        lambda df: 0.65*_n(df,"xA per 90") + 0.35*_n(df,"Passes to penalty area per 90"),
            "Ball Carrying":     lambda df: 0.6*_n(df,"Dribbles per 90") + 0.4*_n(df,"Progressive runs per 90"),
            "Pass Volume":       lambda df: _n(df,"Passes per 90"),
            "Pressing Work":     lambda df: _n(df,"Defensive duels per 90"),
        },
        "CF": {
            "Opportunities":     lambda df: 0.7*_n(df,"Touches in box per 90") + 0.3*_n(df,"xG per 90"),
            "Aerial Requirement":lambda df: _n(df,"Aerial duels per 90") * _n(df,"Aerial duels won, %") / 100.0,
            "Ball Carrying":     lambda df: 0.65*_n(df,"Dribbles per 90") + 0.35*_n(df,"Progressive runs per 90"),
            "Pass Volume":       lambda df: _n(df,"Passes per 90"),
            "Goal Output":       lambda df: _n(df,"Non-penalty goals per 90"),
        },
        "CM": {
            "Pass Verticality":  lambda df: (_n(df,"Forward passes per 90") /
                                              _n(df,"Passes per 90").replace(0, np.nan)).fillna(0.0),
            "Progression":       lambda df: _n(df,"Progressive passes per 90") + _n(df,"Progressive runs per 90"),
            "Defensive Vol":     lambda df: _n(df,"Defensive duels per 90"),
            "Interceptions":     lambda df: _n(df,"PAdj Interceptions"),
            "Pass Volume":       lambda df: _n(df,"Passes per 90"),
        },
        "FB": {
            "Attacking Contrib": lambda df: (0.4*_n(df,"xA per 90") + 0.2*_n(df,"Crosses per 90") +
                                              0.2*_n(df,"Touches in box per 90") + 0.1*_n(df,"Shots per 90") +
                                              0.1*_n(df,"Passes to penalty area per 90")),
            "Defensive Vol":     lambda df: (0.5*_n(df,"Defensive duels per 90") +
                                              0.3*_n(df,"PAdj Interceptions") + 0.2*_n(df,"Aerial duels per 90")),
            "Progression":       lambda df: _n(df,"Progressive passes per 90") + _n(df,"Progressive runs per 90"),
            "Pass Volume":       lambda df: _n(df,"Passes per 90"),
        },
        "CB": {
            "Aerial Volume":     lambda df: _n(df,"Aerial duels per 90"),
            "Defensive Vol":     lambda df: _n(df,"Defensive duels per 90"),
            "Passing Volume":    lambda df: _n(df,"Passes per 90"),
            "Progression":       lambda df: _n(df,"Progressive passes per 90") + _n(df,"Progressive runs per 90"),
        },
    }

    composites = COMPOSITES.get(role_key, COMPOSITES["ATT"])
    cols = list(composites.keys())

    # ── Build composite cols on the full reference pool and candidate pool ─────
    ref_all = full_df.copy()
    for col, fn in composites.items():
        ref_all[col] = fn(ref_all)

    scored = pool.copy()
    for col, fn in composites.items():
        scored[col] = fn(scored)

    template_league = team_profile.get("league", "")

    # ── Build template target vector from team_profile stats directly ──────────
    # Map each composite dimension to the team_profile percentile already computed
    # in build_team_profile (ppda_pct, possession_pct, crosses_pct etc).
    # This avoids the broken player-CSV team lookup.
    COMPOSITE_TO_TEAM_PCT = {
        # ATT / CF
        "Goal Threat":        team_profile.get("xg_pct", 50),
        "Creativity":         team_profile.get("crosses_pct", 50),
        "Ball Carrying":      team_profile.get("prog_runs_pct", 50),
        "Pass Volume":        50 + (team_profile.get("possession_pct", 50) - 50) * 0.6,
        "Pressing Work":      team_profile.get("ppda_pct", 50),
        "Opportunities":      team_profile.get("xg_pct", 50),
        "Aerial Requirement": team_profile.get("aerial_pct", 50),
        "Goal Output":        team_profile.get("xg_pct", 50),
        # CM / FB
        "Pass Verticality":   max(0, 100 - team_profile.get("possession_pct", 50)),
        "Progression":        team_profile.get("prog_runs_pct", 50),
        "Defensive Vol":      team_profile.get("ppda_pct", 50),
        "Interceptions":      team_profile.get("ppda_pct", 50),
        "Attacking Contrib":  team_profile.get("crosses_pct", 50),
        # CB
        "Aerial Volume":      team_profile.get("aerial_pct", 50),
        "Defensive Vol":      team_profile.get("ppda_pct", 50),
        "Passing Volume":     team_profile.get("possession_pct", 50),
        "Attacking Contrib":  team_profile.get("crosses_pct", 50),
    }
    tmpl_pct = {col: float(COMPOSITE_TO_TEAM_PCT.get(col, 50)) for col in cols}

    # ── Candidate percentile vectors per-league ────────────────────────────────
    pct_cols = {col: np.full(len(scored), 50.0) for col in cols}

    for lg in scored["League"].dropna().astype(str).unique():
        idx_mask = (scored["League"].astype(str) == lg).values
        ref_lg = ref_all[ref_all["League"].astype(str) == lg]
        if ref_lg.empty:
            continue
        for col in cols:
            s = pd.to_numeric(ref_lg[col], errors="coerce").dropna()
            if s.empty:
                continue
            cand_col_vals = pd.to_numeric(scored.loc[scored["League"].astype(str) == lg, col], errors="coerce")
            pcts = cand_col_vals.apply(
                lambda v: float((s <= v).mean() * 100) if pd.notna(v) else 50.0
            ).values
            pct_cols[col][idx_mask] = pcts

    # ── Distance → exp-decay score ─────────────────────────────────────────────
    diff_matrix = np.stack([pct_cols[col] - tmpl_pct[col] for col in cols], axis=1)
    distances   = np.linalg.norm(diff_matrix, axis=1)

    d_min, d_max = distances.min(), distances.max()
    rng = d_max - d_min
    if rng < 1e-9:
        base_score = np.full(len(distances), 80.0)   # everyone fits equally well
    else:
        base_score = 100.0 * np.exp(-5.0 * (distances - d_min) / rng)

    return pd.Series(np.clip(base_score, 0.0, 100.0), index=pool.index)


def _safe_num(df, col):
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

def _safe_div_s(df, num_col, den_col):
    n = _safe_num(df, num_col)
    d = _safe_num(df, den_col).replace(0, np.nan)
    return (n / d).fillna(0.0)


def apply_realism_filter(scored, requesting_league, realism_level):
    """
    Continuous league-strength multiplier based on actual strength scores (0–100).

    Two-sided logic:
    - Players from leagues MUCH STRONGER than the requesting club get penalised
      (a League Two club can't realistically sign a La Liga regular)
    - Players from leagues MUCH WEAKER also get penalised unless they are
      genuinely elite performers in their league (handled by their scout score)
    - Sweet spot: similar strength ±15 pts → no penalty
    - An elite player from a weaker league gets a softer penalty than an average
      one (their high scout score already reflects quality)

    Realism levels control how aggressively the curve bends:
      Off    → multiplier = 1.0 always
      Soft   → gentle curve, only extreme gaps penalised
      Medium → moderate curve, 25pt gap = ~70% score
      Strict → sharp curve, 15pt gap = ~70% score
    """
    if realism_level == "Off" or not requesting_league:
        return scored

    if "_scout_score" not in scored.columns:
        return scored

    req_str = float(LEAGUE_STRENGTHS.get(str(requesting_league).strip(), 50.0))

    # Steepness of the penalty curve per realism level
    # Higher k = steeper penalty for the same strength gap
    K = {"Soft": 0.018, "Medium": 0.038, "Strict": 0.065}.get(realism_level, 0.038)

    # How asymmetric: being ABOVE the club is penalised more than being BELOW
    # above_factor > 1 means upward gaps hurt more
    ABOVE_FACTOR = {"Soft": 1.2, "Medium": 1.6, "Strict": 2.2}.get(realism_level, 1.6)

    # Grace zone: gaps within this range get no penalty at all
    GRACE = {"Soft": 20.0, "Medium": 12.0, "Strict": 6.0}.get(realism_level, 12.0)

    def _multiplier(row):
        pl = str(row.get("League", ""))
        pl_str = float(LEAGUE_STRENGTHS.get(pl.strip(), 50.0))
        gap = pl_str - req_str   # positive = player is in STRONGER league

        if abs(gap) <= GRACE:
            return 1.0

        if gap > 0:
            # Player is in a stronger league — apply asymmetric penalty
            effective_gap = (gap - GRACE) * ABOVE_FACTOR
        else:
            # Player is in a weaker league — softer penalty
            effective_gap = abs(gap) - GRACE

        multiplier = np.exp(-K * effective_gap)
        return float(np.clip(multiplier, 0.05, 1.0))

    multipliers = scored.apply(_multiplier, axis=1)
    scored = scored.copy()
    scored["_scout_score"] = scored["_scout_score"] * multipliers.values
    return scored.sort_values("_scout_score", ascending=False)

def score_candidates(pool, params, team_profile, full_pool,
                     scoring_modes=None, reference_player_row=None,
                     forced_role=None):
    """
    Multi-mode scorer.
    Percentiles = per position-group within league (LWs ranked vs LWs in same league).
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

    # ── Role bucket: forced override > team-profile inference ─────────────────
    matched_role_name = ""
    matched_role_metrics = {}

    if forced_role and role_key in ROLE_BUCKETS:
        if forced_role in ROLE_BUCKETS[role_key]:
            matched_role_name    = forced_role
            matched_role_metrics = ROLE_BUCKETS[role_key][forced_role]

    if not matched_role_metrics and team_profile and role_key in ROLE_BUCKETS:
        ppda      = team_profile.get("ppda", 10)
        poss      = team_profile.get("possession", 50)
        long_p90  = team_profile.get("long_passes_p90", 40)
        crosses   = team_profile.get("crosses_p90", 20)
        prog_runs = team_profile.get("prog_runs_p90", 20)
        scores = {}
        for rname, rmetrics in ROLE_BUCKETS[role_key].items():
            fit = 0.0
            if ppda < 9   and any("Defensive" in m or "Interception" in m for m in rmetrics): fit += 2.0
            if long_p90>45 and any("Aerial" in m for m in rmetrics):                          fit += 2.5
            if poss > 53  and any("passes" in m.lower() for m in rmetrics):                   fit += 2.0
            if crosses>25 and any("Cross" in m or "box" in m.lower() for m in rmetrics):      fit += 1.5
            if prog_runs>25 and any("Progressive runs" in m or "Dribble" in m for m in rmetrics): fit += 1.5
            scores[rname] = fit
        if scores:
            matched_role_name    = max(scores, key=scores.get)
            matched_role_metrics = ROLE_BUCKETS[role_key][matched_role_name]

    # ── Style traits → weight boosts ──────────────────────────────────────────
    TRAIT_METRIC_MAP = {
        "target man":  {"Aerial duels per 90":2.5,"Aerial duels won, %":2.5},
        "goal threat": {"Non-penalty goals per 90":3.0,"xG per 90":2.5,"Shots per 90":1.5},
        "goalscorer":  {"Non-penalty goals per 90":2.5,"xG per 90":2.0,"Shots per 90":1.5},
        "creative":    {"xA per 90":2.5,"Key passes per 90":2.0,"Smart passes per 90":1.5},
        "playmaker":   {"xA per 90":2.5,"Progressive passes per 90":2.0,"Key passes per 90":1.5},
        "wide creator":{"Crosses per 90":2.0,"xA per 90":2.0,"Dribbles per 90":1.5},
        "dribbler":    {"Dribbles per 90":2.5,"Successful dribbles, %":2.0},
        "ball carrier":{"Dribbles per 90":2.0,"Progressive runs per 90":2.0,"Accelerations per 90":1.5},
        "pressing":    {"Defensive duels per 90":2.0,"PAdj Interceptions":2.0},
        "high press":  {"Defensive duels per 90":2.0,"PAdj Interceptions":2.5,"Accelerations per 90":1.5},
        "defensive":   {"Defensive duels won, %":2.0,"PAdj Interceptions":2.0},
        "box to box":  {"Defensive duels per 90":1.5,"Progressive runs per 90":1.5,"xG per 90":1.5},
        "link-up":     {"Passes per 90":1.5,"xA per 90":2.0,"Touches in box per 90":1.5},
        "build-up":    {"Accurate passes, %":2.0,"Progressive passes per 90":2.0},
        "fast":        {"Progressive runs per 90":1.5,"Accelerations per 90":2.0},
        "aerial":      {"Aerial duels per 90":2.0,"Aerial duels won, %":2.0},
    }
    for trait in [s.lower() for s in (params.get("key_style_traits") or [])]:
        for phrase, boosts in TRAIT_METRIC_MAP.items():
            if phrase in trait or trait in phrase:
                for met, boost in boosts.items():
                    if met in weights:
                        weights[met] = max(weights[met], boost)

    role_weights = dict(weights)
    if matched_role_metrics:
        for met, w in matched_role_metrics.items():
            if met in role_weights:
                role_weights[met] = role_weights[met] * (1.0 + w * 0.3)

    # ── Per-position-group-in-league percentile ───────────────────────────────
    # Each candidate ranked vs SAME position group in SAME league.
    # Scores are percentile ranks (0-100) so a 90th pct player in Norway 1.
    # scores the same as a 90th pct player in England 2. — fair cross-league comparison.
    # Falls back to full league if position group has <10 players.
    def _percentile_pos_in_league(df_scored, w_map):
        acc  = np.zeros(len(df_scored))
        tot  = 0.0
        idx_arr = np.arange(len(df_scored))

        for m in all_metrics:
            if m not in full_pool.columns: continue
            w = w_map.get(m, 1.0)
            metric_pcts = np.full(len(df_scored), 50.0)

            for lg in df_scored["League"].dropna().astype(str).unique():
                cand_mask = (df_scored["League"].astype(str) == lg)
                # same role group + same league in full pool
                ref_mask = (
                    (full_pool["League"].astype(str) == lg) &
                    (full_pool["Position"].astype(str).apply(
                        lambda p: POS_TO_ROLE_KEY.get(
                            p.split(",")[0].strip().upper(), "ATT") == role_key
                    ))
                )
                ref_vals = pd.to_numeric(full_pool.loc[ref_mask, m], errors="coerce").dropna()
                if len(ref_vals) < 10:   # too few — fall back to full league
                    ref_vals = pd.to_numeric(
                        full_pool.loc[full_pool["League"].astype(str)==lg, m],
                        errors="coerce").dropna()
                if ref_vals.empty:
                    continue

                cand_vals = pd.to_numeric(df_scored.loc[cand_mask, m], errors="coerce")
                pcts = cand_vals.apply(
                    lambda v: float((ref_vals <= v).mean()*100) if pd.notna(v) else 50.0
                ).values
                positions_in_scored = np.where(cand_mask.values)[0]
                for pi, pv in zip(positions_in_scored, pcts):
                    metric_pcts[pi] = pv

            acc += metric_pcts * w
            tot += w

        return acc / tot if tot > 0 else np.full(len(df_scored), 50.0)

    scored = pool.copy()
    active_modes, mode_scores = [], {}

    if scoring_modes.get("top_performers"):
        mode_scores["top"] = _percentile_pos_in_league(scored, weights)
        active_modes.append("top")

    if scoring_modes.get("role_fit"):
        if matched_role_metrics:
            mode_scores["role"] = _proper_role_fit_score(scored, matched_role_metrics, full_pool)
        else:
            mode_scores["role"] = _percentile_pos_in_league(scored, role_weights)
        active_modes.append("role")

    if scoring_modes.get("team_style"):
        if team_profile:
            mode_scores["team"] = _proper_team_style_score(scored, team_profile, role_key, full_pool)
        else:
            # No team stats found — contribute neutral 50 so mode is still blended
            mode_scores["team"] = np.full(len(scored), 50.0)
        active_modes.append("team")

    if scoring_modes.get("similar_player") and reference_player_row is not None:
        mode_scores["player"] = _proper_similarity_score(
            scored, reference_player_row, all_metrics, full_pool).values
        active_modes.append("player")

    final_score = (
        np.mean([mode_scores[m] for m in active_modes], axis=0)
        if active_modes else _percentile_pos_in_league(scored, weights)
    )

    scored["_scout_score"]   = final_score
    scored["_matched_role"]  = matched_role_name
    # Store individual mode scores for display in cards
    scored["_score_top"]    = mode_scores.get("top",    np.full(len(scored), np.nan))
    scored["_score_role"]   = mode_scores.get("role",   np.full(len(scored), np.nan))
    scored["_score_team"]   = mode_scores.get("team",   np.full(len(scored), np.nan))
    scored["_score_player"] = mode_scores.get("player", np.full(len(scored), np.nan))
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

def generate_mini_report(client, player, params, team_profile, full_pool,
                         fm_data, tm_data, bio_context, season,
                         scoring_modes=None):
    prefixes = [p.upper().strip() for p in (params.get("position_prefixes") or ["CF"])]
    primary_pos = expand_positions(prefixes)[0] if prefixes else "CF"
    role_key = POS_TO_ROLE_KEY.get(primary_pos, "ATT")
    metrics = POSITION_METRICS.get(primary_pos,
              POSITION_METRICS.get(role_key, POSITION_METRICS["CF"]))
    metrics = [m for m in metrics if m in full_pool.columns]

    player_league = str(player.get("League", ""))
    matched_role  = str(player.get("_matched_role", ""))

    # ── Stats: per-position-group-in-league percentile ────────────────────────
    stats_lines = []
    for m in metrics:
        val = pd.to_numeric(player.get(m), errors="coerce")
        if pd.isna(val): continue
        # Same role group, same league
        ref_mask = (
            (full_pool["League"].astype(str) == player_league) &
            (full_pool["Position"].astype(str).apply(
                lambda p: POS_TO_ROLE_KEY.get(p.split(",")[0].strip().upper(), "ATT") == role_key
            ))
        )
        peer_vals = pd.to_numeric(full_pool.loc[ref_mask, m], errors="coerce").dropna()
        if len(peer_vals) < 10:  # too few — fall back to full league
            peer_vals = pd.to_numeric(
                full_pool.loc[full_pool["League"].astype(str)==player_league, m],
                errors="coerce").dropna()
        pct = int((peer_vals <= val).mean() * 100) if not peer_vals.empty else 50
        stats_lines.append(f"  {m}: {val:.2f} [{pct}th pct vs {role_key}s in {player_league}]")

    # ── FM / TM blocks ────────────────────────────────────────────────────────
    fm_block = "Not available."
    if fm_data:
        parts = [f"{a.replace('_',' ').title()}: {fm_data[a]}/20"
                 for a in ["pace","acceleration","strength","jumping_reach","stamina"]
                 if a in fm_data]
        if "height" in fm_data: parts.append(f"Height: {fm_data['height']}cm")
        fm_block = ", ".join(parts) if parts else "Not available."

    tm_block = "Not fetched."
    if tm_data:
        tm_block = f"Market Value: {tm_data.get('value_str','—')}, Contract: {tm_data.get('contract','—')}"

    # ── Role requirements: matched role's top metrics vs player's values ──────
    role_req_lines = []
    if matched_role and role_key in ROLE_BUCKETS:
        role_mets = ROLE_BUCKETS[role_key].get(matched_role, {})
        if role_mets:
            role_req_lines.append(f"ROLE: '{matched_role}' — key metrics (weight × player value [pct]):")
            for met, w in sorted(role_mets.items(), key=lambda x: -x[1])[:5]:
                val = pd.to_numeric(player.get(met, np.nan), errors="coerce")
                if pd.isna(val):
                    role_req_lines.append(f"  {met} (×{w:.0f}): no data")
                    continue
                ref_mask = (
                    (full_pool["League"].astype(str) == player_league) &
                    (full_pool["Position"].astype(str).apply(
                        lambda p: POS_TO_ROLE_KEY.get(
                            p.split(",")[0].strip().upper(), "ATT") == role_key
                    ))
                )
                peer_vals = pd.to_numeric(full_pool.loc[ref_mask, met], errors="coerce").dropna()
                if len(peer_vals) < 10:
                    peer_vals = pd.to_numeric(
                        full_pool.loc[full_pool["League"].astype(str)==player_league, met],
                        errors="coerce").dropna()
                pct = int((peer_vals <= val).mean() * 100) if not peer_vals.empty else 50
                role_req_lines.append(f"  {met} (×{w:.0f}): {val:.2f} [{pct}th pct]")
    role_req_block = "\n".join(role_req_lines) if role_req_lines else "Role requirements: not matched."

    # ── Team style comparison ─────────────────────────────────────────────────
    team_style_block = ""
    player_team_style_block = ""
    if team_profile:
        team_style_block = (
            f"REQUESTING CLUB — {team_profile['team']} ({team_profile['league']}):\n"
            f"  PPDA: {team_profile['ppda']} ({team_profile['press_style']}, {team_profile['ppda_pct']}th pct)\n"
            f"  Possession: {team_profile['possession']}% ({team_profile['poss_style']})\n"
            f"  Long passes p90: {team_profile['long_passes_p90']} ({team_profile['directness']})\n"
            f"  Aerial duels p90: {team_profile['aerial_p90']} ({team_profile['aerial_pct']}th pct)\n"
            f"  xG p90: {team_profile['xg_p90']} ({team_profile['xg_pct']}th pct in league)"
        )
        # Try to find player's current club stats for direct comparison
        player_team_name = str(player.get("Team", ""))
        if "Team" in full_pool.columns and "League" in full_pool.columns:
            pt_mask = (
                (full_pool["Team"].astype(str) == player_team_name) &
                (full_pool["League"].astype(str) == player_league)
            )
            if pt_mask.any():
                # Estimate pressing proxy from player stats
                def_duels = pd.to_numeric(
                    full_pool.loc[pt_mask, "Defensive duels per 90"], errors="coerce"
                ).mean() if "Defensive duels per 90" in full_pool.columns else np.nan
                pass_vol = pd.to_numeric(
                    full_pool.loc[pt_mask, "Passes per 90"], errors="coerce"
                ).mean() if "Passes per 90" in full_pool.columns else np.nan
                long_p = pd.to_numeric(
                    full_pool.loc[pt_mask, "Long passes per 90"], errors="coerce"
                ).mean() if "Long passes per 90" in full_pool.columns else np.nan
                if not np.isnan(def_duels) or not np.isnan(pass_vol):
                    player_team_style_block = (
                        f"PLAYER'S CURRENT CLUB — {player_team_name} ({player_league}) estimated profile:\n"
                    )
                    if not np.isnan(def_duels):
                        player_team_style_block += f"  Defensive duels p90 (squad avg): {def_duels:.2f}\n"
                    if not np.isnan(pass_vol):
                        player_team_style_block += f"  Passes p90 (squad avg): {pass_vol:.2f}\n"
                    if not np.isnan(long_p):
                        player_team_style_block += f"  Long passes p90 (squad avg): {long_p:.2f}"

    # ── Level gap ─────────────────────────────────────────────────────────────
    level_note = ""
    if team_profile:
        p_band = get_league_band(player_league)
        c_band = get_league_band(team_profile["league"])
        if c_band - p_band < -1:
            level_note = (f"LEVEL GAP: Player plays in {BAND_LABELS.get(p_band,'?')} "
                         f"— requesting club is {BAND_LABELS.get(c_band,'?')}. "
                         f"Significant step down in level. Flag in report.")

    bio_section = f"Career context: {bio_context}" if bio_context else ""

    # ── Adaptive sentence structure based on active modes ─────────────────────
    sm = scoring_modes or {}
    has_team_style  = bool(sm.get("team_style") and team_profile)
    has_role_fit    = bool(sm.get("role_fit") or matched_role)
    has_similar_pl  = bool(sm.get("similar_player"))

    if has_team_style and has_role_fit:
        s1 = "S1 — TEAM STYLE FIT: Compare the player's press intensity (defensive duels p90), passing volume, and directness to the requesting club's PPDA, possession%, and long passes p90. Use exact numbers from both profiles. If player's club stats are available, note the similarity or difference."
        s2 = "S2 — ROLE FIT: Name the matched role. Cite the top 3 weighted role metrics with the player's actual value and percentile. Be specific about how well they fit."
    elif has_team_style:
        s1 = "S1 — TEAM STYLE FIT: Compare how this player's statistical profile matches the club's tactical style. Use exact numbers."
        s2 = "S2 — STANDOUT STAT: Best single metric with exact value and percentile."
    elif has_role_fit:
        s1 = "S1 — ROLE FIT: Name the matched role. Cite the top 3 weighted role metrics with the player's actual value and percentile."
        s2 = "S2 — SECOND STRENGTH: Second best quality with exact value and percentile."
    else:
        s1 = "S1 — BEST STAT: Single clearest data-backed strength with exact value and percentile."
        s2 = "S2 — SECOND STRENGTH: Second best quality with exact value and percentile."

    prompt = f"""You are Head of Recruitment at a data-driven Championship club (Brentford / Brighton methodology).
Write a scouting report. Experienced, direct, grounded in numbers. No filler.

HARD RULES:
- EXACTLY 5 sentences. One per line. No headers, bullets, or markdown.
- Every claim needs a specific number from the data.
- Percentile ≥70 = above average. NEVER call this a weakness.
- Only flag a weakness if a stat is genuinely <40th percentile.
- If no <40th pct stat exists: describe the tactical adjustment needed for the step up, not a made-up flaw.

SENTENCE STRUCTURE:
{s1}
{s2}
S3 — STANDOUT QUALITY or CAREER CONTEXT: Clearest individual strength with exact value/pct, or career context if bio provided.
S4 — HONEST WEAKNESS or TRANSITION NOTE: Sub-40th pct = weakness with the stat. No weakness = state the one tactical adjustment for this level. Do not fabricate.
S5 — VERDICT: SIGN / MONITOR / PASS. One decisive reason. State level gap plainly if {level_note}.

PERCENTILE SCALE: 90th+=Elite · 80-89=Strong · 70-79=Above avg · 50-69=Functional · <40=Weakness

=== DATA ===
Player: {player.get('Player','—')} | {player.get('Team','—')} ({player_league}) | {season}
Age: {player.get('Age','—')} | Position: {player.get('Position','—')} | Foot: {player.get('Foot','—')}
Minutes: {player.get('Minutes played','—')} | Contract: {player.get('Contract expires','—')} | Value: {fmt_mv(player.get('Market value'))}
{bio_section}

PLAYER STATS (per-position-group in {player_league}):
{chr(10).join(stats_lines) if stats_lines else 'Unavailable.'}

FM Physical: {fm_block}
Transfermarkt: {tm_block}

{team_style_block}
{player_team_style_block}

{role_req_block}

{level_note}
Query: {params.get('_raw_query','—')}
===

Write exactly 5 sentences, one per line:"""

    return claude_call(client, "claude-sonnet-4-6",
                       [{"role": "user", "content": prompt}], max_tokens=420)
# ══════════════════════════════════════════════════════════════════════════════
# STAT PILLS + FM PILLS
# ══════════════════════════════════════════════════════════════════════════════
def render_stat_pills(player: pd.Series, params: dict, full_pool: pd.DataFrame) -> str:
    # Use player's actual primary position — not what was searched
    actual_pos = str(player.get("Position", "")).split(",")[0].strip().upper()
    role_key = POS_TO_ROLE_KEY.get(actual_pos, "ATT")
    primary_pos = actual_pos if actual_pos in POSITION_METRICS else next(
        (p for p in POSITION_METRICS if POS_TO_ROLE_KEY.get(p) == role_key), "CF"
    )
    metrics = POSITION_METRICS.get(primary_pos, POSITION_METRICS["CF"])[:6]
    metrics = [m for m in metrics if m in full_pool.columns]
    player_league = str(player.get("League", ""))
    pills = []
    for m in metrics:
        val = pd.to_numeric(player.get(m), errors="coerce")
        if pd.isna(val): continue
        # Per-position-group in same league
        ref_mask = (
            (full_pool["League"].astype(str) == player_league) &
            (full_pool["Position"].astype(str).apply(
                lambda p: POS_TO_ROLE_KEY.get(p.split(",")[0].strip().upper(), "ATT") == role_key
            ))
        )
        peer_vals = pd.to_numeric(full_pool.loc[ref_mask, m], errors="coerce").dropna()
        if len(peer_vals) < 10:
            peer_vals = pd.to_numeric(
                full_pool.loc[full_pool["League"].astype(str) == player_league, m],
                errors="coerce").dropna()
        pct = int((peer_vals <= val).mean() * 100) if not peer_vals.empty else 50
        col = "#22c55e" if pct >= 80 else ("#f59e0b" if pct >= 50 else "#ef4444")
        short = (m.replace(" per 90","p90").replace("Non-penalty goals","NP Goals")
                  .replace("Accurate ","").replace(" won, %"," Win%")
                  .replace("PAdj Interceptions","PAdj Int"))
        pills.append(
            f"<div class='pill'><span class='plab'>{short}</span>"
            f"<span class='pval' style='color:{col}'>{val:.2f} "
            f"<span style='color:#6b7280;font-size:10px'>({pct}th)</span></span></div>"
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

# ── Filter overrides ──────────────────────────────────────────────────────────
# Position → available style groups
_STYLE_BY_POS_GROUP = {
    "GK":  ["Shot Stopper GK", "Ball Playing GK", "Sweeper GK"],
    "CB":  ["Ball Playing CB", "Wide CB", "Box Defender"],
    "FB":  ["Attacking FB", "Build Up FB", "Defensive FB"],
    "CM":  ["Deep Playmaker CM", "Advanced Playmaker CM", "Defensive CM", "Ball Carrying CM"],
    "ATT": ["Goal Threat ATT", "Playmaker ATT", "Ball Carrier", "Wide Creator"],
    "CF":  ["Target Man CF", "Goal Threat CF", "Link Up CF"],
}
_POS_TO_STYLE_GROUP = {
    "GK":"GK","CB":"CB","LCB":"CB","RCB":"CB",
    "LB":"FB","RB":"FB","LWB":"FB","RWB":"FB",
    "DMF":"CM","LDMF":"CM","RDMF":"CM","LCMF":"CM","RCMF":"CM",
    "AMF":"ATT","LAMF":"ATT","RAMF":"ATT","LW":"ATT","LWF":"ATT","RW":"ATT","RWF":"ATT",
    "CF":"CF",
}

with st.expander("🔧 Filter overrides (optional)", expanded=False):
    st.caption("Tick to enforce. Unticked = Claude uses what it extracts from your query.")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        fo_use_pos = st.checkbox("📍 Position", key="fo_use_pos")
        if fo_use_pos:
            _all_pos = ["GK","CB","LCB","RCB","LB","RB","LWB","RWB",
                        "DMF","LDMF","RDMF","LCMF","RCMF",
                        "AMF","LAMF","RAMF","LW","LWF","RW","RWF","CF"]
            fo_positions = st.multiselect("Position(s)", _all_pos, key="fo_positions")
        else:
            fo_positions = []

        fo_use_age = st.checkbox("🎂 Age", key="fo_use_age")
        if fo_use_age:
            fo_age_min, fo_age_max = st.slider("Age range", 14, 45, (16, 30), key="fo_age")
        else:
            fo_age_min = fo_age_max = None

    with fc2:
        fo_use_region = st.checkbox("🌍 Region", key="fo_use_region")
        if fo_use_region:
            fo_regions = st.multiselect("Region(s)", [
                "europe","south america","north america","asia","africa",
                "top 5","efl","championship","league one","league two",
            ], key="fo_regions")
        else:
            fo_regions = []

        fo_use_band = st.checkbox("🏷️ League Band", key="fo_use_band")
        if fo_use_band:
            fo_bands = st.multiselect("Band(s)", [1,2,3,4,5,6], key="fo_bands",
                help="1=Top5 EU · 2=Champ/Strong EU · 3=L1/Mid EU · 4=L2/Lower EU · 5=NL · 6=Amateur")
        else:
            fo_bands = []

    with fc3:
        fo_use_value = st.checkbox("💰 Market Value (max)", key="fo_use_value")
        if fo_use_value:
            fo_value_max = st.number_input("Max £m", 0.0, 200.0, 5.0, 0.5, key="fo_value")
        else:
            fo_value_max = None

        fo_use_style = st.checkbox("🎯 Style / Role", key="fo_use_style")
        if fo_use_style:
            if fo_positions:
                _groups = list({_POS_TO_STYLE_GROUP.get(p,"ATT") for p in fo_positions})
                _hint = f"Showing styles for: {', '.join(_groups)}"
            else:
                _groups = list(_STYLE_BY_POS_GROUP.keys())
                _hint = "Tick Position above to filter by role group"
            _opts = list(dict.fromkeys(
                s for g in _groups for s in _STYLE_BY_POS_GROUP.get(g, [])
            ))
            st.caption(_hint)
            fo_styles = st.multiselect("Style / Role", _opts, key="fo_styles")
        else:
            fo_styles = []

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

    # ── Apply filter overrides ────────────────────────────────────────────────
    if fo_use_pos and fo_positions:
        params["position_prefixes"] = fo_positions
    if fo_use_age and fo_age_min is not None:
        params["min_age"] = fo_age_min
        params["max_age"] = fo_age_max
    if fo_use_value and fo_value_max is not None:
        params["max_market_value_m"] = fo_value_max
    if fo_use_region and fo_regions:
        params["regions"] = fo_regions
        params["leagues"] = None
    if fo_use_band and fo_bands:
        available_in_df = player_df["League"].dropna().unique().tolist()
        params["leagues"] = [lg for lg in LEAGUE_STRENGTHS
                              if get_league_band(lg) in fo_bands
                              and lg in available_in_df] or None

    # Style override → forced role bucket name for scorer
    fo_override_role = None
    if fo_use_style and fo_styles:
        fo_override_role = fo_styles[0]
        params.setdefault("key_style_traits", [])
        params["key_style_traits"] = list(set(
            params["key_style_traits"] + [s.lower() for s in fo_styles]
        ))

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
            forced_role=fo_override_role,
        )
        # Apply realism filter — derive requesting_league via multiple fallbacks
        requesting_league = (
            params.get("club_league") or
            params.get("requesting_league") or
            ""
        )
        # Fallback 1: team_profile if found
        if not requesting_league and team_profile:
            requesting_league = team_profile.get("league", "")
        # Fallback 2: if leagues filter is a single specific league, use that
        if not requesting_league:
            leagues_param = params.get("leagues") or []
            if len(leagues_param) == 1:
                requesting_league = leagues_param[0]
        # Fallback 3: scan query text for known league strings
        if not requesting_league and query:
            q_lower = query.lower()
            _league_keywords = {
                "premier league": "England 1.", "championship": "England 2.",
                "league one": "England 3.", "league two": "England 4.",
                "england 1": "England 1.", "england 2": "England 2.",
                "england 3": "England 3.", "england 4": "England 4.",
                "la liga": "Spain 1.", "bundesliga": "Germany 1.",
                "serie a": "Italy 1.", "ligue 1": "France 1.",
                "eredivisie": "Netherlands 1.", "primeira liga": "Portugal 1.",
            }
            for kw, lg in _league_keywords.items():
                if kw in q_lower:
                    requesting_league = lg
                    break
        scored = apply_realism_filter(scored, requesting_league, realism_level)

    if scored.empty:
        st.warning("No players found. Try relaxing the filters.")
        st.stop()

    top_candidates = scored.head(top_n).copy()
    leagues_str = ", ".join(params.get("leagues") or ["all loaded leagues"])
    wanted_raw = [p.upper() for p in (params.get("position_prefixes") or [])]
    pos_str  = " + ".join(wanted_raw) if wanted_raw else "all positions"
    modes_str   = " · ".join(active_mode_labels) if active_mode_labels else "Top performers"

    with st.expander(f"ℹ️ Search details — {len(scored):,} candidates · {pos_str} · {realism_level} realism", expanded=False):
        st.caption(
            f"Positions: **{pos_str}** · Leagues: {leagues_str}\n\n"
            f"Scoring: {modes_str} · Realism: {realism_level} · "
            f"Requesting league (for realism): **{requesting_league or 'not detected — realism inactive'}**"
        )

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

        # Build score breakdown line (only for active modes)
        breakdown_parts = []
        s_top    = player.get("_score_top")
        s_role   = player.get("_score_role")
        s_team   = player.get("_score_team")
        s_player = player.get("_score_player")
        if scoring_modes.get("top_performers") and s_top is not None and not (isinstance(s_top, float) and np.isnan(s_top)):
            breakdown_parts.append(f"🏆 {float(s_top):.0f}")
        if scoring_modes.get("role_fit") and s_role is not None and not (isinstance(s_role, float) and np.isnan(s_role)):
            breakdown_parts.append(f"🎯 {float(s_role):.0f}")
        if scoring_modes.get("team_style") and team_profile and s_team is not None and not (isinstance(s_team, float) and np.isnan(s_team)):
            breakdown_parts.append(f"🏟️ {float(s_team):.0f}")
        if scoring_modes.get("similar_player") and reference_player_row is not None and s_player is not None and not (isinstance(s_player, float) and np.isnan(s_player)):
            breakdown_parts.append(f"👤 {float(s_player):.0f}")
        breakdown_html = (
            f"<div style='color:#6b7280;font-size:11px;margin-top:2px;'>"
            f"{'  ·  '.join(breakdown_parts)}</div>"
        ) if len(breakdown_parts) > 1 else ""

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
                fm_data, tm_data, bio_context, season,
                scoring_modes=scoring_modes,
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
  <div class='cand-rank'>#{rank} · Scout Score {score:.0f}/100{' ℹ️' if breakdown_html else ''}</div>
  {breakdown_html}
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