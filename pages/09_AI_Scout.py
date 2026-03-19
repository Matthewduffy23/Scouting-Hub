# 06_AI_Scout.py — AI Scout Page (Full Build)
# Drop into /pages/ folder in your Streamlit app
# Features:
#   - Natural language scouting queries via Claude API
#   - League-realistic suggestions (no PL suggestions for League Two targets)
#   - Transfermarkt market value fetching
#   - SofIFA attribute reference
#   - Role scores calculated & displayed for each candidate
#   - Full written reports on Top 3 only; Top 4-10 listed only
#   - Season detection from CSV filename (WORLDJUNE25 = 24/25, else 25/26)
#   - Club tactical profile from team stats CSV
#   - Web enrichment for career history, caps, bio info

import io
import re
import os
import math
import time
import unicodedata
import json
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Scout", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# SEASON DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_season(filename: str) -> str:
    """WORLDJUNE25.csv -> '2024/25'. Anything else -> '2025/26'."""
    fn = str(filename).upper()
    if "JUNE25" in fn or "JUN25" in fn:
        return "2024/25"
    return "2025/26"

# ─────────────────────────────────────────────────────────────────────────────
# LEAGUE STRENGTH TABLE
# ─────────────────────────────────────────────────────────────────────────────
LEAGUE_STRENGTHS = {
    "England 1.": 100.00, "Spain 1.": 87.84, "Germany 1.": 87.45,
    "Italy 1.": 85.88, "France 1.": 83.14, "England 2.": 75.10,
    "Belgium 1.": 74.51, "Brazil 1.": 74.31, "Portugal 1.": 72.94,
    "Argentina 1.": 71.37, "USA 1.": 70.00, "Denmark 1.": 70.78,
    "Poland 1.": 69.61, "Turkey 1.": 69.02, "Netherlands 1.": 69.02,
    "Croatia 1.": 68.43, "Germany 2.": 68.04, "Japan 1.": 67.84,
    "Switzerland 1.": 67.45, "Spain 2.": 67.06, "Norway 1.": 66.67,
    "Mexico 1.": 66.47, "Sweden 1.": 66.27, "Colombia 1.": 65.88,
    "Czech 1.": 65.29, "Ecuador 1.": 65.29, "Greece 1.": 64.12,
    "Saudi 1.": 64.12, "Italy 2.": 63.53, "Hungary 1.": 63.53,
    "Austria 1.": 63.33, "Morocco 1.": 63.14, "Korea 1.": 62.75,
    "France 2.": 64.00, "England 3.": 61.96, "Romania 1.": 61.76,
    "Scotland 1.": 61.76, "Russia 1.": 62.41, "Uruguay 1.": 60.39,
    "Chile 1.": 59.80, "Israel 1.": 58.43, "Brazil 2.": 58.04,
    "Slovenia 1.": 57.45, "Slovakia 1.": 56.47, "Germany 3.": 54.51,
    "Ukraine 1.": 54.31, "Portugal 2.": 53.14, "Serbia 1.": 52.16,
    "Japan 2.": 50.98, "England 4.": 50.78, "Ireland 1.": 50.59,
    "France 3.": 49.61, "Belgium 2.": 48.43, "Finland 1.": 48.43,
    "Switzerland 2.": 46.47, "Norway 2.": 45.88, "Sweden 2.": 45.69,
    "Turkey 2.": 44.51, "Czech 2.": 43.33, "Netherlands 2.": 42.16,
    "Italy 3.": 45.00, "Denmark 2.": 40.39, "Scotland 2.": 38.63,
    "England 5.": 33.33, "England 6.": 16.08, "England 7.": 37.25,
    "England 8.": 15.69, "England 9.": 31.37, "England 10.": 3.92,
    "Germany 4.": 35.29, "Portugal 3.": 35.29,
}

# ─────────────────────────────────────────────────────────────────────────────
# ROLE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
ROLE_BUCKETS = {
    "CB": {
        "Ball Playing CB": {
            "Passes per 90": 2, "Accurate passes, %": 2, "Forward passes per 90": 2,
            "Accurate forward passes, %": 2, "Progressive passes per 90": 2,
            "Progressive runs per 90": 1.5, "Dribbles per 90": 1.5,
            "Accurate long passes, %": 1, "Passes to final third per 90": 1.5,
        },
        "Wide CB": {
            "Defensive duels per 90": 1.5, "Defensive duels won, %": 2,
            "Dribbles per 90": 2, "Forward passes per 90": 1,
            "Progressive passes per 90": 1, "Progressive runs per 90": 2,
        },
        "Box Defender": {
            "Aerial duels per 90": 1, "Aerial duels won, %": 3,
            "PAdj Interceptions": 2, "Shots blocked per 90": 1,
            "Defensive duels won, %": 4,
        },
        "PL Profile": {
            "Defensive duels won, %": 2, "Aerial duels won, %": 3,
            "Shots blocked per 90": 1, "PAdj Interceptions": 1,
        },
    },
    "FB": {
        "Build Up FB": {
            "Passes per 90": 2, "Accurate passes, %": 1.5, "Forward passes per 90": 2,
            "Accurate forward passes, %": 2, "Progressive passes per 90": 2.5,
            "Progressive runs per 90": 2, "Dribbles per 90": 2,
            "Passes to final third per 90": 2, "xA per 90": 1,
        },
        "Attacking FB": {
            "Crosses per 90": 2, "Dribbles per 90": 3.5, "Accelerations per 90": 1,
            "Successful dribbles, %": 1, "Touches in box per 90": 2,
            "Progressive runs per 90": 3, "Passes to penalty area per 90": 2, "xA per 90": 3,
        },
        "Defensive FB": {
            "Aerial duels per 90": 1, "Aerial duels won, %": 1.5,
            "Defensive duels per 90": 2, "PAdj Interceptions": 3,
            "Shots blocked per 90": 1, "Defensive duels won, %": 3.5,
        },
    },
    "CM": {
        "Deep Playmaker CM": {
            "Passes per 90": 1, "Accurate passes, %": 1, "Forward passes per 90": 2,
            "Accurate forward passes, %": 1.5, "Progressive passes per 90": 3,
            "Passes to final third per 90": 2.5, "Accurate long passes, %": 1,
        },
        "Advanced Playmaker CM": {
            "Deep completions per 90": 1.5, "Smart passes per 90": 2,
            "xA per 90": 4, "Passes to penalty area per 90": 2,
        },
        "Defensive CM": {
            "Defensive duels per 90": 4, "Defensive duels won, %": 4,
            "PAdj Interceptions": 3, "Aerial duels per 90": 0.5, "Aerial duels won, %": 1,
        },
        "Ball Carrying CM": {
            "Dribbles per 90": 4, "Successful dribbles, %": 2,
            "Progressive runs per 90": 3, "Accelerations per 90": 3,
        },
    },
    "ATT": {
        "Playmaker ATT": {
            "Passes per 90": 2, "xA per 90": 3, "Key passes per 90": 1,
            "Deep completions per 90": 1.5, "Smart passes per 90": 1.5,
            "Passes to penalty area per 90": 2,
        },
        "Goal Threat ATT": {
            "xG per 90": 3, "Non-penalty goals per 90": 3,
            "Shots per 90": 2, "Touches in box per 90": 2,
        },
        "Ball Carrier ATT": {
            "Dribbles per 90": 4, "Successful dribbles, %": 2,
            "Progressive runs per 90": 3, "Accelerations per 90": 3,
        },
    },
    "CF": {
        "Target Man CF": {
            "Aerial duels per 90": 3, "Aerial duels won, %": 5,
        },
        "Goal Threat CF": {
            "Non-penalty goals per 90": 3, "Shots per 90": 1.5, "xG per 90": 3,
            "Touches in box per 90": 1, "Shots on target, %": 0.5,
        },
        "Link Up CF": {
            "Passes per 90": 2, "Passes to penalty area per 90": 1.5,
            "Deep completions per 90": 1, "Smart passes per 90": 1.5,
            "Accurate passes, %": 1.5, "Key passes per 90": 1,
            "Dribbles per 90": 2, "Successful dribbles, %": 1,
            "Progressive runs per 90": 2, "xA per 90": 3,
        },
    },
    "GK": {
        "Shot Stopper GK": {
            "Prevented goals per 90": 3, "Save rate, %": 1,
        },
        "Ball Playing GK": {
            "Passes per 90": 1, "Accurate passes, %": 3, "Accurate long passes, %": 2,
        },
        "Sweeper GK": {
            "Exits per 90": 1,
        },
    },
}

POSITION_TO_ROLE_KEY = {
    "GK": "GK",
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "DMF": "CM", "LDMF": "CM", "RDMF": "CM", "LCMF": "CM", "RCMF": "CM",
    "AMF": "ATT", "LAMF": "ATT", "RAMF": "ATT",
    "LW": "ATT", "LWF": "ATT", "RW": "ATT", "RWF": "ATT",
    "CF": "CF",
}

def get_role_key(position: str) -> str:
    tok = str(position).split(",")[0].strip().upper()
    return POSITION_TO_ROLE_KEY.get(tok, "CF")

def compute_role_scores(player_row: pd.Series, pool_df: pd.DataFrame, position: str) -> dict:
    """Compute all role scores for a player vs their league pool."""
    role_key = get_role_key(position)
    roles = ROLE_BUCKETS.get(role_key, {})
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

def role_score_color(v: float) -> str:
    if v >= 85: return "#2E6114"
    if v >= 75: return "#5C9E2E"
    if v >= 66: return "#7FBC41"
    if v >= 55: return "#A7D763"
    if v >= 41: return "#F6D645"
    if v >= 25: return "#D77A2E"
    return "#C63733"

# ─────────────────────────────────────────────────────────────────────────────
# LEAGUE REALISM BANDS
# ─────────────────────────────────────────────────────────────────────────────
def get_league_band(league: str) -> int:
    s = LEAGUE_STRENGTHS.get(league, 40.0)
    if s >= 80: return 1
    if s >= 65: return 2
    if s >= 55: return 3
    if s >= 45: return 4
    if s >= 30: return 5
    return 6

BAND_DESCRIPTION = {
    1: "Top 5 European (PL/La Liga/Bundesliga/Serie A/Ligue 1)",
    2: "Strong European — Championship, Eredivisie, Pro League, Primeira Liga",
    3: "Mid-tier European — League One standard, lower Championship",
    4: "League Two / National League standard",
    5: "Non-League / Conference level",
    6: "Amateur / youth leagues",
}

# ─────────────────────────────────────────────────────────────────────────────
# SCOUT PERSONA SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
def build_scout_system_prompt(player_league: str, requesting_league: str = None) -> str:
    player_band = get_league_band(player_league)
    player_strength = LEAGUE_STRENGTHS.get(player_league, 50.0)

    if requesting_league:
        req_band = get_league_band(requesting_league)
        req_strength = LEAGUE_STRENGTHS.get(requesting_league, 50.0)
        realism_note = f"""
LEAGUE REALISM — CRITICAL RULE:
Requesting club: {requesting_league} (strength {req_strength:.1f}, Band {req_band}).
Target player: {player_league} (strength {player_strength:.1f}, Band {player_band}).
Only suggest clubs within 1 band of the player's current level.
Band {player_band} player → realistic clubs are Band {max(1, player_band-1)} to Band {min(6, player_band+1)}.
A League Two (Band 4) player should NEVER be suggested to Premier League (Band 1) clubs.
Contextualise all stats: {player_league} 90th pct ≈ ~{min(90, int(90 * player_strength / 100))}th pct in England 1."""
    else:
        realism_note = f"""
LEAGUE REALISM — CRITICAL RULE:
Player competes in {player_league} (strength {player_strength:.1f}/100, Band {player_band}).
Realistic next-step is one band up at most. Do not overstate the player's level."""

    return f"""You are a senior scout at a professional football club.
You write concise, direct scouting reports in the style of a top-flight recruitment department.

SCORING SYSTEM:
Role scores are percentile-based (0-100) vs league pool for position.
85+ = Elite for level. 75-84 = Strong. 66-74 = Above average. 55-65 = Average.
41-54 = Below average. 25-40 = Weakness. <25 = Liability.

LEAGUE STRENGTHS (benchmark England 1. = 100):
England 1.=100 | England 2.=75 | England 3.=62 | England 4.=51 | England 5.=33
Spain 1.=88 | Germany 1.=87 | Italy 1.=86 | France 1.=83 | Championship=75
A player at 80th pct in England 4. ≈ 55th-60th pct in Championship ≈ 45th pct in PL.

TACTICAL CONTEXT:
PPDA<7=Very High Press | PPDA 7-9=High Press | PPDA 9-12=Moderate | PPDA>14=Deep Block
Possession>55%=Possession-dominant | <45%=Reactive/direct | Long passes>55 + poss<50%=Direct

{realism_note}

REPORT STYLE — MANDATORY FORMAT:
Sentence 1: Player's single defining quality — factual, one sentence.
Sentence 2: Best statistical evidence with league context and season reference.
Sentence 3: Fit to the requesting club's tactical profile.
Sentence 4: One specific genuine risk or weakness with data evidence.
Sentence 5: RECOMMENDATION: [SIGN/MONITOR/PASS] — PRIORITY: [HIGH/MEDIUM/LOW] — brief rationale.

RULES:
- Never use "impressive", "talented", "exciting" without supporting data.
- Always cite actual metric values, not vague descriptors.
- Reference games played, contract situation, age trajectory, market value context.
- If bio data available: reference specific clubs played for, seasons, goal records.
- Be realistic about level. Do not oversell.
- Mention the season data is from (e.g. '2024/25 data shows...')

EXAMPLE:
"Morgan ranks 94th percentile for progressive passes per 90 in League One (2024/25),
averaging 6.2 per game — the highest in the bottom half of the table. His aerial win rate
of 71% (85th pct) suits a high-line structure; previously at Shrewsbury and Crewe, he has
108 senior appearances and won promotion from League Two in 2023. At 23 with 12 months
remaining on his Shrewsbury contract (TM value £400k), he represents realistic Championship
value. His defensive duel win rate of 58% (52nd pct) will be tested in transition-heavy systems.
RECOMMENDATION: MONITOR — PRIORITY: MEDIUM — Reassess in January if contract situation develops."
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    repl = {"o":"o","ae":"ae","a":"a","a":"a","o":"o","u":"u",
            "ss":"ss","l":"l","d":"d","d":"d","th":"th","c":"c","s":"s","g":"g","i":"i"}
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s)

def format_market_value(v) -> str:
    try:
        v = float(v)
        if not np.isfinite(v) or v <= 0:
            return "—"
        if v >= 1_000_000:
            return f"€{v/1_000_000:.1f}m"
        if v >= 1_000:
            return f"€{int(v/1_000)}k"
        return f"€{int(v)}"
    except Exception:
        return "—"

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_transfermarkt_value(player: str, team: str) -> dict:
    """Fetch market value from Transfermarkt."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-GB,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        query = f"{player} {team}"
        url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={requests.utils.quote(query)}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return {}
        html = r.text
        # Try to find market value in search results
        # Look for value patterns like "£1.50m" or "€500k"
        mv_patterns = [
            r'class="rechts hauptlink"[^>]*>[\s\S]{0,200}?([£€]\d+(?:\.\d+)?(?:m|k|bn))',
            r'"marketValue":\s*"([^"]+)"',
            r'marktwert[^>]*>([^<]+)',
        ]
        for pat in mv_patterns:
            m = re.search(pat, html, re.IGNORECASE)
            if m:
                return {"market_value": m.group(1).strip(), "source": "transfermarkt"}
        return {"market_value": "see transfermarkt.com", "source": "transfermarkt"}
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_sofifa_attributes(player: str) -> dict:
    """Fetch FC/FIFA ratings from SofIFA."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }
        url = f"https://sofifa.com/players?keyword={requests.utils.quote(player)}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return {}
        html = r.text
        overall = re.search(r'class="col col-oa[^"]*"[^>]*>\s*<span[^>]*>(\d+)</span>', html)
        potential = re.search(r'class="col col-pt[^"]*"[^>]*>\s*<span[^>]*>(\d+)</span>', html)
        if overall:
            result = {"overall": int(overall.group(1)), "source": "sofifa"}
            if potential:
                result["potential"] = int(potential.group(1))
            return result
        return {}
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=1800)
def load_csv_cached(data: bytes, filename: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    df["_source_file"] = filename
    df["_season"] = detect_season(filename)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_and_merge(uploaded_files) -> tuple:
    frames = []
    filenames = []
    try:
        file_pairs = [(f.name, f.getvalue()) for f in uploaded_files]
    except Exception:
        return pd.DataFrame(), []

    for name, data in file_pairs:
        try:
            df = load_csv_cached(data, name)
            frames.append(df)
            filenames.append(name)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")

    if not frames:
        return pd.DataFrame(), []

    merged = pd.concat(frames, ignore_index=True)
    if "Player" in merged.columns and "Team" in merged.columns:
        merged["Minutes played"] = pd.to_numeric(merged.get("Minutes played", 0), errors="coerce").fillna(0)
        merged = (merged
                  .sort_values("Minutes played", ascending=False)
                  .drop_duplicates(subset=["Player", "Team"], keep="first")
                  .reset_index(drop=True))
    return merged, filenames

# ─────────────────────────────────────────────────────────────────────────────
# POSITION FILTER
# ─────────────────────────────────────────────────────────────────────────────
POSITION_PREFIXES = {
    "CB": ("LCB", "RCB", "CB"),
    "FB": ("LB", "RB", "LWB", "RWB"),
    "GK": ("GK",),
    "CM": ("DMF", "LDMF", "RDMF", "LCMF", "RCMF", "CMF"),
    "ATT": ("AMF", "LAMF", "RAMF", "LW", "LWF", "RW", "RWF"),
    "CF": ("CF",),
    "DEF": ("LCB", "RCB", "CB", "LB", "RB", "LWB", "RWB"),
    "MID": ("DMF", "LDMF", "RDMF", "LCMF", "RCMF", "CMF", "AMF", "LAMF", "RAMF"),
    "FWD": ("LW", "LWF", "RW", "RWF", "CF", "AMF"),
}

def position_matches(pos_str: str, target_positions: list) -> bool:
    tok = str(pos_str).split(",")[0].strip().upper()
    for tp in target_positions:
        tp_upper = tp.upper()
        prefixes = POSITION_PREFIXES.get(tp_upper, (tp_upper,))
        if any(tok.startswith(p) for p in prefixes):
            return True
        if tok == tp_upper:
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE API CALLS
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(api_key: str, messages: list, system: str = "",
                model: str = "claude-haiku-4-5", max_tokens: int = 1000) -> str:
    try:
        payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            payload["system"] = system
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if r.status_code != 200:
            return f"[API Error {r.status_code}: {r.text[:200]}]"
        return r.json()["content"][0]["text"]
    except Exception as e:
        return f"[Error: {e}]"

def call_claude_with_search(api_key: str, prompt: str, system: str = "",
                             max_tokens: int = 600) -> str:
    """Sonnet with web search tool for live player bio/career data."""
    try:
        payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
        if system:
            payload["system"] = system
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        if r.status_code != 200:
            return ""
        data = r.json()
        text_parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        return " ".join(text_parts).strip()
    except Exception:
        return ""

def extract_parameters(api_key: str, query: str) -> dict:
    """Haiku: parse natural language query into structured parameters."""
    system = """Extract scouting parameters. Return ONLY valid JSON with these fields:
- position: list of strings (Wyscout format: "CB","LCB","CF","DMF" etc.)
- max_age: int or null
- min_age: int or null  
- leagues: list of Wyscout league names WITH trailing dot e.g. ["England 4.","England 3."]
- max_budget_eur: int or null (convert £ to EUR at 1.2x if needed)
- style_traits: list of strings
- physical_traits: list of strings
- requesting_club: string or null
- requesting_league: string or null (Wyscout format with dot)
- foot: string or null ("left"/"right"/"both")
- min_minutes: int (default 500)
Return only the JSON object. No markdown, no explanation."""

    resp = call_claude(api_key,
                       [{"role": "user", "content": f"Parse this scouting query: {query}"}],
                       system=system, model="claude-haiku-4-5", max_tokens=500)
    try:
        clean = resp.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(clean)
    except Exception:
        return {"position": ["CF"], "min_minutes": 500}

def build_club_narrative(api_key: str, club: str, league: str, stats: dict) -> str:
    """Haiku: tactical 3-sentence club profile."""
    if not stats:
        return f"{club} ({league}) — tactical profile data unavailable."
    system = build_scout_system_prompt(league, league)
    stat_text = "; ".join([f"{k}: {round(float(v), 2) if isinstance(v, (int, float)) else v}"
                           for k, v in stats.items() if pd.notna(v)])
    prompt = (f"Write a 3-sentence tactical profile of {club} ({league}). "
              f"Stats: {stat_text}. "
              f"End with: 'This suggests they need a [type] who can [function].' "
              f"Reference actual stat values.")
    return call_claude(api_key,
                       [{"role": "user", "content": prompt}],
                       system=system, model="claude-haiku-4-5", max_tokens=250)

def fetch_player_bio(api_key: str, player: str, team: str, league: str,
                     season: str, position: str) -> str:
    """Sonnet + web search: career history, caps, bio."""
    prompt = (f"Search for {player} who plays for {team} in {league} ({season}) as {position}. "
              f"Find: nationality, age, career clubs and seasons, international caps, "
              f"goals/assists record, any notable facts. "
              f"Return a 3-sentence factual career summary only. No speculation.")
    system = "Football analyst. Factual only. Be concise. Reference specific clubs and seasons."
    result = call_claude_with_search(api_key, prompt, system=system, max_tokens=400)
    return result if result and "[Error" not in result and len(result) > 20 else ""

def generate_full_report(api_key: str, player_name: str, team: str, league: str,
                          position: str, season: str, metrics_summary: str,
                          role_scores: dict, club_narrative: str,
                          bio_context: str, tm_data: dict, sofifa_data: dict,
                          params: dict) -> str:
    """Sonnet: full written scout report."""
    requesting_league = params.get("requesting_league") or league
    system = build_scout_system_prompt(league, requesting_league)

    tm_text = f"Transfermarkt value: {tm_data.get('market_value', 'check TM')}." if tm_data else ""
    sofa_text = f"SofIFA overall: {sofifa_data.get('overall', 'N/A')} (potential: {sofifa_data.get('potential', 'N/A')})." if sofifa_data else ""

    best_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    role_text = " | ".join([f"{r}: {s:.0f}/100" for r, s in best_roles])

    bio_section = f"\nCARREER CONTEXT: {bio_context}" if bio_context else ""
    budget_text = f"Max budget: €{params.get('max_budget_eur', 0):,}." if params.get("max_budget_eur") else ""
    style_text = f"Required traits: {', '.join(params.get('style_traits', []))}." if params.get("style_traits") else ""

    player_band = get_league_band(league)
    realistic_level = BAND_DESCRIPTION.get(max(1, player_band - 1), "one level up")

    prompt = f"""Write a 5-sentence scouting report for {player_name}.

DATA ({season} season):
Position: {position} | Club: {team} | League: {league} (Band {player_band})
{bio_section}

KEY METRICS (percentile vs league pool):
{metrics_summary}

ROLE SCORES: {role_text}

MARKET: {tm_text} {sofa_text}

CLUB CONTEXT: {club_narrative}

CONSTRAINTS: {budget_text} {style_text}

REALISTIC NEXT LEVEL: {realistic_level}

Follow the 5-sentence report format exactly. 
Reference the {season} season in sentence 2.
End with: RECOMMENDATION: [SIGN/MONITOR/PASS] — PRIORITY: [HIGH/MEDIUM/LOW] — [one-sentence rationale]."""

    return call_claude(api_key,
                       [{"role": "user", "content": prompt}],
                       system=system,
                       model="claude-sonnet-4-6",
                       max_tokens=600)

def generate_chief_scout_summary(api_key: str, candidates: list, club_narrative: str, params: dict) -> str:
    """Sonnet: executive chief scout recommendation."""
    if not candidates:
        return ""
    requesting_league = params.get("requesting_league", "")
    system = build_scout_system_prompt(requesting_league or "England 3.", requesting_league)

    cands_text = "\n\n".join([
        f"#{i+1} {c['player']} ({c['team']}, {c['league']}, {c.get('season','')}) — "
        f"Impact: {c.get('impact', 0):.0f} | Best Role: {c.get('best_role','')}: {c.get('best_role_score', 0):.0f}\n"
        f"Report summary: {c.get('report', '')[:250]}..."
        for i, c in enumerate(candidates)
    ])

    prompt = (f"As Chief Scout, write a 4-sentence executive summary for these candidates:\n\n{cands_text}\n\n"
              f"Club context: {club_narrative}\n\n"
              f"Identify: (1) primary recommendation with rationale, "
              f"(2) best value alternative, (3) timing/market considerations. "
              f"Be decisive and realistic about level.")
    return call_claude(api_key,
                       [{"role": "user", "content": prompt}],
                       system=system,
                       model="claude-sonnet-4-6",
                       max_tokens=400)

# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def score_candidates(df: pd.DataFrame, params: dict, team_stats_row: dict = None) -> pd.DataFrame:
    """Score with style-adaptive weights + league adjustment."""
    df = df.copy()
    if df.empty:
        return df

    positions = params.get("position", ["CF"])
    pos_key = get_role_key(positions[0] if positions else "CF")
    roles = ROLE_BUCKETS.get(pos_key, {})
    if not roles:
        df["_impact_score"] = 50.0
        return df

    # Style-adaptive multipliers
    mults = {}
    if team_stats_row:
        ppda = float(team_stats_row.get("PPDA", 10) or 10)
        poss = float(team_stats_row.get("Possession %", 50) or 50)
        long_p = float(team_stats_row.get("Long passes p90", 40) or 40)
        aer = float(team_stats_row.get("Aerial Duels Won %", 50) or 50)
        if ppda < 8:
            mults["Defensive duels per 90"] = 1.8
            mults["Accelerations per 90"] = 1.5
        if poss > 57:
            mults["Accurate passes, %"] = 1.6
            mults["Progressive passes per 90"] = 1.4
        if poss < 45 or long_p > 55:
            mults["Aerial duels won, %"] = 2.0
            mults["Aerial duels per 90"] = 1.8
        if aer > 60:
            mults["Aerial duels won, %"] = max(mults.get("Aerial duels won, %", 1.0), 1.6)

    best_scores = []
    for _, row in df.iterrows():
        role_scores_local = {}
        for role_name, metrics in roles.items():
            total_w, wsum = 0.0, 0.0
            for met, w in metrics.items():
                if met not in df.columns:
                    continue
                pool_vals = pd.to_numeric(df[met], errors="coerce").dropna()
                player_val = pd.to_numeric(row.get(met, np.nan), errors="coerce")
                if pd.isna(player_val) or pool_vals.empty:
                    continue
                pct = float((pool_vals < player_val).mean() * 100 + (pool_vals == player_val).mean() * 50)
                adj_w = w * mults.get(met, 1.0)
                wsum += pct * adj_w
                total_w += adj_w
            if total_w > 0:
                role_scores_local[role_name] = wsum / total_w
        best = max(role_scores_local.values()) if role_scores_local else 0.0
        best_scores.append(best)

    df["_base_score"] = best_scores
    df["_league_strength"] = df["League"].map(LEAGUE_STRENGTHS).fillna(50.0)
    ls_norm = np.clip(df["_league_strength"] / 100.0, 0.30, 1.00)
    df["_impact_raw"] = df["_base_score"] * (ls_norm ** 1.6)

    raw = df["_impact_raw"]
    lo, hi = raw.min(), raw.max()
    df["_impact_score"] = 100.0 * (raw - lo) / (hi - lo) if hi > lo else 50.0
    return df

def filter_candidates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    result = df.copy()
    for col in ["Minutes played", "Age", "Market value"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    positions = params.get("position", [])
    if positions:
        result = result[result["Position"].astype(str).apply(
            lambda p: position_matches(p, positions))]

    min_mins = params.get("min_minutes", 500)
    if "Minutes played" in result.columns:
        result = result[result["Minutes played"] >= min_mins]

    if params.get("max_age") and "Age" in result.columns:
        result = result[result["Age"] <= params["max_age"]]
    if params.get("min_age") and "Age" in result.columns:
        result = result[result["Age"] >= params["min_age"]]

    if params.get("max_budget_eur") and "Market value" in result.columns:
        result = result[result["Market value"] <= params["max_budget_eur"]]

    leagues = params.get("leagues", [])
    if leagues and "League" in result.columns:
        result = result[result["League"].isin(leagues)]

    if params.get("foot"):
        for col in ("Foot", "Preferred foot", "Preferred Foot"):
            if col in result.columns:
                result = result[
                    result[col].astype(str).str.lower().str.startswith(
                        params["foot"].lower()[:1])]
                break

    return result.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
KEY_METRICS_BY_ROLE = {
    "CB": ["Aerial duels won, %", "Defensive duels won, %", "PAdj Interceptions",
           "Progressive passes per 90", "Accurate passes, %", "Shots blocked per 90",
           "Aerial duels per 90", "Defensive duels per 90"],
    "FB": ["Defensive duels won, %", "xA per 90", "Crosses per 90",
           "Progressive runs per 90", "Accurate passes, %", "PAdj Interceptions",
           "Dribbles per 90", "Passes to final third per 90"],
    "CM": ["Progressive passes per 90", "Passes to final third per 90", "xA per 90",
           "Defensive duels won, %", "PAdj Interceptions", "Accurate passes, %",
           "Dribbles per 90", "Key passes per 90"],
    "ATT": ["xG per 90", "xA per 90", "Non-penalty goals per 90", "Dribbles per 90",
            "Touches in box per 90", "Progressive runs per 90", "Shots per 90",
            "Successful dribbles, %"],
    "CF": ["Non-penalty goals per 90", "xG per 90", "Aerial duels won, %",
           "Touches in box per 90", "Shots on target, %", "Dribbles per 90",
           "Aerial duels per 90", "Passes per 90"],
    "GK": ["Save rate, %", "Prevented goals per 90", "Exits per 90",
           "Accurate long passes, %", "Passes per 90"],
}

def build_metrics_summary(player_row: pd.Series, pool_df: pd.DataFrame, position: str) -> str:
    role_key = get_role_key(position)
    key_mets = KEY_METRICS_BY_ROLE.get(role_key, KEY_METRICS_BY_ROLE["CF"])
    lines = []
    for met in key_mets:
        if met not in pool_df.columns:
            continue
        val = pd.to_numeric(player_row.get(met, np.nan), errors="coerce")
        if pd.isna(val):
            continue
        pool_vals = pd.to_numeric(pool_df[met], errors="coerce").dropna()
        if pool_vals.empty:
            continue
        pct = int((pool_vals < val).mean() * 100 + (pool_vals == val).mean() * 50)
        lines.append(f"  {met}: {val:.2f} ({pct}th pct)")
    return "\n".join(lines) if lines else "No metrics available."

# ─────────────────────────────────────────────────────────────────────────────
# ROLE SCORE HTML
# ─────────────────────────────────────────────────────────────────────────────
def render_role_scores_html(role_scores: dict) -> str:
    if not role_scores:
        return ""
    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
    pills = []
    for role, score in sorted_roles:
        color = role_score_color(score)
        fg = "#000" if score >= 54 else "#fff"
        pills.append(
            f"<span style='background:{color};color:{fg};padding:2px 10px;border-radius:6px;"
            f"font-size:13px;font-weight:700;margin:2px;display:inline-block;'>"
            f"{role}: {score:.0f}</span>"
        )
    return "<div style='margin:6px 0;'>" + " ".join(pills) + "</div>"

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🤖 AI Scout")
st.caption("Natural language scouting powered by Claude — statistics, market intelligence, career context.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Claude API Key", type="password", key="ai_scout_api_key",
                             help="From console.anthropic.com")
    st.markdown("---")
    st.subheader("📂 Player CSVs")
    uploaded_files = st.file_uploader(
        "Upload WORLD*.csv files",
        type=["csv"], accept_multiple_files=True,
        key="ai_scout_player_csvs",
        help="WORLDJUNE25.csv = 2024/25 season. Other files = 2025/26."
    )

    st.subheader("📊 Team Stats CSV")
    team_stats_file = st.file_uploader(
        "Upload team stats CSV (optional — for club profiles)",
        type=["csv"], key="ai_scout_team_stats",
    )

    st.markdown("---")
    st.subheader("🔧 Settings")
    top_n = st.slider("Candidates to evaluate", 5, 15, 10, key="ai_scout_top_n")
    fetch_bio = st.checkbox("Fetch live player bio (web)", value=True, key="ai_scout_fetch_bio",
                             help="Sonnet + web search. ~£0.01 per player.")
    fetch_tm = st.checkbox("Fetch Transfermarkt values", value=True, key="ai_scout_fetch_tm")
    fetch_sofa = st.checkbox("Fetch SofIFA ratings", value=False, key="ai_scout_fetch_sofa")

# ── Load data ─────────────────────────────────────────────────────────────────
df_players = pd.DataFrame()
filenames = []

if uploaded_files:
    df_players, filenames = load_and_merge(uploaded_files)
    if not df_players.empty:
        seasons = df_players["_season"].unique().tolist()
        st.success(
            f"✅ {len(df_players):,} players loaded from {len(filenames)} file(s). "
            f"Season(s): {', '.join(seasons)}"
        )
else:
    st.info("👈 Upload WORLD*.csv player files in the sidebar to begin.")

df_team_stats = pd.DataFrame()
if team_stats_file:
    try:
        df_team_stats = pd.read_csv(io.BytesIO(team_stats_file.getvalue()))
        st.success(f"✅ Team stats: {len(df_team_stats)} teams loaded.")
    except Exception as e:
        st.warning(f"Team stats error: {e}")

# ── Query ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Scout Query")

query = st.text_area(
    "Describe what you're looking for",
    placeholder=(
        "e.g. 'Salford City need a U23 CB under £500k. Direct team, aerial dominant, "
        "can play out from back. League Two or League One standard.'\n\n"
        "e.g. 'Championship club looking for creative CM, max 26, possession-based, "
        "progressive passer, max £2m budget.'"
    ),
    height=120,
    key="ai_scout_query",
)

run_btn = st.button("🚀 Run AI Scout", type="primary",
                     disabled=(not api_key or df_players.empty))

# ── Main Logic ────────────────────────────────────────────────────────────────
if run_btn and api_key and not df_players.empty:

    # Step 1: Extract parameters
    with st.spinner("Analysing query..."):
        params = extract_parameters(api_key, query)

    st.markdown("---")
    col_p, col_b = st.columns([2, 1])
    with col_p:
        st.markdown("**🎯 Detected Parameters**")
        st.json({k: v for k, v in params.items() if v})
    with col_b:
        req_league = params.get("requesting_league", "")
        if req_league:
            band = get_league_band(req_league)
            st.info(f"**Club level:** Band {band}\n\n{BAND_DESCRIPTION.get(band, '')}")

    # Step 2: Club narrative
    club_narrative = ""
    team_stats_row = {}
    req_club = params.get("requesting_club", "")

    if req_club and not df_team_stats.empty:
        matches = df_team_stats[
            df_team_stats["Team"].astype(str).str.lower().str.contains(
                req_club.lower()[:8], na=False
            )
        ]
        if not matches.empty:
            ts_row = matches.iloc[0]
            team_stats_row = ts_row.to_dict()
            relevant_stats = {k: v for k, v in team_stats_row.items()
                              if k in ["PPDA", "Possession %", "Passes p90", "Long passes p90",
                                       "Aerial Duels Won %", "xG p90", "xG Against p90",
                                       "Progressive Passes p90", "Aerial duels p90"]
                              and pd.notna(v)}
            with st.spinner(f"Building tactical profile for {req_club}..."):
                club_narrative = build_club_narrative(
                    api_key, req_club,
                    req_league or str(ts_row.get("League", "")),
                    relevant_stats
                )

    if club_narrative:
        st.markdown("---")
        st.markdown("**🏟️ Club Tactical Profile**")
        st.markdown(f"*{club_narrative}*")

    # Step 3: Filter & score
    st.markdown("---")
    with st.spinner("Filtering and scoring candidates..."):
        df_filtered = filter_candidates(df_players, params)
        if df_filtered.empty:
            st.warning("No candidates match the filters. Try broadening position, leagues, age or budget.")
            st.stop()

        df_scored = score_candidates(df_filtered, params, team_stats_row)
        df_ranked = (df_scored
                     .sort_values("_impact_score", ascending=False)
                     .head(int(top_n))
                     .reset_index(drop=True))

    st.success(f"**{len(df_filtered):,}** candidates matched → top {len(df_ranked)} scored.")

    # ── TOP 3: Full Reports ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Scouting Reports — Top 3")

    chief_candidates = []

    for rank_i in range(min(3, len(df_ranked))):
        row = df_ranked.iloc[rank_i]
        player_name = str(row.get("Player", "Unknown"))
        team = str(row.get("Team", ""))
        league = str(row.get("League", ""))
        position = str(row.get("Position", ""))
        season = str(row.get("_season", "2025/26"))
        impact = float(row.get("_impact_score", 0))

        # Player header
        st.markdown(f"### #{rank_i + 1} — {player_name}")
        st.markdown(f"**{team}** · {league} · {position} · {season} season")

        # Detail row
        detail = []
        age = row.get("Age", "")
        mins = row.get("Minutes played", "")
        goals = row.get("Goals", "")
        assists = row.get("Assists", "")
        mv = row.get("Market value", "")
        contract = row.get("Contract expires", "")

        if pd.notna(age) and str(age) not in ("", "nan"):
            detail.append(f"Age {int(float(age))}")
        if pd.notna(mins) and str(mins) not in ("", "nan"):
            detail.append(f"{int(float(mins))} mins")
        if pd.notna(goals) and str(goals) not in ("", "nan", "0"):
            detail.append(f"⚽ {int(float(goals))}")
        if pd.notna(assists) and str(assists) not in ("", "nan", "0"):
            detail.append(f"🅰 {int(float(assists))}")
        if pd.notna(mv) and str(mv) not in ("", "nan", "0"):
            detail.append(f"MV: {format_market_value(mv)}")
        if pd.notna(contract) and str(contract) not in ("", "nan"):
            detail.append(f"Contract: {str(contract)[:7]}")

        if detail:
            st.markdown("  |  ".join(detail))

        # League context warning
        player_band = get_league_band(league)
        req_band_val = get_league_band(req_league) if req_league else None
        if req_band_val and player_band > req_band_val + 1:
            st.warning(f"⚠️ Note: Player is {player_band - req_band_val} league band(s) below requesting club level. "
                       f"Significant step up required.")

        # Impact score
        imp_color = role_score_color(impact)
        imp_fg = "#000" if impact >= 54 else "#fff"
        st.markdown(
            f"<span style='background:{imp_color};color:{imp_fg};padding:4px 12px;"
            f"border-radius:8px;font-weight:800;font-size:14px;'>"
            f"Impact Score: {impact:.0f}/100</span>",
            unsafe_allow_html=True,
        )

        # Role scores
        role_key = get_role_key(position)
        pos_pool = df_players[
            (df_players["Position"].astype(str).apply(lambda p: get_role_key(p) == role_key)) &
            (df_players["League"] == league)
        ]
        if len(pos_pool) < 10:
            pos_pool = df_players[
                df_players["Position"].astype(str).apply(lambda p: get_role_key(p) == role_key)
            ]

        with st.spinner(f"Computing role scores for {player_name}..."):
            role_scores = compute_role_scores(row, pos_pool, position)

        if role_scores:
            best_role = max(role_scores, key=role_scores.get)
            best_score = role_scores[best_role]
            st.markdown("**Role Scores (vs league pool):**")
            st.markdown(render_role_scores_html(role_scores), unsafe_allow_html=True)
        else:
            best_role, best_score = "N/A", 0

        # Key metrics
        metrics_summary = build_metrics_summary(row, pos_pool, position)
        with st.expander("📊 Key Metrics (percentile vs league pool)", expanded=False):
            for line in metrics_summary.strip().split("\n"):
                st.text(line)

        # Fetch live data
        tm_data, sofifa_data, bio_context = {}, {}, ""

        if fetch_tm:
            with st.spinner(f"Transfermarkt: {player_name}..."):
                tm_data = fetch_transfermarkt_value(player_name, team)
            if tm_data.get("market_value"):
                st.caption(f"💰 Transfermarkt: {tm_data['market_value']}")

        if fetch_sofa:
            with st.spinner(f"SofIFA: {player_name}..."):
                sofifa_data = fetch_sofifa_attributes(player_name)
            if sofifa_data.get("overall"):
                st.caption(f"🎮 SofIFA: {sofifa_data['overall']} overall / {sofifa_data.get('potential', 'N/A')} potential")

        if fetch_bio:
            with st.spinner(f"Searching career data for {player_name}..."):
                bio_context = fetch_player_bio(api_key, player_name, team, league, season, position)
            if bio_context:
                st.markdown(f"**📰 Career:** {bio_context}")

        # Full report
        with st.spinner(f"Writing scout report for {player_name}..."):
            report = generate_full_report(
                api_key=api_key,
                player_name=player_name,
                team=team, league=league,
                position=position, season=season,
                metrics_summary=metrics_summary,
                role_scores=role_scores,
                club_narrative=club_narrative,
                bio_context=bio_context,
                tm_data=tm_data, sofifa_data=sofifa_data,
                params=params,
            )

        st.markdown("**📝 Scout Report:**")
        st.markdown(
            f"<div style='background:#0f1628;border:1px solid #1e2d4a;border-radius:12px;"
            f"padding:16px 20px;color:#e8ecff;font-size:14.5px;line-height:1.8;'>{report}</div>",
            unsafe_allow_html=True,
        )

        chief_candidates.append({
            "player": player_name, "team": team, "league": league, "season": season,
            "impact": impact, "best_role": best_role, "best_role_score": best_score,
            "report": report,
        })
        st.markdown("---")

    # ── CANDIDATES 4-10: Listed only ─────────────────────────────────────────
    remaining = df_ranked.iloc[3:]
    if not remaining.empty:
        st.subheader(f"📋 Further Candidates (#{len(df_ranked.head(3))+1}–{len(df_ranked)})")
        st.caption("Statistical shortlist only — review manually if top 3 don't progress.")

        list_rows = []
        for _, row in remaining.iterrows():
            pos_str = str(row.get("Position", ""))
            rk = get_role_key(pos_str)
            q_pool = df_players[
                (df_players["Position"].astype(str).apply(lambda p: get_role_key(p) == rk)) &
                (df_players["League"] == str(row.get("League", "")))
            ]
            if len(q_pool) < 10:
                q_pool = df_players[df_players["Position"].astype(str).apply(lambda p: get_role_key(p) == rk)]

            qs = compute_role_scores(row, q_pool, pos_str)
            br = max(qs, key=qs.get) if qs else "—"
            bs = qs.get(br, 0)

            age_v = row.get("Age", "")
            mins_v = row.get("Minutes played", "")
            mv_v = row.get("Market value", "")

            list_rows.append({
                "Player": str(row.get("Player", "")),
                "Team": str(row.get("Team", "")),
                "League": str(row.get("League", "")),
                "Season": str(row.get("_season", "2025/26")),
                "Pos": pos_str.split(",")[0].strip().upper(),
                "Age": int(float(age_v)) if pd.notna(age_v) and str(age_v) not in ("", "nan") else "—",
                "Mins": int(float(mins_v)) if pd.notna(mins_v) and str(mins_v) not in ("", "nan") else "—",
                "MV": format_market_value(mv_v),
                "Best Role": f"{br} ({bs:.0f})",
                "Impact": f"{row.get('_impact_score', 0):.0f}",
            })

        if list_rows:
            st.dataframe(pd.DataFrame(list_rows), use_container_width=True, hide_index=True)

    # ── Chief Scout Summary ───────────────────────────────────────────────────
    if chief_candidates:
        st.subheader("🏆 Chief Scout Summary")
        with st.spinner("Writing chief scout recommendation..."):
            chief = generate_chief_scout_summary(api_key, chief_candidates, club_narrative, params)

        if chief:
            st.markdown(
                f"<div style='background:#0a1628;border:2px solid #ef4444;border-radius:12px;"
                f"padding:20px 24px;color:#f5f5f5;font-size:15px;line-height:1.8;'>"
                f"<div style='font-size:10px;font-weight:900;letter-spacing:.18em;color:#ef4444;"
                f"margin-bottom:10px;'>CHIEF SCOUT RECOMMENDATION</div>{chief}</div>",
                unsafe_allow_html=True,
            )

        # Download
        st.markdown("---")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        output = f"AI SCOUT REPORT\nGenerated: {ts}\nQuery: {query}\n\n"
        output += f"CLUB CONTEXT:\n{club_narrative}\n\n{'='*60}\n\nTOP 3 CANDIDATES:\n"
        for i, c in enumerate(chief_candidates):
            output += f"\n#{i+1} {c['player']} ({c['team']}, {c['league']}, {c['season']})\n"
            output += f"Impact: {c['impact']:.0f} | {c['best_role']}: {c['best_role_score']:.0f}\n"
            output += f"{c['report']}\n"
        if chief:
            output += f"\n{'='*60}\nCHIEF SCOUT:\n{chief}\n"

        st.download_button(
            "⬇️ Download Scout Report (.txt)",
            data=output,
            file_name=f"scout_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )

elif run_btn and not api_key:
    st.error("Enter your Claude API key in the sidebar.")
elif run_btn and df_players.empty:
    st.error("Upload player CSV files in the sidebar.")