# 06_AI_Scout.py — AI Scout Assistant
# Drop this file into your /pages/ folder alongside your other position pages.
# Requires: pip install anthropic

import os
import re
import io
import json
import unicodedata
import requests

import numpy as np
import pandas as pd
import streamlit as st

# ── Anthropic ──────────────────────────────────────────────────────────────────
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
# CSS  — matches your existing dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
:root{--bg:#0b0f1f;--card:#111827;--stroke:#1f2937;--text:#f1f5f9;--muted:#9fb0c8;--accent:#7c3aed;}
.stApp{background:var(--bg);}
.block-container{max-width:1100px;padding-top:40px;}

/* page title */
.scout-title{font-weight:900;font-size:clamp(30px,4vw,46px);color:var(--text);margin:0;}
.scout-sub{color:var(--muted);margin:4px 0 28px 0;font-size:15px;}

/* search box label */
.search-label{color:var(--muted);font-size:13px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px;}

/* club profile card */
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

/* fm badge */
.fm-row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;}
.fm-pill{background:#0f3460;border:1px solid #1e4d8c;border-radius:6px;
  padding:4px 10px;font-size:12px;color:#93c5fd;}
.fm-pill span{font-weight:700;}

/* warning / info */
.warn-box{background:#1c1208;border:1px solid #92400e;border-radius:10px;
  padding:12px 16px;color:#fbbf24;font-size:13px;margin-bottom:16px;}
.info-box{background:#0c1a2e;border:1px solid #1e3a5f;border-radius:10px;
  padding:12px 16px;color:#93c5fd;font-size:13px;margin-bottom:16px;}
</style>
""", unsafe_allow_html=True)

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
    'Austria 1.':63.33,'Morocco 1.':63.14,'Korea 1.':62.75,'Paraguay 1.':62.55,'France 2.':64.00,
    'England 3.':61.96,'Romania 1.':61.76,'Scotland 1.':61.76,'Algeria 1.':61.57,'Uruguay 1.':60.39,
    'Chile 1.':59.80,'Egypt 1.':59.22,'Israel 1.':58.43,'Brazil 2.':58.04,'Slovenia 1.':57.45,
    'Bolivia 1.':57.25,'Slovakia 1.':56.47,'Azerbaijan 1.':56.47,'South Africa 1.':56.27,
    'UAE 1.':55.49,'Costa Rica 1.':54.90,'Peru 1.':54.90,'Germany 3.':54.51,'Ukraine 1.':54.31,
    'Spain 3.':54.31,'Portugal 2.':53.14,'Bulgaria 1.':53.14,'Australia 1.':52.75,
    'Serbia 1.':52.16,'Albania 1.':51.96,'Bosnia 1.':51.76,'Kosovo 1.':51.37,
    'Japan 2.':50.98,'England 4.':50.78,'Ireland 1.':50.59,'Russia 1.':62.41,
    'Kazakhstan 1.':50.39,'Nigeria 1.':50.00,'France 3.':49.61,'Tunisia 1.':49.22,
    'Venezuela 1.':48.63,'Belgium 2.':48.43,'Finland 1.':48.43,'Armenia 1.':47.84,
    'Georgia 1.':47.65,'Switzerland 2.':46.47,'Qatar 1.':46.27,'Uzbekistan 1.':46.27,
    'Poland 2.':46.27,'Iceland 1.':46.08,'Norway 2.':45.88,'Sweden 2.':45.69,
    'North Macedonia 1.':44.71,'Turkey 2.':44.51,'Korea 2.':43.53,'Czech 2.':43.33,
    'Brazil 3.':43.14,'Lithuania 1.':42.35,'Netherlands 2.':42.16,'Malta 1.':41.96,
    'Italy 3.':45.00,'Denmark 2.':40.39,'Moldova 1.':40.39,'USA 2.':40.00,
    'Latvia 1.':40.00,'Montenegro 1.':39.80,'Scotland 2.':38.63,'Canada 1.':38.24,
    'Austria 2.':38.24,'Israel 2.':38.04,'England 7.':37.25,'Germany 4.':35.29,
    'Portugal 3.':35.29,'England 5.':33.33,'Estonia 1.':40.00,'England 9.':31.37,
    'Northern Ireland 1.':30.98,'Serbia 2.':30.39,'Denmark 3.':29.41,'Sweden 3.':29.41,
    'Slovenia 2.':28.82,'Slovakia 2.':28.24,'Greece 2.':27.06,'Wales 1.':26.67,
    'USA 3.':22.55,'Scotland 3.':20.00,'England 6.':16.08,'England 8.':15.69,
    'England 10.':3.92,'Estonia 2.':3.00,'Ireland 2.':10.00,'Faroe Islands 1.':35.02,
    'Cyprus 1.':60.00,
}

# Key metrics per position
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
}

# FM physical attributes we'll look for
FM_PHYSICAL_ATTRS = ["pace","acceleration","strength","jumping_reach","stamina",
                      "work_rate","natural_fitness","height"]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_team_stats(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

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
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

def percentile_in_pool(series: pd.Series, value: float) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return 0
    return int((s <= float(value)).mean() * 100)

def describe_ppda(ppda: float) -> str:
    if ppda < 7:
        return "Very High Press"
    if ppda < 9:
        return "High Press"
    if ppda < 11:
        return "Moderate Press"
    if ppda < 14:
        return "Low Block"
    return "Deep Block"

def describe_possession(poss: float) -> str:
    if poss >= 58:
        return "Dominant Possession"
    if poss >= 53:
        return "Possession-Based"
    if poss >= 47:
        return "Balanced"
    if poss >= 42:
        return "Transitional"
    return "Direct / Counter"

def describe_directness(long_p90: float) -> str:
    if long_p90 >= 55:
        return "Very Direct"
    if long_p90 >= 45:
        return "Direct"
    if long_p90 >= 35:
        return "Mixed"
    return "Short / Build-Up"

# ══════════════════════════════════════════════════════════════════════════════
# FMINSIDE FETCH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fminside_player(player_name: str, team_name: str) -> dict | None:
    """
    Searches FMInside for a player by name and returns FM attributes.
    Returns None if not found or fetch fails.
    """
    try:
        surname = _slug(_surname(player_name))
        full_slug = _slug(player_name)

        search_url = f"https://fminside.net/players?search={requests.utils.quote(player_name)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(search_url, headers=headers, timeout=8)

        if r.status_code != 200:
            return None

        # Pull player links from the page
        links = re.findall(r'href="(/players/\d+/[^"]+)"', r.text)
        if not links:
            return None

        # Pick best matching link
        best_link, best_score = None, 0.0
        for lnk in links[:8]:
            slug_part = lnk.split("/")[-1].replace("-", " ")
            sc = _similar(_slug(slug_part), full_slug)
            sn_sc = _similar(_slug(slug_part.split()[-1] if slug_part.split() else ""), surname)
            combined = max(sc, sn_sc)
            if combined > best_score:
                best_score = combined
                best_link = lnk

        if best_score < 0.45 or not best_link:
            return None

        player_url = f"https://fminside.net{best_link}"
        rp = requests.get(player_url, headers=headers, timeout=8)
        if rp.status_code != 200:
            return None

        html = rp.text
        attrs = {}

        # Extract key FM attributes from page HTML
        fm_fields = {
            "pace": r'Pace[^<]*<[^>]+>\s*(\d+)',
            "acceleration": r'Acceleration[^<]*<[^>]+>\s*(\d+)',
            "strength": r'Strength[^<]*<[^>]+>\s*(\d+)',
            "jumping_reach": r'Jumping Reach[^<]*<[^>]+>\s*(\d+)',
            "stamina": r'Stamina[^<]*<[^>]+>\s*(\d+)',
            "work_rate": r'Work Rate[^<]*<[^>]+>\s*([^<]+)',
            "height": r'(\d{3})\s*cm',
            "natural_fitness": r'Natural Fitness[^<]*<[^>]+>\s*(\d+)',
        }

        for attr, pattern in fm_fields.items():
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                attrs[attr] = int(val) if val.isdigit() else val

        attrs["_url"] = player_url
        attrs["_match_score"] = round(best_score, 2)
        return attrs if len(attrs) > 2 else None

    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFERMARKT FETCH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_transfermarkt_value(player_name: str, team_name: str) -> dict | None:
    """
    Searches Transfermarkt for a player's market value.
    Returns dict with 'value_str', 'value_eur', 'contract', 'nationality' or None.
    Note: Transfermarkt ToS prohibits automated scraping. Use at low volume for
    internal/research purposes only.
    """
    try:
        slug = player_name.lower().strip().replace(" ", "-")
        slug = re.sub(r"[^a-z0-9\-]", "", slug)

        search_url = f"https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query={requests.utils.quote(player_name)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-GB,en;q=0.9",
        }
        r = requests.get(search_url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None

        html = r.text

        # Extract player profile links
        player_links = re.findall(
            r'href="(/[^/]+/profil/spieler/\d+)"[^>]*>([^<]+)</a>', html
        )

        if not player_links:
            return None

        best_link, best_score = None, 0.0
        full_slug = _slug(player_name)
        for lnk, name_text in player_links[:6]:
            sc = _similar(_slug(name_text), full_slug)
            if sc > best_score:
                best_score = sc
                best_link = lnk

        if best_score < 0.45 or not best_link:
            return None

        profile_url = f"https://www.transfermarkt.co.uk{best_link}"
        rp = requests.get(profile_url, headers=headers, timeout=10)
        if rp.status_code != 200:
            return None

        phtml = rp.text

        # Market value
        mv_match = re.search(
            r'marketValueDevelopment.*?(\d+[\.,]?\d*)\s*(k|m|Th\.|Mill\.)',
            phtml, re.IGNORECASE | re.DOTALL
        )
        value_eur = None
        value_str = "—"

        # Try alternate pattern
        mv2 = re.search(r'class="[^"]*marketValue[^"]*"[^>]*>\s*€\s*([\d,\.]+)\s*(k|m|Th\.?|Mill\.?)',
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

        # Contract expiry
        contract_match = re.search(r'Contract expires[^:]*:\s*([A-Za-z]+ \d{4}|\d{2}/\d{2}/\d{4})',
                                   phtml, re.IGNORECASE)
        contract = contract_match.group(1).strip() if contract_match else "—"

        return {
            "value_str": value_str,
            "value_eur": value_eur,
            "contract": contract,
            "_url": profile_url,
            "_match_score": round(best_score, 2),
        }

    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# TEAM PROFILE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_team_profile(team_name: str, team_df: pd.DataFrame) -> dict | None:
    """Find team in team stats CSV and build a tactical profile dict."""
    if team_df is None or team_df.empty:
        return None

    # Flexible name match
    mask = team_df["Team"].astype(str).str.lower() == team_name.lower().strip()
    if not mask.any():
        # Try partial
        mask = team_df["Team"].astype(str).str.lower().str.contains(
            team_name.lower().strip(), na=False
        )
    if not mask.any():
        return None

    row = team_df[mask].iloc[0]

    def safe(col, default=0.0):
        try:
            return float(row[col])
        except Exception:
            return default

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
        "pass_acc": round(safe("Pass Accuracy %"), 1),
        "long_passes_p90": round(safe("Long Passes p90"), 1),
        "long_pass_acc": round(safe("Long Pass Accuracy %"), 1),
        "prog_passes_p90": round(safe("Progressive Passes p90"), 1),
        "prog_runs_p90": round(safe("Progressive Runs p90"), 1),
        "crosses_p90": round(safe("Crosses p90"), 1),
        "aerial_p90": round(safe("Aerial Duels p90"), 1),
        "aerial_won_pct": round(safe("Aerial Duels Won %"), 1),
        "def_duels_p90": round(safe("Defensive Duels p90"), 1),
        "shots_p90": round(safe("Shots p90"), 1),
        "touches_box_p90": round(safe("Touches in Box p90"), 1),
        # Derived descriptors
        "press_style": describe_ppda(safe("PPDA")),
        "poss_style": describe_possession(safe("Possession %")),
        "directness": describe_directness(safe("Long Passes p90")),
    }

    # Percentile ranks vs all teams in same league
    league_teams = team_df[team_df["League"] == row["League"]]

    def pct_rank(col):
        try:
            vals = pd.to_numeric(league_teams[col], errors="coerce").dropna()
            v = float(row[col])
            # For PPDA, lower is better (more pressing)
            if col == "PPDA":
                return int((vals >= v).mean() * 100)
            return int((vals <= v).mean() * 100)
        except Exception:
            return 50

    profile["ppda_pct"] = pct_rank("PPDA")
    profile["xg_pct"] = pct_rank("xG p90")
    profile["aerial_pct"] = pct_rank("Aerial Duels p90")
    profile["possession_pct"] = pct_rank("Possession %")
    profile["crosses_pct"] = pct_rank("Crosses p90")
    profile["prog_runs_pct"] = pct_rank("Progressive Runs p90")

    return profile

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER EXTRACTION (Claude)
# ══════════════════════════════════════════════════════════════════════════════

def extract_parameters(client, query: str) -> dict:
    """Ask Claude to extract structured search parameters from natural language."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system="""You are a football data analyst. Extract search parameters from 
a scout query and return ONLY valid JSON. Use these exact field names:
{
  "club": "Salford City",           // club name mentioned, null if none
  "position_prefixes": ["CF"],      // list of Wyscout position codes e.g. CF, LW, RW, AMF, CMF, DMF, CB, RB, LB
  "max_age": 23,                    // null if not specified
  "min_age": null,                  // null if not specified
  "min_minutes": 500,               // default 500
  "leagues": ["England 4.", "England 3.", "England 2."],  // Wyscout format e.g. "England 4." — infer from context, null = all
  "max_market_value_m": 1.0,        // in millions EUR, null if not specified
  "foot": null,                     // "left", "right", or null
  "key_style_traits": ["target man", "aerial", "pressing"], // tactical/style keywords
  "physical_traits": ["tall", "fast", "strong"],  // physical keywords for FM lookup
  "priority_metrics": ["xG per 90", "Aerial duels won, %"],  // top 3 metrics to prioritise
  "fetch_fminside": true,           // true if physical traits mentioned
  "fetch_transfermarkt": true       // true if value/contract mentioned
}
Return ONLY the JSON, no markdown, no explanation.""",
        messages=[{"role": "user", "content": query}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def filter_candidates(df: pd.DataFrame, params: dict, team_profile: dict | None) -> pd.DataFrame:
    """Apply structured filters to the merged player dataframe."""
    pool = df.copy()

    # Numeric coercions
    for col in ["Minutes played", "Age", "Market value"]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce")

    # Position filter
    prefixes = [p.upper().strip() for p in (params.get("position_prefixes") or [])]
    if prefixes:
        pool = pool[pool["Position"].astype(str).str.upper().apply(
            lambda p: any(p.startswith(px) for px in prefixes)
        )]

    # Age
    if params.get("max_age"):
        pool = pool[pool["Age"] <= float(params["max_age"])]
    if params.get("min_age"):
        pool = pool[pool["Age"] >= float(params["min_age"])]

    # Minutes
    min_mins = float(params.get("min_minutes") or 500)
    pool = pool[pool["Minutes played"] >= min_mins]

    # Leagues
    leagues = params.get("leagues")
    if leagues:
        pool = pool[pool["League"].isin(leagues)]

    # Foot
    if params.get("foot"):
        pool = pool[pool.get("Foot", pd.Series(dtype=str)).astype(str).str.lower().str.startswith(
            params["foot"][0].lower(), na=False
        )]

    # Market value (CSV column, rough filter before Transfermarkt)
    if params.get("max_market_value_m") and "Market value" in pool.columns:
        pool = pool[pool["Market value"] <= float(params["max_market_value_m"]) * 1_000_000]

    return pool

def score_candidates(pool: pd.DataFrame, params: dict, team_profile: dict | None,
                     full_pool: pd.DataFrame) -> pd.DataFrame:
    """Score and rank candidates."""
    if pool.empty:
        return pool

    prefixes = [p.upper().strip() for p in (params.get("position_prefixes") or ["CF"])]
    primary_pos = prefixes[0] if prefixes else "CF"

    # Get relevant metrics for this position
    default_metrics = POSITION_METRICS.get(primary_pos, POSITION_METRICS["CF"])
    priority = params.get("priority_metrics") or []
    all_metrics = list(dict.fromkeys(priority + default_metrics))  # priority first, no dupes
    all_metrics = [m for m in all_metrics if m in full_pool.columns]

    # If team profile exists, weight metrics that match their style
    weights = {m: 1.0 for m in all_metrics}
    if team_profile:
        # High pressing team → weight defensive contribution
        if team_profile["ppda"] < 9:
            for m in ["Defensive duels per 90", "PAdj Interceptions"]:
                if m in weights:
                    weights[m] = 1.8
        # Direct team with high aerial → weight aerial metrics
        if team_profile["long_passes_p90"] > 45 and team_profile["aerial_p90"] > 50:
            for m in ["Aerial duels per 90", "Aerial duels won, %"]:
                if m in weights:
                    weights[m] = 2.0
        # Low possession → weight progressive carries
        if team_profile["possession"] < 47:
            for m in ["Progressive runs per 90", "Dribbles per 90"]:
                if m in weights:
                    weights[m] = 1.5

    # Style traits from query
    style_traits = [s.lower() for s in (params.get("key_style_traits") or [])]
    if any(t in style_traits for t in ["target man", "aerial", "header"]):
        for m in ["Aerial duels per 90", "Aerial duels won, %"]:
            if m in weights:
                weights[m] = 2.5
    if any(t in style_traits for t in ["dribbler", "dribble", "carries", "direct"]):
        for m in ["Dribbles per 90", "Successful dribbles, %", "Progressive runs per 90"]:
            if m in weights:
                weights[m] = 2.0
    if any(t in style_traits for t in ["pressing", "press", "high press"]):
        for m in ["Defensive duels per 90", "PAdj Interceptions"]:
            if m in weights:
                weights[m] = 2.0
    if any(t in style_traits for t in ["creative", "playmaker", "link-up"]):
        for m in ["xA per 90", "Key passes per 90", "Smart passes per 90"]:
            if m in weights:
                weights[m] = 2.0

    # Compute weighted percentile score
    scored = pool.copy()
    score_acc = np.zeros(len(scored))
    weight_acc = 0.0

    for m in all_metrics:
        if m not in full_pool.columns:
            continue
        vals_full = pd.to_numeric(full_pool[m], errors="coerce").dropna()
        if vals_full.empty:
            continue
        candidate_vals = pd.to_numeric(scored[m], errors="coerce")
        pcts = candidate_vals.apply(
            lambda v: (vals_full <= v).mean() * 100 if pd.notna(v) else 50.0
        ).values
        w = weights.get(m, 1.0)
        score_acc += pcts * w
        weight_acc += w

    if weight_acc > 0:
        scored["_scout_score"] = score_acc / weight_acc
    else:
        scored["_scout_score"] = 50.0

    return scored.sort_values("_scout_score", ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE — CLUB PROFILE NARRATIVE
# ══════════════════════════════════════════════════════════════════════════════

def generate_club_narrative(client, team_profile: dict, query: str) -> str:
    prompt = f"""You are a chief scout. Write a brief tactical profile of {team_profile['team']} 
based on this data. Be specific and concise — 3-4 sentences max.

Data:
- League: {team_profile['league']}
- Record: {team_profile['wins']}W {team_profile['draws']}D {team_profile['losses']}L ({team_profile['points']} pts, xPts: {team_profile['xpoints']})
- Pressing: PPDA {team_profile['ppda']} ({team_profile['press_style']}) — {team_profile['ppda_pct']}th percentile in league
- Possession: {team_profile['possession']}% ({team_profile['poss_style']})
- Directness: {team_profile['long_passes_p90']} long passes p90 ({team_profile['directness']})
- xG p90: {team_profile['xg_p90']} (attack {team_profile['xg_pct']}th pct in league)
- xGA p90: {team_profile['xga_p90']} (defensive)
- Aerial duels p90: {team_profile['aerial_p90']} — {team_profile['aerial_pct']}th percentile
- Progressive runs p90: {team_profile['prog_runs_p90']} — {team_profile['prog_runs_pct']}th percentile
- Average squad age: {team_profile['avg_age']}
- Scout query context: {query}

Write the profile now. End with one sentence on what type of player would suit them based on their style."""

    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text.strip()

# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE — MINI SCOUT REPORT PER CANDIDATE
# ══════════════════════════════════════════════════════════════════════════════

def generate_mini_report(client, player: pd.Series, params: dict,
                         team_profile: dict | None, full_pool: pd.DataFrame,
                         fm_data: dict | None, tm_data: dict | None) -> str:
    """Generate a mini professional scout report for one candidate."""

    # Build stats block with percentiles
    prefixes = [p.upper().strip() for p in (params.get("position_prefixes") or ["CF"])]
    primary_pos = prefixes[0] if prefixes else "CF"
    metrics = POSITION_METRICS.get(primary_pos, POSITION_METRICS["CF"])
    metrics = [m for m in metrics if m in full_pool.columns]

    stats_lines = []
    for m in metrics:
        val = pd.to_numeric(player.get(m), errors="coerce")
        if pd.isna(val):
            continue
        peer_vals = pd.to_numeric(full_pool[m], errors="coerce").dropna()
        pct = int((peer_vals <= val).mean() * 100) if not peer_vals.empty else 50
        stats_lines.append(f"  {m}: {val:.2f} [{pct}th pct]")

    stats_block = "\n".join(stats_lines) if stats_lines else "Stats unavailable."

    # FM block
    fm_block = "Not available."
    if fm_data:
        parts = []
        for attr in ["pace","acceleration","strength","jumping_reach","stamina","natural_fitness"]:
            if attr in fm_data:
                parts.append(f"{attr.replace('_',' ').title()}: {fm_data[attr]}/20")
        if "height" in fm_data:
            parts.append(f"Height: {fm_data['height']}cm")
        fm_block = ", ".join(parts) if parts else "Partial data only."

    # TM block
    tm_block = "Not fetched."
    if tm_data:
        tm_block = f"Market Value: {tm_data.get('value_str','—')}, Contract: {tm_data.get('contract','—')}"

    # Club context
    club_ctx = "No club context provided."
    if team_profile:
        club_ctx = (f"{team_profile['team']} ({team_profile['league']}) — "
                    f"{team_profile['press_style']}, {team_profile['poss_style']}, "
                    f"{team_profile['directness']}. "
                    f"Aerial duels p90: {team_profile['aerial_p90']} "
                    f"({team_profile['aerial_pct']}th pct in league).")

    contract_raw = str(player.get("Contract expires", "—"))
    mv_raw = fmt_mv(player.get("Market value"))

    prompt = f"""Write a concise professional scouting report in 4-5 sentences.
Focus on: fit for the requesting club, statistical standouts, physical profile, and one risk.
Do NOT use bullet points. Write in flowing scouting prose.

Player: {player.get('Player','—')}
Club: {player.get('Team','—')} ({player.get('League','—')})
Age: {player.get('Age','—')} | Position: {player.get('Position','—')} | Foot: {player.get('Foot','—')}
Minutes: {player.get('Minutes played','—')} | Contract: {contract_raw} | Value: {mv_raw}

Performance stats (pool percentiles):
{stats_block}

FM Physical Attributes:
{fm_block}

Transfermarkt:
{tm_block}

Requesting club context:
{club_ctx}

Scout query: {params.get('_raw_query','—')}

Write the report now:"""

    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text.strip()

# ══════════════════════════════════════════════════════════════════════════════
# STAT PILLS RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_stat_pills(player: pd.Series, params: dict, full_pool: pd.DataFrame) -> str:
    prefixes = [p.upper().strip() for p in (params.get("position_prefixes") or ["CF"])]
    primary_pos = prefixes[0] if prefixes else "CF"
    metrics = POSITION_METRICS.get(primary_pos, POSITION_METRICS["CF"])[:6]
    metrics = [m for m in metrics if m in full_pool.columns]

    pills = []
    for m in metrics:
        val = pd.to_numeric(player.get(m), errors="coerce")
        if pd.isna(val):
            continue
        peer_vals = pd.to_numeric(full_pool[m], errors="coerce").dropna()
        pct = int((peer_vals <= val).mean() * 100) if not peer_vals.empty else 50
        short_label = (m.replace(" per 90","p90")
                        .replace("Non-penalty goals","NP Goals")
                        .replace("Accurate ","")
                        .replace(" won, %"," Win%")
                        .replace("PAdj Interceptions","PAdj Int"))
        pills.append(
            f"<div class='pill'><span class='plab'>{short_label}</span>"
            f"<span class='pval'>{val:.2f} <span style='color:#6b7280;font-size:10px'>({pct}th)</span></span></div>"
        )
    return "".join(pills)

def render_fm_pills(fm_data: dict) -> str:
    if not fm_data:
        return ""
    parts = []
    for attr in ["pace","acceleration","strength","jumping_reach","stamina"]:
        if attr in fm_data:
            label = attr.replace("_"," ").title()
            val = fm_data[attr]
            color = "#22c55e" if val >= 14 else ("#f59e0b" if val >= 10 else "#ef4444")
            parts.append(f"<div class='fm-pill'>{label}: <span style='color:{color}'>{val}</span>/20</div>")
    if "height" in fm_data:
        parts.append(f"<div class='fm-pill'>Height: <span>{fm_data['height']}cm</span></div>")
    return "<div class='fm-row'>" + "".join(parts) + "</div>"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — API KEY + CSV SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🤖 AI Scout")
    st.markdown("---")

    # API Key
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Get yours at console.anthropic.com",
        placeholder="sk-ant-api03-..."
    )

    st.markdown("---")

    # CSV file selector — scans cwd for WORLD*.csv files
    import glob
    from pathlib import Path

    csv_candidates = sorted(
        glob.glob(str(Path.cwd() / "WORLD*.csv")) +
        glob.glob(str(Path.cwd().parent / "WORLD*.csv"))
    )
    csv_labels = [Path(p).name for p in csv_candidates]

    if csv_labels:
        selected_csvs = st.multiselect(
            "Player datasets (select one or more)",
            options=csv_labels,
            default=csv_labels[:1] if csv_labels else [],
            help="Select multiple CSVs to search across all of them simultaneously"
        )
    else:
        st.info("No WORLD*.csv files found in app directory. Upload below.")
        selected_csvs = []

    # Manual upload fallback
    uploaded_files = st.file_uploader(
        "Or upload CSV(s)",
        type=["csv"],
        accept_multiple_files=True
    )

    st.markdown("---")

    # Team stats CSV
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
        st.info("No team stats CSV found. Club context will be unavailable.")
        uploaded_team_stats = st.file_uploader("Upload team stats CSV", type=["csv"])
    
    st.markdown("---")
    st.caption("Top N results")
    top_n = st.slider("Candidates to return", 3, 15, 8)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_merge(csv_paths: tuple, upload_bytes: tuple) -> pd.DataFrame:
    """Load and merge multiple player CSVs."""
    frames = []
    for path in csv_paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass
    for name, data in zip(*upload_bytes) if upload_bytes[0] else ([], []):
        try:
            frames.append(pd.read_csv(io.BytesIO(data)))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    # Deduplicate by player+team, keep first occurrence
    if "Player" in merged.columns and "Team" in merged.columns:
        merged = merged.drop_duplicates(subset=["Player","Team"], keep="first")
    return merged

# Resolve paths
csv_paths_to_load = tuple(
    str(Path.cwd() / name) for name in selected_csvs
    if (Path.cwd() / name).exists()
)
upload_names = tuple(f.name for f in (uploaded_files or []))
upload_data = tuple(f.getvalue() for f in (uploaded_files or []))

player_df = load_and_merge(csv_paths_to_load, (upload_names, upload_data))

# Load team stats
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
    'The AI will analyse your database, pull FM attributes and Transfermarkt values, '
    'and reason over candidates like a senior scout.</div>',
    unsafe_allow_html=True
)

# Status checks
if player_df.empty:
    st.markdown(
        "<div class='warn-box'>⚠️ No player data loaded. Select a WORLD*.csv in the sidebar.</div>",
        unsafe_allow_html=True
    )
elif not api_key_input:
    st.markdown(
        "<div class='warn-box'>⚠️ Enter your Anthropic API key in the sidebar to use the AI Scout.</div>",
        unsafe_allow_html=True
    )
else:
    sources = []
    if selected_csvs:
        sources.append(f"{len(selected_csvs)} CSV{'s' if len(selected_csvs)>1 else ''}")
    if uploaded_files:
        sources.append(f"{len(uploaded_files)} uploaded")
    n_players = len(player_df)
    n_leagues = player_df["League"].nunique() if "League" in player_df.columns else 0

    info_msg = f"✅ {n_players:,} players · {n_leagues} leagues loaded"
    if team_df is not None:
        info_msg += f" · Team stats: {len(team_df)} teams"
    else:
        info_msg += " · ⚠️ No team stats loaded (club context unavailable)"

    st.markdown(f"<div class='info-box'>{info_msg}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH BOX
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='search-label'>Scout Request</div>", unsafe_allow_html=True)

query = st.text_area(
    label="scout_query",
    label_visibility="collapsed",
    placeholder=(
        "e.g. Salford City in League Two have £1m to spend on a striker. "
        "Find me the best U23 CFs in the EFL — a tall target man with high xG and good dribbles. "
        "Athletically fast with a Transfermarkt value under £1m."
    ),
    height=100,
    key="scout_query_input"
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    run = st.button("🔍 Scout", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=False):
        st.session_state.pop("scout_results", None)
        st.rerun()

# Example queries
with st.expander("💡 Example queries"):
    examples = [
        "Brentford need a left-footed creative winger, U25, max £15m. Someone who dribbles well and creates chances. Preferably from a top 5 European league.",
        "Find me a ball-playing CB for a Championship club. Must be composed on the ball, good in the air, under 28. Budget £5m. Check FM for pace and strength.",
        "A defensive midfielder for a mid-table Bundesliga side — high pressing, wins duels, good passer. Under 26, max €8m Transfermarkt value.",
        "Salford City, League Two, £1m budget. U23 CF, tall target man, high xG and dribbles, fast. EFL leagues only.",
        "Brighton need an attacking midfielder — creative, progressive passer, high xA. U23, no budget cap, check FM attributes for technical ratings.",
    ]
    for ex in examples:
        if st.button(f"→ {ex[:80]}...", key=f"ex_{ex[:20]}"):
            st.session_state["scout_query_prefill"] = ex
            st.rerun()

# Pre-fill if example clicked
if "scout_query_prefill" in st.session_state and not query:
    query = st.session_state.pop("scout_query_prefill")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCOUT LOGIC
# ══════════════════════════════════════════════════════════════════════════════

if run and query.strip() and not player_df.empty and api_key_input:

    client = anthropic.Anthropic(api_key=api_key_input)

    # ── Step 1: Extract parameters ────────────────────────────────────────────
    with st.spinner("🧠 Reading your request..."):
        params = extract_parameters(client, query)
        params["_raw_query"] = query

    if not params:
        st.error("Couldn't parse your request. Try being more specific about position and league.")
        st.stop()

    # ── Step 2: Club profile ──────────────────────────────────────────────────
    team_profile = None
    club_name = params.get("club")

    if club_name and team_df is not None:
        with st.spinner(f"📊 Loading {club_name} tactical profile..."):
            team_profile = build_team_profile(club_name, team_df)

    # ── Step 3: Show club profile card ───────────────────────────────────────
    if team_profile:
        with st.spinner(f"✍️ Generating {club_name} profile narrative..."):
            club_narrative = generate_club_narrative(client, team_profile, query)

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
  <div class='report-text' style='margin-top:16px'>{club_narrative}</div>
</div>
""", unsafe_allow_html=True)

    elif club_name:
        st.markdown(
            f"<div class='warn-box'>⚠️ '{club_name}' not found in team stats. "
            f"Club context unavailable — results will be purely statistical.</div>",
            unsafe_allow_html=True
        )

    # ── Step 4: Filter and score candidates ──────────────────────────────────
    with st.spinner("🔎 Filtering player database..."):
        filtered = filter_candidates(player_df, params, team_profile)
        scored = score_candidates(filtered, params, team_profile, player_df)

    if scored.empty:
        st.warning("No players found matching those criteria. Try relaxing the filters.")
        st.stop()

    top_candidates = scored.head(top_n).copy()

    leagues_searched = params.get("leagues") or ["all loaded leagues"]
    leagues_str = ", ".join(leagues_searched) if isinstance(leagues_searched, list) else str(leagues_searched)

    st.markdown(f"""
<div class='info-box'>
  Found <strong>{len(scored):,}</strong> candidates matching filters across 
  <strong>{leagues_str}</strong>. Showing top {len(top_candidates)}.
</div>
""", unsafe_allow_html=True)

    # ── Step 5: Render each candidate ────────────────────────────────────────
    st.markdown(f"## 🎯 Top Candidates")

    for rank, (_, player) in enumerate(top_candidates.iterrows(), 1):
        player_name = str(player.get("Player", "Unknown"))
        team_name = str(player.get("Team", "—"))
        league = str(player.get("League", "—"))
        age = player.get("Age", "—")
        pos = str(player.get("Position", "—"))
        foot = str(player.get("Foot", "—"))
        minutes = player.get("Minutes played", "—")
        contract = str(player.get("Contract expires", "—"))
        mv = fmt_mv(player.get("Market value"))
        score = float(player.get("_scout_score", 0))

        # FM fetch
        fm_data = None
        if params.get("fetch_fminside") and player_name != "Unknown":
            with st.spinner(f"⚽ Fetching FM data for {player_name}..."):
                fm_data = fetch_fminside_player(player_name, team_name)

        # Transfermarkt fetch
        tm_data = None
        if params.get("fetch_transfermarkt") and player_name != "Unknown":
            with st.spinner(f"💰 Checking Transfermarkt for {player_name}..."):
                tm_data = fetch_transfermarkt_value(player_name, team_name)

        # Generate mini report
        with st.spinner(f"✍️ Writing report for {player_name}..."):
            mini_report = generate_mini_report(
                client, player, params, team_profile, player_df, fm_data, tm_data
            )

        # Stat pills HTML
        stat_pills_html = render_stat_pills(player, params, player_df)
        fm_pills_html = render_fm_pills(fm_data) if fm_data else ""

        # TM value display
        tm_value_display = ""
        if tm_data and tm_data.get("value_str") and tm_data["value_str"] != "—":
            tm_value_display = f" · TM: {tm_data['value_str']}"
            if tm_data.get("contract") and tm_data["contract"] != "—":
                tm_value_display += f" · Contract: {tm_data['contract']}"

        # Score bar colour
        bar_color = "#22c55e" if score >= 75 else ("#f59e0b" if score >= 55 else "#ef4444")

        st.markdown(f"""
<div class='cand-card'>
  <div class='cand-rank'>#{rank} · Score {score:.0f}/100</div>
  <div class='cand-name'>{player_name}</div>
  <div class='cand-meta'>
    {team_name} · {league} · {pos} · Age {age} · {foot} foot · 
    {int(minutes) if str(minutes).replace('.','').isdigit() else minutes} mins · 
    CSV MV: {mv}{tm_value_display}
  </div>
  <div class='stat-pills'>{stat_pills_html}</div>
  {fm_pills_html}
  <div class='report-text'>{mini_report}</div>
</div>
""", unsafe_allow_html=True)

    # ── Step 6: Summary recommendation ───────────────────────────────────────
    if len(top_candidates) >= 3:
        with st.spinner("📝 Writing overall recommendation..."):
            top3_summary = "\n".join([
                f"{i+1}. {str(r.get('Player','?'))} ({str(r.get('Team','?'))}, {str(r.get('League','?'))}, age {r.get('Age','?')})"
                for i, (_, r) in enumerate(top_candidates.head(3).iterrows())
            ])
            club_ctx_summary = (
                f"{team_profile['team']} — {team_profile['press_style']}, "
                f"{team_profile['poss_style']}, {team_profile['directness']}."
                if team_profile else "No club context."
            )

            summary_response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=350,
                messages=[{"role": "user", "content": f"""You are a chief scout presenting to a director of football.
Summarise your recommendation in 3-4 sentences. Name your top pick and why. Mention one value alternative.

Scout query: {query}
Club context: {club_ctx_summary}
Top 3 candidates:
{top3_summary}

Write the recommendation now:"""}]
            )

            summary_text = summary_response.content[0].text.strip()

        st.markdown(f"""
<div class='club-card' style='margin-top:24px;border-color:#7c3aed;'>
  <h3>🏆 Chief Scout Recommendation</h3>
  <div class='report-text' style='border-color:#7c3aed;margin-top:8px'>{summary_text}</div>
</div>
""", unsafe_allow_html=True)

elif run and not query.strip():
    st.warning("Please enter a scout request above.")
elif run and player_df.empty:
    st.error("No player data loaded. Select a CSV file in the sidebar.")
elif run and not api_key_input:
    st.error("Enter your Anthropic API key in the sidebar.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='color:#4b5563;font-size:12px;text-align:center'>"
    "AI Scout · Powered by Claude · Data: Wyscout CSV + FMInside + Transfermarkt · "
    "For internal scouting use only"
    "</div>",
    unsafe_allow_html=True
)