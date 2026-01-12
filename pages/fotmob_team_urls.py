# team_fotmob_map.py

import unicodedata

def _norm(s: str) -> str:
    if not s:
        return ""
    return (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
    )

# ✅ You maintain this manually
TEAM_TO_FOTMOB_SQUAD_URL = {
    "Swansea City": "https://www.fotmob.com/teams/10003/squad/swansea-city",
    "Liverpool": "https://www.fotmob.com/teams/8650/squad/liverpool",
    "Arsenal": "https://www.fotmob.com/teams/9825/squad/arsenal",
    "Manchester City": "https://www.fotmob.com/teams/8456/squad/manchester-city",
    "Real Madrid": "https://www.fotmob.com/teams/8633/squad/real-madrid",
}

# (optional) aliases so weird variants still resolve
ALIASES = {
    "man city": "Manchester City",
    "spurs": "Tottenham Hotspur",
}

# build a normalized lookup once
_NORM_LOOKUP = {_norm(k): v for k, v in TEAM_TO_FOTMOB_SQUAD_URL.items()}
_NORM_ALIASES = {_norm(k): _norm(v) for k, v in ALIASES.items()}

def get_fotmob_url(team: str) -> str:
    """
    Returns FotMob squad URL for a team, or "" if not found.
    """
    t = _norm(team)
    if not t:
        return ""
    # alias -> canonical
    if t in _NORM_ALIASES:
        t = _NORM_ALIASES[t]
    return _NORM_LOOKUP.get(t, "")

