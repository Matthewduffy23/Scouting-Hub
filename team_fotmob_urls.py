# team_fotmob_urls.py
import unicodedata

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.strip().lower().split())

FOTMOB_TEAM_URLS = {
    "Swansea City": "https://www.fotmob.com/teams/10003/squad/swansea-city",
    "Arsenal": "https://www.fotmob.com/teams/9825/squad/arsenal",
    "Stoke City": "https://www.fotmob.com/teams/10194/squad/stoke-city",
}

_FM = {_norm(k): v.strip() for k, v in FOTMOB_TEAM_URLS.items() if str(v).strip()}

def get_fotmob_url(team: str) -> str:
    return _FM.get(_norm(team), "")
