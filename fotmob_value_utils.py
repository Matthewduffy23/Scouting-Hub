# fotmob_value_utils.py
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Any, Dict, List, Tuple

import requests

# ------------------ helpers ------------------
def _slug(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _player_surname(player: str) -> str:
    p = (player or "").strip()
    if not p:
        return ""
    if "," in p:
        return p.split(",", 1)[0].strip()
    parts = p.split()
    return parts[-1].strip() if parts else ""

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _fotmob_team_id_from_url(team_url: str) -> str:
    m = re.search(r"/teams/(\d+)/", str(team_url or ""))
    return m.group(1) if m else ""

def _coerce_to_eur(v: Any) -> Optional[float]:
    """
    FotMob may store market value as number, dict, or string like '€12.5m'.
    Return EUR float or None.
    """
    if v is None:
        return None

    if isinstance(v, (int, float)):
        return float(v)

    if isinstance(v, dict):
        for k in ("value", "amount", "eur", "marketValue", "market_value"):
            if k in v:
                return _coerce_to_eur(v[k])
        return None

    s = str(v).strip().lower().replace(",", "")
    s = s.replace("€", "").replace("eur", "").strip()

    mult = 1.0
    if s.endswith("m"):
        mult = 1_000_000.0
        s = s[:-1]
    elif s.endswith("k"):
        mult = 1_000.0
        s = s[:-1]

    try:
        return float(s) * mult
    except Exception:
        return None

# ------------------ HTTP (reuse connection) ------------------
_SESSION = requests.Session()
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# ------------------ FotMob API ------------------
def _fotmob_team_json(team_id: str) -> Dict[str, Any]:
    url = f"https://www.fotmob.com/api/teams?id={team_id}"
    r = _SESSION.get(url, timeout=12, headers=_HEADERS)
    r.raise_for_status()
    return r.json() or {}

def _extract_squad_members(team_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    squad = team_json.get("squad")
    out: List[Dict[str, Any]] = []

    if isinstance(squad, list):
        for sec in squad:
            members = sec.get("members") or sec.get("players") or []
            if isinstance(members, list):
                out.extend([m for m in members if isinstance(m, dict)])

    elif isinstance(squad, dict):
        for k in ("members", "players"):
            members = squad.get(k)
            if isinstance(members, list):
                out.extend([m for m in members if isinstance(m, dict)])

        nested = squad.get("squad")
        if isinstance(nested, list):
            for sec in nested:
                members = sec.get("members") or sec.get("players") or []
                if isinstance(members, list):
                    out.extend([m for m in members if isinstance(m, dict)])

    return out

# ------------------ CACHES (IMPORTANT) ------------------
# team_id -> squad list
_TEAM_SQUAD_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# team_id -> indexes for faster matching
# (surname_slug -> list[members]), (fullname_slug -> member)
_TEAM_INDEX_CACHE: Dict[str, Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]] = {}


def _get_team_squad_cached(team_id: str) -> List[Dict[str, Any]]:
    if not team_id:
        return []
    if team_id in _TEAM_SQUAD_CACHE:
        return _TEAM_SQUAD_CACHE[team_id]

    try:
        tj = _fotmob_team_json(team_id)
        squad = _extract_squad_members(tj)
    except Exception:
        squad = []

    _TEAM_SQUAD_CACHE[team_id] = squad
    return squad


def _build_team_indexes(team_id: str) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    if team_id in _TEAM_INDEX_CACHE:
        return _TEAM_INDEX_CACHE[team_id]

    squad = _get_team_squad_cached(team_id)
    surname_map: Dict[str, List[Dict[str, Any]]] = {}
    full_map: Dict[str, Dict[str, Any]] = {}

    for m in squad:
        name = m.get("name") or m.get("playerName") or ""
        sn = _slug(_player_surname(name))
        fn = _slug(name)
        if sn:
            surname_map.setdefault(sn, []).append(m)
        if fn:
            full_map[fn] = m

    _TEAM_INDEX_CACHE[team_id] = (surname_map, full_map)
    return surname_map, full_map


# ------------------ PUBLIC: value resolver ------------------
def fotmob_market_value_eur(player: str, team: str) -> Optional[float]:
    """
    FotMob lookup by team + surname match.
    Returns EUR float if found, else None.
    """
    from team_fotmob_urls import get_fotmob_url  # your existing fotmob URL mapping

    team_url = get_fotmob_url(team)
    tid = _fotmob_team_id_from_url(team_url)
    if not tid:
        return None

    squad = _get_team_squad_cached(tid)
    if not squad:
        return None

    surname_map, _full_map = _build_team_indexes(tid)

    target_surname = _slug(_player_surname(player))
    target_full = _slug(player)

    best_member = None

    # 1) exact surname (fast via index)
    if target_surname and target_surname in surname_map:
        # if multiple share surname, try the one whose full name contains match
        candidates = surname_map[target_surname]
        if len(candidates) == 1:
            best_member = candidates[0]
        else:
            # prefer one whose name contains player string
            for m in candidates:
                name = m.get("name") or m.get("playerName") or ""
                if target_full and target_full in _slug(name):
                    best_member = m
                    break
            if best_member is None:
                best_member = candidates[0]

    # 2) full-name contains (fallback scan)
    if best_member is None and target_full:
        for m in squad:
            name = m.get("name") or m.get("playerName") or ""
            if target_full in _slug(name):
                best_member = m
                break

    # 3) fuzzy surname (fallback scan)
    if best_member is None and target_surname:
        best_score = 0.0
        best_pick = None
        for m in squad:
            name = m.get("name") or m.get("playerName") or ""
            sc = _similar(_slug(_player_surname(name)), target_surname)
            if sc > best_score:
                best_score = sc
                best_pick = m
        if best_score >= 0.86:
            best_member = best_pick

    if not best_member:
        return None

    # Try common FotMob fields for value
    for key in ("marketValue", "market_value", "value", "playerValue", "estimatedValue"):
        if key in best_member:
            v = _coerce_to_eur(best_member.get(key))
            if v is not None:
                return v

    return None


def best_available_market_value_eur(player: str, team: str, csv_market_value: Any) -> Optional[float]:
    """
    Use FotMob if possible, else fallback to CSV 'Market value'.
    """
    v = fotmob_market_value_eur(player, team)
    if v is not None and float(v) > 0:
        return float(v)

    try:
        if csv_market_value is None:
            return None
        fv = float(csv_market_value)
        return fv if fv > 0 else None
    except Exception:
        return None

