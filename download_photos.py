"""
download_photos.py
Run this once locally from your Scouting-Hub folder:
    python download_photos.py

It will:
1. Read all WORLD*.csv files to get every player + team
2. Fetch each team's squad from FotMob (your home IP, not blocked)
3. Match player surnames to get FotMob player IDs
4. Download photos and remove backgrounds
5. Save to photos/{normalized_player_name}.png
"""

import os
import re
import io
import csv
import sys
import time
import json
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import requests
from PIL import Image
from tqdm import tqdm

# ── optional background removal ──────────────────────────────────────────────
try:
    from rembg import remove as rembg_remove
    REMBG_OK = True
    print("✅ rembg loaded — backgrounds will be removed")
except Exception:
    REMBG_OK = False
    print("⚠️  rembg not available — photos saved as-is")

# ── config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PHOTOS_DIR   = SCRIPT_DIR / "photos"
CACHE_FILE   = SCRIPT_DIR / "photo_id_cache.json"   # saves player→fotmob_id
PHOTOS_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/html, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer":         "https://www.fotmob.com/",
}

DELAY_BETWEEN_TEAMS   = 1.2   # seconds — be polite
DELAY_BETWEEN_PLAYERS = 0.3

# ── helpers ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.strip().lower().split())


def _slug(s: str) -> str:
    """Aggressive normalisation for fuzzy matching."""
    s = _norm(s)
    return re.sub(r"[^a-z0-9]", "", s)


def _surname(name: str) -> str:
    """Last word of a name (handles 'van Dijk', 'de Bruyne' etc.)."""
    parts = name.strip().split()
    return parts[-1] if parts else ""


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def safe_filename(player: str) -> str:
    """Convert player name to a safe filename."""
    n = _norm(player)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = "_".join(n.split())
    return n or "unknown"


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )

# ── read CSVs ─────────────────────────────────────────────────────────────────

def read_players_from_csvs() -> dict[str, set[str]]:
    """Returns {team_name: {player_name, ...}}"""
    team_players: dict[str, set[str]] = {}
    csv_files = sorted(SCRIPT_DIR.glob("WORLD*.csv"))
    if not csv_files:
        print("❌  No WORLD*.csv files found in", SCRIPT_DIR)
        sys.exit(1)
    print(f"📂  Reading {len(csv_files)} CSV file(s)…")
    for csv_path in csv_files:
        try:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    player = (row.get("Player") or "").strip()
                    team   = (row.get("Team")   or "").strip()
                    if player and team:
                        team_players.setdefault(team, set()).add(player)
        except Exception as e:
            print(f"  ⚠️  Could not read {csv_path.name}: {e}")
    total_players = sum(len(v) for v in team_players.values())
    print(f"   Found {total_players:,} player rows across {len(team_players):,} teams")
    return team_players

# ── FotMob squad fetch ────────────────────────────────────────────────────────

def fotmob_team_id(team_url: str) -> str:
    m = re.search(r"/teams/(\d+)/", team_url or "")
    return m.group(1) if m else ""


def fetch_squad(team_id: str) -> list[dict]:
    """Fetch squad list from FotMob API."""
    url = f"https://www.fotmob.com/api/teams?id={team_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()

        squad: list[dict] = []
        raw = data.get("squad")

        if isinstance(raw, list):
            for section in raw:
                members = section.get("members") or section.get("players") or []
                squad.extend([m for m in members if isinstance(m, dict)])
        elif isinstance(raw, dict):
            for key in ("members", "players"):
                members = raw.get(key) or []
                squad.extend([m for m in members if isinstance(m, dict)])
            nested = raw.get("squad")
            if isinstance(nested, list):
                for section in nested:
                    members = section.get("members") or section.get("players") or []
                    squad.extend([m for m in members if isinstance(m, dict)])
        return squad
    except Exception:
        return []


def match_player_id(csv_name: str, squad: list[dict]) -> str:
    """
    Match abbreviated CSV name (e.g. 'V. van Dijk') to a FotMob squad member.
    Returns the FotMob player ID string, or "" if not found.
    """
    target_surname = _slug(_surname(csv_name))
    target_full    = _slug(csv_name)

    best_id    = ""
    best_score = 0.0

    for member in squad:
        full_name = member.get("name") or member.get("playerName") or ""
        pid = str(
            member.get("id") or member.get("playerId") or member.get("primaryId") or ""
        ).strip()
        if not pid or not pid.isdigit():
            continue

        slug_surname = _slug(_surname(full_name))
        slug_full    = _slug(full_name)

        # Exact surname match wins immediately
        if slug_surname == target_surname:
            if target_full in slug_full or slug_full in target_full:
                return pid          # perfect match
            best_id    = pid
            best_score = 1.0
            continue

        # Fuzzy fallback
        sc = _similar(slug_surname, target_surname)
        if sc > best_score:
            best_score = sc
            best_id    = pid

    return best_id if best_score >= 0.82 else ""

# ── photo download + background removal ──────────────────────────────────────

def download_and_save(player_name: str, fotmob_id: str) -> bool:
    filename = PHOTOS_DIR / f"{safe_filename(player_name)}.png"
    if filename.exists():
        return True   # already done

    photo_url = f"https://images.fotmob.com/image_resources/playerimages/{fotmob_id}.png"
    try:
        r = requests.get(photo_url, headers=HEADERS, timeout=10)
        if r.status_code != 200 or "image" not in r.headers.get("content-type", ""):
            return False

        img_bytes = r.content

        if REMBG_OK:
            try:
                img_bytes = rembg_remove(img_bytes)
            except Exception:
                pass   # keep original if rembg fails

        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        img.save(filename, "PNG")
        return True
    except Exception:
        return False

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # import here so the script works even without team_fotmob_urls in path
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from team_fotmob_urls import FOTMOB_TEAM_URLS, _norm as _norm_tm
    except ImportError:
        print("❌  team_fotmob_urls.py not found — make sure this script is in your Scouting-Hub folder")
        sys.exit(1)

    team_players = read_players_from_csvs()
    cache        = load_cache()

    # Only process teams that exist in both the CSV and team_fotmob_urls
    teams_to_process = []
    for team_name, players in team_players.items():
        url = FOTMOB_TEAM_URLS.get(team_name, "")
        if not url:
            # Try normalised lookup
            nm = _norm(team_name)
            url = next(
                (v for k, v in FOTMOB_TEAM_URLS.items() if _norm(k) == nm),
                ""
            )
        if url:
            teams_to_process.append((team_name, url, players))

    print(f"\n🏟️  {len(teams_to_process):,} teams matched to FotMob URLs")
    print(f"📸  Photos will be saved to: {PHOTOS_DIR}\n")

    downloaded  = 0
    skipped     = 0
    not_matched = 0

    for team_name, team_url, players in tqdm(teams_to_process, desc="Teams", unit="team"):
        tid = fotmob_team_id(team_url)
        if not tid:
            continue

        # Fetch squad (with delay to be polite)
        squad = fetch_squad(tid)
        time.sleep(DELAY_BETWEEN_TEAMS)

        if not squad:
            skipped += len(players)
            continue

        for player_name in players:
            filename = PHOTOS_DIR / f"{safe_filename(player_name)}.png"
            if filename.exists():
                skipped += 1
                continue

            # Check cache first
            cache_key = _norm(player_name)
            fotmob_id = cache.get(cache_key, "")

            if not fotmob_id:
                fotmob_id = match_player_id(player_name, squad)
                if fotmob_id:
                    cache[cache_key] = fotmob_id
                    save_cache(cache)

            if fotmob_id:
                ok = download_and_save(player_name, fotmob_id)
                if ok:
                    downloaded += 1
                else:
                    not_matched += 1
            else:
                not_matched += 1

            time.sleep(DELAY_BETWEEN_PLAYERS)

    print(f"\n✅  Done!")
    print(f"   Downloaded : {downloaded:,}")
    print(f"   Skipped    : {skipped:,} (already existed)")
    print(f"   Not matched: {not_matched:,}")
    print(f"\n📁  Photos saved to: {PHOTOS_DIR}")
    print("👉  Next step: git add photos/ && git commit -m 'Add player photos' && git push")


if __name__ == "__main__":
    main()