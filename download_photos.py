"""
download_photos.py — FotMob Search Edition (with initial matching)
Run from your Scouting-Hub folder:
    python download_photos.py

For each player in WORLDplayers_updated.csv:
1. Search FotMob by surname → get candidates
2. Filter by first initial (e.g. M. Smith → must start with M)
3. Score by team name match
4. Download photo + remove background
5. Save to photos/{player}__{team}.png
"""

import re, io, csv, sys, time, json, unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import requests
from PIL import Image
from tqdm import tqdm

try:
    from rembg import remove as rembg_remove
    REMBG_OK = False
    print("✅ rembg loaded — backgrounds will be removed")
except Exception:
    REMBG_OK = False
    print("⚠️  rembg not available — photos saved as-is")

SCRIPT_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = Path(r"C:\Users\matth\OneDrive\Documents\GitHub\scouting-photos\photos")
CACHE_FILE = SCRIPT_DIR / "photo_id_cache.json"
CSV_FILE   = SCRIPT_DIR / "WORLDplayers_updated.csv"
PHOTOS_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://www.fotmob.com/",
    "Accept": "*/*",
}

SEARCH_URL = "https://www.fotmob.com/api/data/search/suggest?hits=50&lang=en&term={}"
PHOTO_URL  = "https://images.fotmob.com/image_resources/playerimages/{}.png"

DELAY_SEARCH = 0.8
DELAY_PHOTO  = 0.3

# ── helpers ───────────────────────────────────────────────────────────────────

def _norm(s):
    if not s: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    return " ".join(s.strip().lower().split())

def _slug(s):
    return re.sub(r"[^a-z0-9]", "", _norm(s))

def _similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def safe_filename(player, team):
    p = "_".join(re.sub(r"[^a-z0-9 ]","",_norm(player)).split()) or "unknown"
    t = "_".join(re.sub(r"[^a-z0-9 ]","",_norm(team)).split()) or "unknown"
    return f"{p}__{t}"

def load_cache():
    if CACHE_FILE.exists():
        try: return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except: pass
    return {}

def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

# ── name parsing ──────────────────────────────────────────────────────────────

def parse_name(csv_name: str):
    """
    Parse abbreviated Wyscout names like 'M. Smith' or 'David Raya'.
    Returns (first_initial, surname) or ("", full_name) if not abbreviated.

    Examples:
      'M. Smith'      → ('m', 'smith')
      'V. van Dijk'   → ('v', 'van dijk')
      'David Raya'    → ('d', 'raya')       ← full name, use first letter
      'Trent Alexander-Arnold' → ('t', 'alexander-arnold')
    """
    name = csv_name.strip()

    # Pattern: single letter + dot → abbreviated
    # e.g. "M. Smith", "V. van Dijk", "J. Bowen"
    m = re.match(r'^([A-Za-z])\.\s+(.+)$', name)
    if m:
        initial = m.group(1).lower()
        surname = m.group(2).strip()
        # For compound surnames like "van Dijk", keep whole thing
        return initial, surname

    # Full name — use first letter of first word as initial
    # surname = last word (or last two for compound)
    parts = name.split()
    if len(parts) >= 2:
        initial = parts[0][0].lower()
        surname = parts[-1]
        return initial, surname

    # Single word name
    return "", name


def surname_for_search(csv_name: str) -> str:
    """Extract just the surname part for the FotMob search query."""
    _, surname = parse_name(csv_name)
    return surname


def first_initial(csv_name: str) -> str:
    """Extract first initial — 'm' from 'M. Smith'."""
    initial, _ = parse_name(csv_name)
    return initial

# ── FotMob search ─────────────────────────────────────────────────────────────

def search_fotmob(player_name: str, team_name: str) -> str:
    """
    Search FotMob for a player.
    Uses surname + first initial + team name for matching.
    Returns FotMob player ID string, or "" if not found.
    """
    surname  = surname_for_search(player_name)
    initial  = first_initial(player_name)
    search_term = surname if len(surname) > 2 else player_name

    try:
        url = SEARCH_URL.format(requests.utils.quote(search_term))
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return ""

        data = r.json()

        # Collect all player suggestions
        candidates = []
        for section in data:
            for suggestion in section.get("suggestions", []):
                if suggestion.get("type") == "player":
                    candidates.append(suggestion)

        if not candidates:
            return ""

        target_surname = _slug(surname)
        target_team    = _slug(team_name)

        best_id    = ""
        best_score = 0.0

        for c in candidates:
            cname = c.get("name", "") or ""
            cteam = c.get("teamName", "") or ""
            cid   = str(c.get("id", "")).strip()

            if not cid:
                continue

            cparts = cname.strip().split()
            if not cparts:
                continue

            # First initial filter — if we have an initial, skip non-matches
            if initial:
                candidate_initial = cparts[0][0].lower() if cparts[0] else ""
                if candidate_initial and candidate_initial != initial:
                    continue   # wrong first letter — skip

            # Surname match
            c_surname     = _slug(cparts[-1]) if cparts else ""
            surname_score = _similar(c_surname, target_surname)

            # Team match
            team_score = _similar(_slug(cteam), target_team)

            # Combined score — surname weighted more
            combined = surname_score * 0.55 + team_score * 0.45

            if combined > best_score:
                best_score = combined
                best_id    = cid

        # Threshold — must be reasonably confident
        return best_id if best_score >= 0.50 else ""

    except Exception:
        return ""

# ── photo download ────────────────────────────────────────────────────────────

def download_photo(player, team, fotmob_id):
    fname = PHOTOS_DIR / f"{safe_filename(player, team)}.png"
    if fname.exists():
        return True
    try:
        r = requests.get(PHOTO_URL.format(fotmob_id), headers=HEADERS, timeout=10)
        if r.status_code != 200 or "image" not in r.headers.get("content-type", ""):
            return False

        img_bytes = r.content
        if REMBG_OK:
            try: img_bytes = rembg_remove(img_bytes)
            except: pass

        Image.open(io.BytesIO(img_bytes)).convert("RGBA").save(fname, "PNG")
        return True
    except Exception:
        return False

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    players = read_players()
    cache   = load_cache()

    already_cached = sum(
        1 for p, t in players
        if (PHOTOS_DIR / f"{safe_filename(p,t)}.png").exists()
    )
    print(f"   {already_cached:,} photos already exist — will skip these\n")

    downloaded = skipped = not_found = errors = 0

    print(f"📸  Processing {len(players):,} players…")
    print(f"📁  Saving to: {PHOTOS_DIR}\n")

    for i, (player, team) in enumerate(tqdm(players, desc="Players", unit="player")):

        fname = PHOTOS_DIR / f"{safe_filename(player, team)}.png"
        if fname.exists():
            skipped += 1
            continue

        cache_key = f"{_norm(player)}|{_norm(team)}"
        fotmob_id = cache.get(cache_key, "")

        if not fotmob_id:
            fotmob_id = search_fotmob(player, team)
            time.sleep(DELAY_SEARCH)
            if fotmob_id:
                cache[cache_key] = fotmob_id
            if i % 100 == 0:
                save_cache(cache)

        if not fotmob_id:
            not_found += 1
            continue

        ok = download_photo(player, team, fotmob_id)
        time.sleep(DELAY_PHOTO)

        if ok:
            downloaded += 1
        else:
            errors += 1

    save_cache(cache)

    print(f"\n✅  Done!")
    print(f"   Downloaded : {downloaded:,}")
    print(f"   Skipped    : {skipped:,} (already existed)")
    print(f"   Not found  : {not_found:,} (no FotMob match)")
    print(f"   Errors     : {errors:,} (found but couldn't download)")
    print(f"\n👉  git add photos/ && git commit -m 'Add player photos' && git push")


def read_players():
    if not CSV_FILE.exists():
        print(f"❌  {CSV_FILE.name} not found in {SCRIPT_DIR}")
        sys.exit(1)
    print(f"📂  Reading {CSV_FILE.name}…")
    rows = []
    seen = set()
    with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            player = (row.get("Player") or "").strip()
            team   = (row.get("Team")   or "").strip()
            if player and team:
                key = f"{_norm(player)}|{_norm(team)}"
                if key not in seen:
                    seen.add(key)
                    rows.append((player, team))
    print(f"   {len(rows):,} unique player+team combos")
    return rows


if __name__ == "__main__":
    main()
