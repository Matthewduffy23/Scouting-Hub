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

import argparse, re, io, csv, sys, time, json, unicodedata
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
# Separate from CACHE_FILE so the existing player|team -> id cache keeps its
# shape and needs no migration. See load_missing().
MISSING_FILE = SCRIPT_DIR / "photo_missing_ids.json"
PHOTOS_DIR.mkdir(exist_ok=True)

# ── Source data ───────────────────────────────────────────────────────────────
# CHANGED 2026-08-12. This used to be a single hardcoded
# CSV_FILE = SCRIPT_DIR / "WORLDplayers_updatedddALLAPR26.csv" — a file that has
# since been deleted from this repo, so the script could not run at all. It now
# reads season_manifest.json, the same file Railway's server.py uses to decide
# which season is current, so it follows season transitions automatically.
MANIFEST_FILE = SCRIPT_DIR / "season_manifest.json"

# Which manifest keys to read, in order.
#
# "supplementary" IS included: it holds calendar-league players (Brazil,
# Argentina, Japan, USA, Norway...) at their CURRENT clubs, which "current" does
# not — 9,426 player+team combos that would otherwise have no photo under the
# right team name.
#
# "previous" is deliberately EXCLUDED: those rows are players at clubs they have
# since left, so every photo it adds is keyed to a stale team — 37,897 extra
# combos, roughly 7.4 hours of FotMob calls, for images of old clubs. server.py's
# find_photo_match() already falls back to a name-first match when a stored team
# is stale, so old-team photos are not needed to cover that case either.
SOURCE_KEYS = ("current", "supplementary")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://www.fotmob.com/",
    "Accept": "*/*",
}

SEARCH_URL = "https://www.fotmob.com/api/data/search/suggest?hits=50&lang=en&term={}"
PHOTO_URL  = "https://images.fotmob.com/image_resources/playerimages/{}.png"

DELAY_SEARCH = 0.5
DELAY_PHOTO  = 0.2

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


def load_missing():
    """IDs confirmed to have NO image on FotMob's CDN.

    Keyed by FotMob ID, not player|team: the absence is a property of the ID, and
    the same ID can be reached from several player+team keys after a transfer.

    Added 2026-08-12 after a scoped run logged 138 'errors' that were all the
    same thing — HTTP 403 with an S3 <Code>AccessDenied</Code> body on the exact
    image key. For a public bucket serving by exact key that means the object
    does not exist (S3 masks missing-key as denied when the caller lacks
    ListBucket). Those are permanent, so re-requesting them every run is pure
    waste: ~1.1s each, and at full scale roughly 6,000 players."""
    if MISSING_FILE.exists():
        try:
            return json.loads(MISSING_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_missing(missing):
    MISSING_FILE.write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")

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
    """Returns a status string rather than a bool, so the caller can tell a
    permanent absence from a transient failure:

      "ok"       saved
      "no_image" the CDN has no object at this key — 403/404. Permanent, and
                 recorded in the negative cache so it is never re-requested.
      "error"    anything else (network, timeout, undecodable body). Worth
                 retrying on a later run, so NOT cached.
    """
    fname = PHOTOS_DIR / f"{safe_filename(player, team)}.png"
    if fname.exists():
        return "ok"
    try:
        r = requests.get(PHOTO_URL.format(fotmob_id), headers=HEADERS, timeout=10)

        # 403 here is S3 masking a missing key as AccessDenied (the bucket is
        # public per-key but denies ListBucket); 404 is the same outcome stated
        # plainly. Either way the image does not exist and never will for this id.
        if r.status_code in (403, 404):
            return "no_image"
        if r.status_code != 200 or "image" not in r.headers.get("content-type", ""):
            return "error"

        img_bytes = r.content
        if REMBG_OK:
            try: img_bytes = rembg_remove(img_bytes)
            except: pass

        Image.open(io.BytesIO(img_bytes)).convert("RGBA").save(fname, "PNG")
        return "ok"
    except Exception:
        return "error"

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download player photos from FotMob for the current season's players.")
    ap.add_argument("--league", type=str, default=None,
                    help='Comma-separated league names to scope to, e.g. '
                         '"Brazil 1.,Argentina 1." — trailing periods optional. '
                         'Omit to process every league in the source files.')
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the player list and report what WOULD be fetched, "
                         "then exit without any FotMob calls or downloads.")
    args = ap.parse_args()

    players = read_players(args.league)
    cache   = load_cache()

    if args.dry_run:
        have = sum(1 for p, t in players
                   if (PHOTOS_DIR / f"{safe_filename(p,t)}.png").exists())
        cached = sum(1 for p, t in players if cache.get(f"{_norm(p)}|{_norm(t)}"))
        print(f"\n🧪  DRY RUN — no FotMob calls made, nothing written.")
        print(f"   players in scope    : {len(players):,}")
        print(f"   photos already on disk: {have:,}")
        print(f"   would be fetched    : {len(players) - have:,}")
        print(f"   of those, ID cached : {cached:,}")
        print(f"\n   sample of the list:")
        for p, t in players[:8]:
            mark = "have" if (PHOTOS_DIR / f"{safe_filename(p,t)}.png").exists() else "FETCH"
            print(f"     [{mark:5}] {p}  —  {t}   ->  {safe_filename(p,t)}.png")
        return

    already_cached = sum(
        1 for p, t in players
        if (PHOTOS_DIR / f"{safe_filename(p,t)}.png").exists()
    )
    print(f"   {already_cached:,} photos already exist — will skip these\n")

    missing = load_missing()
    downloaded = skipped = no_match = no_image = errors = cache_hits = 0

    print(f"📸  Processing {len(players):,} players…")
    print(f"📁  Saving to: {PHOTOS_DIR}")
    print(f"🚫  {len(missing):,} ids already known to have no image — will skip instantly\n")

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
            no_match += 1
            continue

        # Negative cache: this id was already confirmed to have no image. Skip
        # before the request, which is the whole point — no HTTP, no sleep.
        if fotmob_id in missing:
            no_image += 1
            cache_hits += 1
            continue

        result = download_photo(player, team, fotmob_id)
        time.sleep(DELAY_PHOTO)

        if result == "ok":
            downloaded += 1
        elif result == "no_image":
            no_image += 1
            missing[fotmob_id] = f"403/404 {time.strftime('%Y-%m-%d')}"
            if len(missing) % 25 == 0:
                save_missing(missing)
        else:
            errors += 1

    save_cache(cache)
    save_missing(missing)

    print(f"\n✅  Done!")
    print(f"   Downloaded         : {downloaded:,}")
    print(f"   Skipped            : {skipped:,} (photo already on disk)")
    print(f"   No photo available : {no_image + no_match:,} "
          f"({no_match:,} no FotMob match, {no_image:,} no image on FotMob)")
    if cache_hits:
        print(f"                        of which {cache_hits:,} skipped instantly "
              f"via the negative cache")
    print(f"   Errors             : {errors:,} (unexpected — network/decode, worth retrying)")
    print(f"\n👉  git add photos/ && git commit -m 'Add player photos' && git push")


def _norm_league(s):
    """Trailing-period-insensitive league key. The player CSVs use 'Brazil 1.'
    while a --league argument may reasonably be typed either way."""
    return _norm(s).rstrip(".").strip()


def resolve_source_files():
    """Season CSVs to read, resolved via season_manifest.json.

    Fails loudly rather than guessing a filename — the previous hardcoded
    CSV_FILE silently pointed at a file that had been deleted from this repo,
    which is exactly the failure this replaces."""
    if not MANIFEST_FILE.exists():
        print(f"❌  {MANIFEST_FILE.name} not found in {SCRIPT_DIR}")
        print("    It is generated by refresh_cycle.py's fast path and pushed here.")
        sys.exit(1)

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    files = []
    for key in SOURCE_KEYS:
        name = manifest.get(key)
        if not name:
            print(f"⚠️   manifest has no '{key}' entry — skipping")
            continue
        path = SCRIPT_DIR / name
        if not path.exists():
            print(f"⚠️   {name} ('{key}') is in the manifest but not on disk — skipping")
            continue
        files.append((key, path))

    if not files:
        print("❌  No usable season files resolved from the manifest. Nothing to do.")
        sys.exit(1)
    return files


def read_players(league_filter=None):
    wanted = None
    if league_filter:
        wanted = {_norm_league(x) for x in league_filter.split(",") if x.strip()}
        print(f"🔎  Scoped to {len(wanted)} league(s): {', '.join(sorted(wanted))}")

    rows, seen = [], set()
    for key, path in resolve_source_files():
        before = len(rows)
        print(f"📂  Reading {path.name} ({key})…")
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                player = (row.get("Player") or "").strip()
                team   = (row.get("Team")   or "").strip()
                if not player or not team:
                    continue
                if wanted is not None and _norm_league(row.get("League") or "") not in wanted:
                    continue
                key_pt = f"{_norm(player)}|{_norm(team)}"
                if key_pt in seen:
                    continue
                seen.add(key_pt)
                rows.append((player, team))
        print(f"   +{len(rows) - before:,} new (running total {len(rows):,})")

    if not rows:
        print("❌  No player+team combos matched. Check the --league spelling.")
        sys.exit(1)
    print(f"   {len(rows):,} unique player+team combos")
    return rows


if __name__ == "__main__":
    main()
