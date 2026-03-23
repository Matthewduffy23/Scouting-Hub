"""
download_photos.py
Run once locally from your Scouting-Hub folder:
    python download_photos.py

Reads WORLDplayers_updated.csv only (has every league/player).
Saves photos as: photos/{player_name}__{team_name}.png
"""

import re, io, csv, sys, time, json, unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import requests
from PIL import Image
from tqdm import tqdm

try:
    from rembg import remove as rembg_remove
    REMBG_OK = True
    print("✅ rembg loaded — backgrounds will be removed")
except Exception:
    REMBG_OK = False
    print("⚠️  rembg not available — photos saved as-is")

SCRIPT_DIR  = Path(__file__).resolve().parent
PHOTOS_DIR  = SCRIPT_DIR / "photos"
CACHE_FILE  = SCRIPT_DIR / "photo_id_cache.json"
CSV_FILE    = SCRIPT_DIR / "WORLDplayers_updated.csv"
PHOTOS_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.fotmob.com/",
}

DELAY_TEAMS   = 1.2
DELAY_PLAYERS = 0.3

# ── helpers ───────────────────────────────────────────────────────────────────

def _norm(s):
    if not s: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    return " ".join(s.strip().lower().split())

def _slug(s):
    return re.sub(r"[^a-z0-9]","",_norm(s))

def _surname(name):
    parts = name.strip().split()
    return parts[-1] if parts else ""

def _similar(a,b):
    return SequenceMatcher(None,a,b).ratio()

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

# ── read CSV ──────────────────────────────────────────────────────────────────

def read_players():
    if not CSV_FILE.exists():
        print(f"❌  {CSV_FILE.name} not found in {SCRIPT_DIR}")
        sys.exit(1)
    print(f"📂  Reading {CSV_FILE.name}…")
    team_players = {}
    with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            player = (row.get("Player") or "").strip()
            team   = (row.get("Team")   or "").strip()
            if player and team:
                team_players.setdefault(team, set()).add(player)
    total = sum(len(v) for v in team_players.values())
    print(f"   {total:,} rows — {len(team_players):,} teams — {sum(1 for v in team_players.values() for _ in v):,} unique player+team combos")
    return team_players

# ── FotMob ────────────────────────────────────────────────────────────────────

def fotmob_tid(url):
    m = re.search(r"/teams/(\d+)/", url or "")
    return m.group(1) if m else ""

def fetch_squad(tid):
    try:
        r = requests.get(f"https://www.fotmob.com/api/teams?id={tid}", headers=HEADERS, timeout=15)
        if r.status_code != 200: return []
        data = r.json()
        squad = []
        raw = data.get("squad")
        if isinstance(raw, list):
            for s in raw:
                squad.extend([m for m in (s.get("members") or s.get("players") or []) if isinstance(m,dict)])
        elif isinstance(raw, dict):
            for k in ("members","players"):
                squad.extend([m for m in (raw.get(k) or []) if isinstance(m,dict)])
            nested = raw.get("squad")
            if isinstance(nested, list):
                for s in nested:
                    squad.extend([m for m in (s.get("members") or s.get("players") or []) if isinstance(m,dict)])
        return squad
    except: return []

def match_id(csv_name, squad):
    ts = _slug(_surname(csv_name))
    tf = _slug(csv_name)
    best_id, best_sc = "", 0.0
    for m in squad:
        name = m.get("name") or m.get("playerName") or ""
        pid  = str(m.get("id") or m.get("playerId") or m.get("primaryId") or "").strip()
        if not pid or not pid.isdigit(): continue
        ss = _slug(_surname(name))
        sf = _slug(name)
        if ss == ts:
            if tf in sf or sf in tf: return pid
            best_id, best_sc = pid, 1.0
            continue
        sc = _similar(ss, ts)
        if sc > best_sc: best_sc, best_id = sc, pid
    return best_id if best_sc >= 0.82 else ""

def download_photo(player, team, fotmob_id):
    fname = PHOTOS_DIR / f"{safe_filename(player, team)}.png"
    if fname.exists(): return True
    try:
        r = requests.get(f"https://images.fotmob.com/image_resources/playerimages/{fotmob_id}.png", headers=HEADERS, timeout=10)
        if r.status_code != 200 or "image" not in r.headers.get("content-type",""): return False
        img_bytes = r.content
        if REMBG_OK:
            try: img_bytes = rembg_remove(img_bytes)
            except: pass
        Image.open(io.BytesIO(img_bytes)).convert("RGBA").save(fname, "PNG")
        return True
    except: return False

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from team_fotmob_urls import FOTMOB_TEAM_URLS
    except ImportError:
        print("❌  team_fotmob_urls.py not found"); sys.exit(1)

    team_players = read_players()
    cache = load_cache()

    teams = []
    for team, players in team_players.items():
        url = FOTMOB_TEAM_URLS.get(team,"") or next((v for k,v in FOTMOB_TEAM_URLS.items() if _norm(k)==_norm(team)),"")
        if url: teams.append((team, url, players))

    print(f"\n🏟️  {len(teams):,} / {len(team_players):,} teams matched to FotMob URLs")
    print(f"📁  Saving to: {PHOTOS_DIR}\n")

    downloaded = skipped = not_matched = 0

    for team, url, players in tqdm(teams, desc="Teams", unit="team"):
        tid = fotmob_tid(url)
        if not tid: continue
        squad = fetch_squad(tid)
        time.sleep(DELAY_TEAMS)
        if not squad: skipped += len(players); continue

        for player in players:
            fname = PHOTOS_DIR / f"{safe_filename(player, team)}.png"
            if fname.exists(): skipped += 1; continue

            ck = f"{_norm(player)}|{_norm(team)}"
            fid = cache.get(ck,"")
            if not fid:
                fid = match_id(player, squad)
                if fid: cache[ck] = fid; save_cache(cache)

            if fid:
                downloaded += 1 if download_photo(player, team, fid) else 0
                if not download_photo(player, team, fid): not_matched += 1
            else:
                not_matched += 1
            time.sleep(DELAY_PLAYERS)

    print(f"\n✅  Done!")
    print(f"   Downloaded : {downloaded:,}")
    print(f"   Skipped    : {skipped:,}")
    print(f"   Not matched: {not_matched:,}")
    print(f"\n👉  git add photos/ && git commit -m 'Add player photos' && git push")

if __name__ == "__main__":
    main()