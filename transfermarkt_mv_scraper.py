"""
update_market_values.py

Updates Market value column in your CSV using Transfermarkt data.
- Searches Transfermarkt by player name + team match
- Replaces Market value if found, keeps existing value if not found
- Saves checkpoint so resumable if it crashes
- Run from your wyscout-downloader folder

HOW TO RUN:
    cd C:\\Users\\matth\\OneDrive\\Desktop\\wyscout-downloader
    python update_market_values.py

INPUT:  wyscout_exports/WORLDplayers_updated.csv
OUTPUT: wyscout_exports/WORLDplayers_updated_mv.csv
"""

import re
import json
import time
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import requests
import pandas as pd
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────
INPUT_CSV   = Path(r"C:\Users\matth\OneDrive\Documents\GitHub\Scouting-Hub\WORLDplayers_updatedddALLAPR26.csv")
OUTPUT_CSV  = Path(r"C:\Users\matth\OneDrive\Documents\GitHub\Scouting-Hub\WORLDplayers_updatedddALLAPR26_mv.csv")
CACHE_FILE  = Path(r"C:\Users\matth\OneDrive\Desktop\wyscout-downloader\mv_cache.json")
CHECKPOINT  = Path(r"C:\Users\matth\OneDrive\Desktop\wyscout-downloader\mv_checkpoint.json")

BASE_URL    = "https://transfermarkt-api.fly.dev"
DELAY       = 0.5   # seconds between requests — be polite

# ── Leagues with poor Transfermarkt coverage — skip to save time ────
SKIP_LEAGUES = {
    "England 8.", "England 9.", "England 10.",
    "Estonia 2.", "Faroe Islands 1.", "Ireland 2.",
    "Malta 1.", "Moldova 1.", "Lithuania 1.",
    "North Macedonia 1.", "Kazakhstan 1.", "Uzbekistan 1.",
    "Bolivia 1.", "Panama 1.", "Costa Rica 1.",
    "Peru 1.", "Venezuela 1.", "Korea 2.",
    "Sweden 3.", "Slovenia 2.", "Slovakia 2.",
    "Greece 2.", "Scotland 3.", "USA 3.",
    "Germany 4.", "Portugal 4.", "Faroe Islands 1.",
    "Nigeria 1.", "Algeria 1.", "Qatar 1.",
    "Saudi 1.", "UAE 1.",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


# ── Helpers ─────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.strip().lower().split())

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _norm(s))

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _surname(name: str) -> str:
    parts = str(name).strip().split()
    return parts[-1] if parts else ""


# ── Cache ────────────────────────────────────────────────────────────
def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_cache(cache: dict):
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        try:
            return set(json.loads(CHECKPOINT.read_text(encoding="utf-8")))
        except Exception:
            pass
    return set()

def save_checkpoint(done: set):
    CHECKPOINT.write_text(
        json.dumps(list(done), ensure_ascii=False),
        encoding="utf-8"
    )


# ── Transfermarkt search ─────────────────────────────────────────────
def search_tm(player_name: str, team_name: str) -> float | None:
    """
    Search Transfermarkt for a player.
    Returns market value in euros (float) or None if not found.
    """
    surname = _surname(player_name)
    search_term = surname if len(surname) > 2 else player_name

    try:
        url = f"{BASE_URL}/players/search/{requests.utils.quote(search_term)}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        results = data.get("results", [])
        if not results:
            return None

        target_surname = _slug(surname)
        target_team    = _slug(team_name)

        best_id    = None
        best_score = 0.0

        for p in results:
            pname = p.get("name", "") or ""
            pteam = p.get("club", {}).get("name", "") or ""
            pid   = p.get("id", "")

            if not pid:
                continue

            p_surname    = _slug(_surname(pname))
            team_score   = _similar(_slug(pteam), target_team)
            surname_score = _similar(p_surname, target_surname)
            combined     = surname_score * 0.55 + team_score * 0.45

            if combined > best_score:
                best_score = combined
                best_id    = pid

        if not best_id or best_score < 0.45:
            return None

        # Get market value for the matched player
        mv_url = f"{BASE_URL}/players/{best_id}/market_value"
        mv_r = requests.get(mv_url, headers=HEADERS, timeout=10)
        if mv_r.status_code != 200:
            return None

        mv_data = mv_r.json()

        # Try to get current market value
        current = mv_data.get("marketValue", None)
        if current is None:
            history = mv_data.get("marketValueHistory", [])
            if history:
                current = history[-1].get("value", None)

        if current is None:
            return None

        # Convert to float euros
        if isinstance(current, (int, float)):
            return float(current)

        # Sometimes comes as string like "€5.5m" or "€500k"
        if isinstance(current, str):
            current = current.replace("€", "").replace(",", "").strip().lower()
            if "m" in current:
                return float(current.replace("m", "")) * 1_000_000
            if "k" in current:
                return float(current.replace("k", "")) * 1_000
            try:
                return float(current)
            except Exception:
                return None

        return None

    except Exception:
        return None


# ── Main ─────────────────────────────────────────────────────────────
def main():
    if not INPUT_CSV.exists():
        print(f"Input file not found: {INPUT_CSV}")
        return

    print(f"Reading {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} rows loaded")

    if "Market value" not in df.columns:
        print("No 'Market value' column found — adding one")
        df["Market value"] = None

    cache      = load_cache()
    done_keys  = load_checkpoint()

    updated = 0
    skipped = 0
    not_found = 0
    kept_existing = 0

    print(f"\nUpdating market values...")
    print(f"  Skipping {len(SKIP_LEAGUES)} low-coverage leagues\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Players", unit="player"):
        player = str(row.get("Player", "") or "").strip()
        team   = str(row.get("Team", "") or "").strip()
        league = str(row.get("League", "") or "").strip()

        if not player or not team:
            continue

        # Skip low-coverage leagues
        if league in SKIP_LEAGUES:
            skipped += 1
            continue

        cache_key = f"{_norm(player)}|{_norm(team)}"

        # Skip if already processed in a previous run
        if cache_key in done_keys:
            skipped += 1
            continue

        # Check cache first
        if cache_key in cache:
            mv = cache[cache_key]
            if mv is not None:
                df.at[idx, "Market value"] = mv
                updated += 1
            else:
                kept_existing += 1
            done_keys.add(cache_key)
            continue

        # Search Transfermarkt
        mv = search_tm(player, team)
        time.sleep(DELAY)

        cache[cache_key] = mv

        if mv is not None:
            df.at[idx, "Market value"] = mv
            updated += 1
        else:
            # Keep existing Wyscout value
            kept_existing += 1
            not_found += 1

        done_keys.add(cache_key)

        # Save cache + checkpoint every 100 players
        if idx % 100 == 0:
            save_cache(cache)
            save_checkpoint(done_keys)

            # Save progress to output file
            df.to_csv(OUTPUT_CSV, index=False)

    # Final save
    save_cache(cache)
    save_checkpoint(done_keys)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone!")
    print(f"  Updated from TM  : {updated:,}")
    print(f"  Kept existing    : {kept_existing:,}")
    print(f"  Skipped leagues  : {skipped:,}")
    print(f"  Not found on TM  : {not_found:,}")
    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"\nNext steps:")
    print(f"  1. Check the output CSV looks correct")
    print(f"  2. Rename to WORLDplayers_updated.csv")
    print(f"  3. Copy to Scouting-Hub folder")

    # Clear checkpoint on success
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print(f"  Checkpoint cleared")


if __name__ == "__main__":
    main()