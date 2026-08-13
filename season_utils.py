"""Season ordering for dataset pickers.

Every page used to sort its CSV list by st_mtime, which is wrong here and was
silently inverted: split_seasons.py writes season files newest-first and
shutil.copy2 preserves those timestamps, so the NEWEST season ends up with the
OLDEST mtime. `reverse=True` on mtime therefore defaulted every page to the
oldest season on disk (Scouting-Hub was defaulting to 2018-19).

Sorting on the season parsed out of the filename is both correct and stable —
it does not change when a file is re-copied, re-downloaded or touched.

Same idea as _season_leading_year() in the Database repo's server.py; kept as a
separate copy because these are separate repos with no shared package.
"""
import re

__all__ = ["season_key", "sort_by_season"]


def season_key(name):
    """First 4-digit year anywhere in the filename. Handles both naming patterns:

        2026-27WORLDFULL.csv   -> 2026     (player files: year leads)
        WORLDTEAMS2026-27.csv  -> 2026     (team files: year follows a prefix)

    A leading-year-only parse would return -1 for every team file and silently
    sort them all equal, so the year is matched anywhere in the string.

    Anything without a 4-digit year returns -1 and sorts last rather than
    raising, so an ad-hoc CSV in the folder can never take the default slot.
    """
    m = re.search(r"(\d{4})", str(name))
    return int(m.group(1)) if m else -1


def sort_by_season(paths, newest_first=True):
    """Sort Path/str entries by parsed season. Ties keep their input order,
    since Python's sort is stable."""
    return sorted(
        paths,
        key=lambda p: season_key(getattr(p, "name", str(p))),
        reverse=newest_first,
    )
