"""League rank ("Top 10% / Top 20%") for the six position pages.

Same rule as Scout Index's League rank filter:
  - Pool = same league + same position group + same season. Each page loads one
    season's CSV, so the season is the loaded file; the position group is the
    page's own position_filter with every sub-role included (Attacker/Fullback
    role toggles only hide players, they don't shrink the pool).
  - Only players who clear the page's Min Minutes slider count, as pool members
    and as qualifiers. Youth leagues use 100 minutes (0 if the slider is at 0).
  - Ranked on the page's Complete Score (each page's own, GK included), recomputed on that
    whole pool - so sidebar/display filters (age, foot, value, contract, ...) only
    hide players, they never change anyone's rank.
  - Top X% <=> fewer than X% of the pool score strictly higher. "Top 8%" means
    floor(100 * strictly_higher / pool_size) + 1, so the label and the filter
    always agree (a "Top 10%" label always passes the Top 10% filter).

Each page calls:
    choice = league_rank_widget("att", selected_file)          # inside the sidebar
    ranks  = league_rank_table(df, position_ok=..., features=FEATURES,
                               score_fn=..., min_minutes=..., youth_leagues=YOUTH_LEAGUES,
                               league_strengths=LEAGUE_STRENGTHS)
    df_display = apply_league_rank(df_display, ranks, choice)
"""
import numpy as np
import pandas as pd
import streamlit as st

LEAGUE_RANK_OPTIONS = ["Off", "Top 10%", "Top 20%"]
LEAGUE_RANK_CUTS = {"Top 10%": 0.10, "Top 20%": 0.20}
LEAGUE_RANK_COL = "League rank"
YOUTH_MIN_MINUTES = 100


def league_rank_widget(prefix, selected_file):
    """Off / Top 10% / Top 20% radio. Call it inside the page's `with st.sidebar:`."""
    return st.radio(
        "League rank (vs same league + position, Min Minutes)",
        LEAGUE_RANK_OPTIONS,
        index=0,
        horizontal=True,
        key=f"{prefix}_league_rank_{selected_file}",
        help="Rank on Complete Score against every player in the same league and position "
             "group this season who meets Min Minutes (youth leagues: 100). Other filters "
             "only hide players - they don't change anyone's rank.",
    )


def pool_min_minutes(league, min_minutes, youth_leagues):
    if league in youth_leagues:
        return YOUTH_MIN_MINUTES if min_minutes > 0 else 0
    return min_minutes


def league_rank_table(df, *, position_ok, features, score_fn, min_minutes,
                      youth_leagues=(), league_strengths=None, inverse_metrics=()):
    """Rank every eligible player in `df` (the whole loaded season, unfiltered).

    position_ok: callable(position_string) -> bool, the page's full position group.
    score_fn:    callable(pool_df) -> Series of the main index score, given a pool
                 that already has "<feat> Percentile" columns (per league) and
                 "League Strength" - exactly what the page's ensure_*_metrics expects.
    Returns a DataFrame indexed like `df` (only ranked rows) with columns
    _lr_better, _lr_n, _lr_frac, League rank.
    """
    youth_leagues = set(youth_leagues or ())
    pool = df[df["Position"].astype(str).map(position_ok).astype(bool)].copy()
    pool["Minutes played"] = pd.to_numeric(pool["Minutes played"], errors="coerce")
    thr = pool["League"].map(lambda lg: pool_min_minutes(lg, min_minutes, youth_leagues))
    pool = pool[pool["Minutes played"] >= thr]
    for c in features:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=list(features))
    if pool.empty:
        return pd.DataFrame(columns=["_lr_better", "_lr_n", "_lr_frac", LEAGUE_RANK_COL])

    if league_strengths is not None:
        pool["League Strength"] = pool["League"].map(league_strengths).fillna(50.0)
    inverse_metrics = set(inverse_metrics or ())
    for feat in features:
        r = pool.groupby("League")[feat].rank(pct=True)
        pool[f"{feat} Percentile"] = ((1.0 - r) if feat in inverse_metrics else r) * 100.0

    score = pd.to_numeric(pd.Series(score_fn(pool)), errors="coerce").reindex(pool.index)
    keys = ["League"]
    if "Season" in pool.columns and pool["Season"].nunique(dropna=True) > 1:
        keys.append("Season")
    ranked = pool.assign(_lr_score=score).dropna(subset=["_lr_score"])
    g = ranked.groupby(keys)["_lr_score"]
    better = g.rank(method="min", ascending=False) - 1      # players strictly above
    n = g.transform("size")
    frac = better / n
    out = pd.DataFrame({"_lr_better": better.astype(int), "_lr_n": n.astype(int), "_lr_frac": frac},
                       index=ranked.index)
    out[LEAGUE_RANK_COL] = "Top " + (np.floor(frac * 100).astype(int) + 1).astype(str) + "%"
    # Pool sizes per league, for checking "X of N" by hand / in tests.
    st.session_state["_league_rank_pool_sizes"] = ranked.groupby("League").size().to_dict()
    st.session_state["_league_rank_top_counts"] = {
        lg: {"n": int(len(f)), "top10": int((f < 0.10).sum()), "top20": int((f < 0.20).sum())}
        for lg, f in out["_lr_frac"].groupby(ranked["League"])
    }
    return out


def apply_league_rank(df_show, ranks, choice):
    """Add the 'League rank' column (right after Player) and apply the Top X% filter.

    Players outside the ranked pool (under Min Minutes for their league) show '—'
    and are dropped whenever a Top X% filter is on.
    """
    df_show = df_show.copy()
    col = ranks[LEAGUE_RANK_COL].reindex(df_show.index) if len(ranks) else pd.Series(index=df_show.index, dtype=object)
    df_show[LEAGUE_RANK_COL] = col.fillna("—")
    cut = LEAGUE_RANK_CUTS.get(choice)
    if cut is not None:
        frac = ranks["_lr_frac"].reindex(df_show.index) if len(ranks) else pd.Series(np.nan, index=df_show.index)
        df_show = df_show[frac < cut]
    cols = list(df_show.columns)
    if "Player" in cols:
        cols.remove(LEAGUE_RANK_COL)
        cols.insert(cols.index("Player") + 1, LEAGUE_RANK_COL)
        df_show = df_show[cols]
    return df_show
