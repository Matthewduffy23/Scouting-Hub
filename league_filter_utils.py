"""Shared, safety-hardened league-selection widget for the sidebar.

Every position page (Center Backs / Fullbacks / Central Midfielders /
Attacker / Strikers / Goalkeeper) builds the same "Leagues (add or prune the
presets)" sidebar multiselect: a session_state key scoped to the page +
dataset, a preset signature that resets the selection back to the presets
when region/youth/cup/hidden toggles change, and a defense against
Streamlit's StreamlitAPIException when a stored selection contains leagues no
longer in `options` (e.g. after swapping in an uploaded CSV with different
leagues, since the widget key is derived from `selected_file`, not the data
itself).

04_Attacker.py alone had a hand-rolled version of that last defense (added
2026-08-13), and it had a 2026-11-16 regression: `_kept if _kept else
default_leagues` could not tell "the stored selection became entirely stale"
apart from "the user deliberately cleared every chip" - both produce an empty
list - so clearing the widget silently snapped right back to every league
before the clear was ever visible on screen. A later "add Armenia 1." then
landed on top of the full pool instead of replacing it - reproduced live on
04_Attacker.py for Armenia 1. / Luxembourg 1. / England 2.

Centralized here, with the fix, so the pages can't drift apart on this again.
"""
import streamlit as st


def multiselect_safe(label, *, options, default=None, key=None, **kwargs):
    """Clamp defaults to current options to avoid Streamlit crashes after dataset switches."""
    options = list(options)
    default = [x for x in (default or []) if x in options]
    return st.multiselect(label, options=options, default=default, key=key, **kwargs)


def resolve_league_selection(prefix, selected_file, leagues_avail, default_leagues, preset_sig):
    """Initialize/repair the stored selection for the leagues multiselect.

    Returns (ms_key, value_to_render) - the page still calls multiselect_safe
    itself so each page's exact label/layout stays untouched:

        ms_key, _default = resolve_league_selection(
            "att", selected_file, leagues_avail, default_leagues, preset_sig
        )
        leagues_sel = multiselect_safe(
            "Leagues (add or prune the presets)",
            options=leagues_avail, default=_default, key=ms_key,
        )

    value_to_render is always a valid subset of leagues_avail, and an
    intentionally-empty selection (the user cleared every chip) is preserved
    as [] rather than silently replaced - callers downstream are expected to
    show a "select at least one league" warning on an empty result rather
    than treat it as "no filter".
    """
    ms_key = f"{prefix}_leagues_sel_{selected_file}"
    sig_key = f"{prefix}_preset_sig"

    if ms_key not in st.session_state:
        st.session_state[ms_key] = default_leagues
        # Record the signature now too. Without this the FIRST rerun after page
        # load saw sig_key missing, treated it as "presets changed" and reset the
        # selection to the defaults - so the first league pick (e.g. England 3.
        # alone) snapped straight back to every league. 2026-11-19.
        st.session_state[sig_key] = preset_sig
        return ms_key, st.session_state[ms_key]

    if st.session_state.get(sig_key) != preset_sig:
        st.session_state[sig_key] = preset_sig
        st.session_state[ms_key] = default_leagues
        return ms_key, st.session_state[ms_key]

    old = st.session_state.get(ms_key, default_leagues)
    kept = [x for x in old if x in leagues_avail]
    if kept:
        st.session_state[ms_key] = kept
    elif old:
        # Every previously-selected league disappeared from the options (e.g.
        # an uploaded dataset swap) - that's staleness, not a deliberate clear.
        st.session_state[ms_key] = default_leagues
    # else: old was already [] - a deliberate clear - leave it alone.

    return ms_key, st.session_state[ms_key]
