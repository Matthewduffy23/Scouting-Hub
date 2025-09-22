# app.py — Scouting Hub (dark + coloured tiles 2x2x2)
import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Scouting HQ",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
}
h1, h2, h3, p, li, div, span {
    color: #e5e7eb !important;
}
.tile-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 28px;
    margin-top: 36px;
}
.tile {
    border-radius: 18px;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: transform 0.15s ease, filter 0.2s ease;
    color: #111827;
    font-weight: 600;
    box-shadow: 0 6px 16px rgba(0,0,0,0.35);
}
.tile:hover {
    transform: translateY(-4px);
    filter: brightness(1.08);
}
.tile h2 {
    margin: 12px 0 6px 0;
    font-size: 1.6rem;
    color: inherit;
}
.tile p {
    margin: 0;
    font-size: 0.9rem;
    font-weight: 400;
    color: rgba(17,24,39,0.8);
}

/* Pastel colours */
.gk { background: #bae6fd; }      /* light blue */
.cb { background: #bbf7d0; }      /* light green */
.fb { background: #fbcfe8; }      /* light pink */
.cm { background: #fde68a; }      /* light yellow */
.att { background: #ddd6fe; }     /* light purple */
.str { background: #fecaca; }     /* light red */
.disabled { opacity: 0.65; cursor: not-allowed; }
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("Scouting HQ")
st.caption("Central scouting dashboard")

# ---------------- Tiles ----------------
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

# Goalkeepers (placeholder)
st.markdown(
    """
    <div class="tile gk disabled">
        <div style="font-size:38px;"/div>
        <h2>Goalkeepers</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Center Backs (placeholder)
st.markdown(
    """
    <div class="tile cb disabled">
        <div style="font-size:38px;"/div>
        <h2>Center Backs</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fullbacks (placeholder)
st.markdown(
    """
    <div class="tile fb disabled">
        <div style="font-size:38px;"/div>
        <h2>Fullbacks</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Central Midfielders (placeholder)
st.markdown(
    """
    <div class="tile cm disabled">
        <div style="font-size:38px;"/div>
        <h2>Central Midfielders</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Attackers (link to page)
st.markdown(
    """
    <div class="tile att" onclick="window.parent.postMessage({type:'streamlit_navigation',page:'02_Attacker.py'}, '*');">
        <div style="font-size:38px;"></div>
        <h2>Attackers</h2>
        <p>Chance creation • Dribbles • xA • Crossing</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Strikers (link to page)
st.markdown(
    """
    <div class="tile str" onclick="window.parent.postMessage({type:'streamlit_navigation',page:'01_Striker.py'}, '*');">
        <div style="font-size:38px;"></div>
        <h2>Strikers</h2>
        <p>Finishing • xG • Box presence • Aerial threat</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Internal scouting tool • © 2025")






