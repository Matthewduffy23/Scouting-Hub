# app.py — Scouting HQ with 6 bright tiles (2x2x2)
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
    background-color: #0f172a; /* deep navy */
}
h1, h2, h3, p, span {
    color: #f8fafc !important; /* almost white */
}
.tile-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 32px;
    margin-top: 36px;
    margin-bottom: 36px;
}
.tile {
    border-radius: 16px;
    padding: 60px 20px;
    text-align: center;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.3s ease;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(0,0,0,0.3);
}
.tile:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.45);
}
.tile h2 {
    margin: 0;
    font-size: 1.8rem;
    color: #111;
}
.tile p {
    margin: 6px 0 0 0;
    font-size: 1rem;
    color: #222;
    font-weight: 400;
}

/* Bright matching colours */
.gk { background: #60a5fa; }   /* bright blue */
.cb { background: #34d399; }   /* bright green */
.fb { background: #f472b6; }   /* bright pink */
.cm { background: #facc15; }   /* bright yellow */
.att { background: #a78bfa; }  /* bright purple */
.str { background: #f87171; }  /* bright red */

.disabled {
    opacity: 0.75;
    cursor: not-allowed;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("Scouting HQ")
st.caption("Central scouting dashboard")

# ---------------- Tiles ----------------
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

# Goalkeepers
st.markdown(
    """
    <div class="tile gk disabled">
        <h2>Goalkeepers</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Center Backs
st.markdown(
    """
    <div class="tile cb disabled">
        <h2>Center Backs</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fullbacks
st.markdown(
    """
    <div class="tile fb disabled">
        <h2>Fullbacks</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Central Midfielders
st.markdown(
    """
    <div class="tile cm disabled">
        <h2>Central Midfielders</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Attackers (linked)
st.markdown(
    """
    <div class="tile att" onclick="window.parent.postMessage({type:'streamlit_navigation',page:'02_Attacker.py'}, '*');">
        <h2>Attackers</h2>
        <p>Creative threat, xA, dribbles</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Strikers (linked)
st.markdown(
    """
    <div class="tile str" onclick="window.parent.postMessage({type:'streamlit_navigation',page:'01_Striker.py'}, '*');">
        <h2>Strikers</h2>
        <p>Finishing, xG, box presence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Internal scouting platform • © 2025")







