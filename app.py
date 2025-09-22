# app.py — Scouting Hub (dark + big tiles landing page)
import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Scouting Hub",
    page_icon="🏟️",
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
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 24px;
    margin-top: 30px;
}
.tile {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 32px 20px;
    text-align: center;
    cursor: pointer;
    transition: transform 0.15s ease, background 0.2s ease;
    box-shadow: 0 8px 18px rgba(0,0,0,0.35);
}
.tile:hover {
    transform: translateY(-4px);
    background: rgba(255,255,255,0.08);
}
.tile h2 {
    margin: 10px 0 6px 0;
    font-size: 1.5rem;
    color: white;
}
.tile p {
    margin: 0;
    font-size: 0.9rem;
    color: #9ca3af;
}
.tile.disabled {
    opacity: 0.5;
    cursor: default;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("🏟️ Scouting Hub")
st.caption("Head scouting dashboard — department access")

# ---------------- Tiles ----------------
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

# Strikers
if st.button("⚽", key="strikers_tile"):
    try:
        st.switch_page("pages/01_Striker.py")
    except Exception:
        st.warning("Use the sidebar → Pages → 01_Striker")
st.markdown(
    """
    <div class="tile" onclick="window.parent.postMessage({type:'streamlit_navigation',page:'01_Striker.py'}, '*');">
        <div style="font-size:36px;">⚽</div>
        <h2>Strikers</h2>
        <p>Finishing • xG • Box presence • Aerial threat</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Attackers
st.markdown(
    """
    <div class="tile" onclick="window.parent.postMessage({type:'streamlit_navigation',page:'02_Attacker.py'}, '*');">
        <div style="font-size:36px;">🎯</div>
        <h2>Attackers</h2>
        <p>Chance creation • Dribbles • xA • Crossing</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Goalkeepers (placeholder)
st.markdown(
    """
    <div class="tile disabled">
        <div style="font-size:36px;">🧤</div>
        <h2>Goalkeepers</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Center Backs (placeholder)
st.markdown(
    """
    <div class="tile disabled">
        <div style="font-size:36px;">🛡️</div>
        <h2>Center Backs</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fullbacks (placeholder)
st.markdown(
    """
    <div class="tile disabled">
        <div style="font-size:36px;">🏃‍♂️</div>
        <h2>Fullbacks</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Central Midfielders (placeholder)
st.markdown(
    """
    <div class="tile disabled">
        <div style="font-size:36px;">🎽</div>
        <h2>Central Midfielders</h2>
        <p>Coming soon</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Internal scouting tool • © 2025")





