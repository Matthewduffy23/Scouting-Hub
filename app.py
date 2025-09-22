# app.py — Scouting Hub (dark + minimal landing page)
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
/* Dark background */
.stApp {
    background-color: #0b1220;
}

/* Text color */
h1, h2, h3, p, li, div, span {
    color: #e5e7eb !important;
}

/* Center the tabs */
[data-baseweb="tab-list"] {
    justify-content: center;
}

/* Tab styling */
[data-baseweb="tab"] {
    font-size: 1.1rem;
    padding: 12px 24px;
    border-radius: 8px 8px 0 0;
    color: #e5e7eb !important;
}

/* Selected tab colors */
[data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
    background: #1d4ed8;   /* Blue for Strikers */
}
[data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
    background: #16a34a;   /* Green for Attackers */
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("🏟️ Scouting Hub")
st.caption("Head scouting dashboard — minimal landing page")

# ---------------- Tabs ----------------
tabs = st.tabs(["⚽ Strikers", "🎯 Attackers"])

with tabs[0]:
    st.subheader("Strikers Department")
    st.write("Focus: finishing, xG, box presence, aerial threat.")
    if st.button("Open Strikers App", use_container_width=True):
        try:
            st.switch_page("pages/01_Striker.py")
        except Exception:
            st.warning("Use the left sidebar → Pages → 01_Striker")

with tabs[1]:
    st.subheader("Attackers Department")
    st.write("Focus: creation, dribbles, xA, crossing, chance-making.")
    if st.button("Open Attackers App", use_container_width=True):
        try:
            st.switch_page("pages/02_Attacker.py")
        except Exception:
            st.warning("Use the left sidebar → Pages → 02_Attacker")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Internal scouting tool • © 2025")





