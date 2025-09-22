# app.py — Scouting HQ with spaced, fully clickable tiles
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
<style>
:root{
  --bg:#0b0f1f;
  --text:#eef3fb;
  --muted:#9fb0c8;
}
.stApp{ background:var(--bg); }
.block-container{ max-width:1120px; padding-top:50px; padding-bottom:48px; }

/* hero */
.hq-title{ font-weight:900; color:var(--text); margin:0; letter-spacing:.2px;
  font-size:clamp(36px,4.8vw,54px); }
.hq-sub{ color:var(--muted); margin:6px 0 40px 0; }

/* grid */
.tile-grid{
  display:grid; grid-template-columns:repeat(2,1fr); gap:36px;
}
@media(max-width:980px){ .tile-grid{ grid-template-columns:1fr; } }

/* tile base */
.tile{
  position:relative; height:200px; border-radius:22px; cursor:pointer;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  color:white; text-align:center;
  font-weight:700; font-size:clamp(22px,2.2vw,28px);
  box-shadow:0 18px 36px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
  transition:transform .18s ease, box-shadow .18s ease;
}
.tile:hover{ transform:translateY(-6px); box-shadow:0 34px 66px rgba(0,0,0,.48); }
.tile .sub{ font-size:15px; font-weight:500; opacity:.85; margin-top:8px; }

/* gradients */
.cb  { background:linear-gradient(135deg,#1d976c,#93f9b9); }
.fb  { background:linear-gradient(135deg,#c94b4b,#4b134f); }
.cm  { background:linear-gradient(135deg,#f7971e,#ffd200); }
.att { background:linear-gradient(135deg,#8e2de2,#4a00e0); }
.str { background:linear-gradient(135deg,#2980b9,#2c3e50); }

/* make Streamlit buttons invisible overlays that fill the card */
.stButton>button{
  position:absolute; inset:0; border:none; background:transparent; cursor:pointer;
  color:transparent; box-shadow:none;
}
.stButton>button:hover{ background:transparent; }
</style>
""", unsafe_allow_html=True)

# ---------------------- HERO ----------------------
st.markdown('<h1 class="hq-title">Scouting HQ</h1>', unsafe_allow_html=True)
st.markdown('<div class="hq-sub">Central scouting dashboard</div>', unsafe_allow_html=True)

# ---------------------- GRID ----------------------
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

# Center Backs (placeholder)
st.markdown('<div class="tile cb">Center Backs<div class="sub">Coming soon</div></div>', unsafe_allow_html=True)
st.button("cb", key="cb")

# Fullbacks (placeholder)
st.markdown('<div class="tile fb">Fullbacks<div class="sub">Coming soon</div></div>', unsafe_allow_html=True)
st.button("fb", key="fb")

# Central Midfielders (placeholder)
st.markdown('<div class="tile cm">Central Midfielders<div class="sub">Coming soon</div></div>', unsafe_allow_html=True)
st.button("cm", key="cm")

# Attackers (clickable)
st.markdown('<div class="tile att">Attackers</div>', unsafe_allow_html=True)
if st.button("att", key="att"):
    st.switch_page("pages/02_Attacker.py")

# Strikers (clickable)
st.markdown('<div class="tile str">Strikers</div>', unsafe_allow_html=True)
if st.button("str", key="str"):
    st.switch_page("pages/01_Striker.py")

st.markdown('</div>', unsafe_allow_html=True)
