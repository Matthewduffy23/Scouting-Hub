# app.py — Scouting HQ with truly clickable tiles
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
<style>
:root{
  --bg:#0b0f1f;
  --text:#ffffff;   /* pure white title/text */
  --muted:#9fb0c8;
}
.stApp{ background:var(--bg); }
.block-container{ max-width:1120px; padding-top:50px; padding-bottom:48px; }

/* hero */
.hq-title{
  font-weight:900; margin:0; letter-spacing:.2px;
  font-size:clamp(36px,4.8vw,54px);
  color:var(--text) !important;   /* hard override */
}
.hq-sub{ color:var(--muted); margin:6px 0 40px 0; }

/* grid */
.tile-grid{
  display:grid; grid-template-columns:repeat(2,1fr); gap:36px;
}
@media(max-width:980px){ .tile-grid{ grid-template-columns:1fr; } }

/* tiles (links) */
.tile{
  display:flex; align-items:center; justify-content:center; text-align:center;
  height:200px; border-radius:22px; font-weight:700;
  font-size:clamp(22px,2.2vw,28px); color:#fff; text-decoration:none;
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
</style>
""", unsafe_allow_html=True)

# ---------------------- HERO ----------------------
st.markdown('<h1 class="hq-title">Scouting HQ</h1>', unsafe_allow_html=True)
st.markdown('<div class="hq-sub">Central scouting dashboard</div>', unsafe_allow_html=True)

# ---------------------- GRID ----------------------
# Use REAL anchors so the entire card is the link
st.markdown("""
<div class="tile-grid">

  <div class="tile cb">Center Backs<div class="sub">Coming soon</div></div>
  <div class="tile fb">Fullbacks<div class="sub">Coming soon</div></div>
  <div class="tile cm">Central Midfielders<div class="sub">Coming soon</div></div>

  <!-- These two are FULLY CLICKABLE -->
  <a class="tile att" href="./Attacker" target="_self" aria-label="Open Attacker page">Attackers</a>
  <a class="tile str" href="./Striker"  target="_self" aria-label="Open Striker page">Strikers</a>

</div>
""", unsafe_allow_html=True)



