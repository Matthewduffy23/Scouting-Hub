# app.py — Scouting HQ (5 clickable tiles)
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
<style>
:root{
  --bg:#0b0f1f;
  --text:#ffffff;
  --muted:#9fb0c8;
}
.stApp{ background:var(--bg); }
.block-container{ max-width:1120px; padding-top:50px; padding-bottom:48px; }

/* hero */
.hq-title{
  font-weight:900; margin:0; letter-spacing:.2px;
  font-size:clamp(36px,4.8vw,54px);
  color:var(--text) !important;
}
.hq-sub{ color:var(--muted); margin:6px 0 40px 0; }

/* stacked tiles */
.tile-list{ display:grid; grid-template-columns:1fr; }

/* tile style */
.tile{
  display:flex; align-items:center; justify-content:center; text-align:center;
  height:200px; border-radius:22px; font-weight:800;
  font-size:clamp(22px,2.2vw,28px);
  box-shadow:0 18px 36px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
  transition:transform .18s ease, box-shadow .18s ease;
  color:#fff; text-decoration:none;
  margin-bottom:36px;
}
.tile:last-child{ margin-bottom:0; }
.tile:hover{ transform:translateY(-6px); box-shadow:0 34px 66px rgba(0,0,0,.48); }

/* force links to be white + no underline */
.tile:link, .tile:visited, .tile:hover, .tile:active {
  color:#fff !important;
  text-decoration:none !important;
}

/* gradients */
.cb  { background:linear-gradient(135deg,#1d976c,#93f9b9); }
.fb  { background:linear-gradient(135deg,#c94b4b,#4b134f); }
.cm  { background:linear-gradient(135deg,#f7971e,#ffd200); color:#1a1a1a; }
.att { background:linear-gradient(135deg,#8e2de2,#4a00e0); }
.str { background:linear-gradient(135deg,#2980b9,#2c3e50); }
</style>
""", unsafe_allow_html=True)

# ---------------------- HERO ----------------------
st.markdown('<h1 class="hq-title">Scouting HQ</h1>', unsafe_allow_html=True)
st.markdown('<div class="hq-sub">Central scouting dashboard</div>', unsafe_allow_html=True)

# ---------------------- TILES ----------------------
# NOTE: href must match page filenames in /pages/ without the numeric prefix or ".py"
# Your files: 01_Center_Backs.py, 02_Fullbacks.py, 03_Central_Midfielders.py, 04_Attacker.py, 05_Strikers.py
st.markdown("""
<div class="tile-list">
  <a class="tile cb"  href="./Center_Backs"         target="_self">Center Backs</a>
  <a class="tile fb"  href="./Fullbacks"            target="_self">Fullbacks</a>
  <a class="tile cm"  href="./Central_Midfielders"  target="_self">Central Midfielders</a>
  <a class="tile att" href="./Attacker"             target="_self">Attackers</a>
  <a class="tile str" href="./Strikers"             target="_self">Strikers</a>
</div>
""", unsafe_allow_html=True)



