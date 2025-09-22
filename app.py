# app.py — Scouting HQ (stacked, fully clickable tiles)
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
<style>
:root{
  --bg:#0b0f1f;
  --text:#ffffff;      /* hard white */
  --muted:#9fb0c8;
}
.stApp{ background:var(--bg); }
.block-container{ max-width:1120px; padding-top:50px; padding-bottom:48px; }

/* hero */
.hq-title{
  font-weight:900; margin:0; letter-spacing:.2px;
  font-size:clamp(36px,4.8vw,54px); color:var(--text) !important;
}
.hq-sub{ color:var(--muted); margin:6px 0 40px 0; }

/* stacked tiles */
.tile-list{ display:grid; grid-template-columns:1fr; gap:36px; }

/* the tile is a REAL <a>, filling the card */
.tile{
  display:flex; align-items:center; justify-content:center; text-align:center;
  height:200px; border-radius:22px; font-weight:800;
  font-size:clamp(22px,2.2vw,28px);
  color:#fff; text-decoration:none;  /* no underline, white text */
  box-shadow:0 18px 36px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
  transition:transform .18s ease, box-shadow .18s ease;
}
.tile:hover{ transform:translateY(-6px); box-shadow:0 34px 66px rgba(0,0,0,.48); }

/* gradients */
.att { background:linear-gradient(135deg,#8e2de2,#4a00e0); }
.str { background:linear-gradient(135deg,#2980b9,#2c3e50); }
</style>
""", unsafe_allow_html=True)

# ---------------------- HERO ----------------------
st.markdown('<h1 class="hq-title">Scouting HQ</h1>', unsafe_allow_html=True)
st.markdown('<div class="hq-sub">Central scouting dashboard</div>', unsafe_allow_html=True)

# ---------------------- STACKED, CLICKABLE TILES ----------------------
# IMPORTANT: The href MUST match the page names in your sidebar ("Attacker" and "Striker")
st.markdown("""
<div class="tile-list">
  <a class="tile att" href="./Attacker" target="_self" aria-label="Open Attacker page">Attackers</a>
  <a class="tile str" href="./Striker"  target="_self" aria-label="Open Striker page">Strikers</a>
</div>
""", unsafe_allow_html=True)




