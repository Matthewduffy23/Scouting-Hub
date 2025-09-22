# app.py — Premium homepage with 4 role tiles (clickable for Strikers & Attackers)
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ===================== Styles =====================
st.markdown("""
<style>
:root{
  --bg:#0b0f1f;
  --text:#f1f5fa;
  --muted:#a4afc2;
  --border:rgba(255,255,255,.08);
}

/* page */
.stApp{ background: var(--bg); }
.block-container{ max-width: 1100px; padding-top: 50px; padding-bottom: 40px; }

/* heading */
.hero h1{
  margin:0 0 8px 0;
  color: var(--text);
  font-weight:900;
  letter-spacing:.2px;
  font-size: clamp(38px,4.8vw,58px);
}
.hero .sub{
  color: var(--muted);
  font-size: 16px;
  margin-bottom: 25px;
}

/* grid */
.tile-grid{
  display:grid;
  grid-template-columns:repeat(2,1fr);
  gap:28px;
}

/* tile */
.tile{
  position:relative;
  height:210px;
  border-radius:22px;
  overflow:hidden;
  border:1px solid var(--border);
  box-shadow:0 20px 40px rgba(0,0,0,.35);
  display:flex;align-items:center;justify-content:center;
  transition: transform .2s ease, box-shadow .2s ease;
}
.tile:hover{ transform:translateY(-6px); box-shadow:0 28px 60px rgba(0,0,0,.45); }

.tile h2{
  color:#fff;
  font-weight:800;
  font-size: clamp(22px,2.3vw,30px);
  text-shadow:0 2px 6px rgba(0,0,0,.45);
}

/* link wrapper */
.tile a{
  display:flex;
  align-items:center;
  justify-content:center;
  text-decoration:none;
  width:100%; height:100%;
}

/* role palettes */
.cb  { background:linear-gradient(135deg,#16a085,#48c9b0); }
.fb  { background:linear-gradient(135deg,#e74c3c,#ff7675); }
.cm  { background:linear-gradient(135deg,#f39c12,#f1c40f); }
.att { background:linear-gradient(135deg,#8e44ad,#9b59b6); }
.str { background:linear-gradient(135deg,#2980b9,#3498db); }

/* responsive */
@media(max-width:950px){
  .tile-grid{ grid-template-columns:1fr; }
  .tile{ height:180px; }
}
</style>
""", unsafe_allow_html=True)

# ===================== Hero =====================
st.markdown("""
<div class="hero">
  <h1>Scouting HQ</h1>
  <div class="sub">Central scouting dashboard</div>
</div>
""", unsafe_allow_html=True)

# ===================== Tiles =====================
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

# Center Backs
st.markdown('<div class="tile cb"><h2>Center Backs<br><small>Coming soon</small></h2></div>', unsafe_allow_html=True)

# Fullbacks
st.markdown('<div class="tile fb"><h2>Fullbacks<br><small>Coming soon</small></h2></div>', unsafe_allow_html=True)

# Central Midfielders
st.markdown('<div class="tile cm"><h2>Central Midfielders<br><small>Coming soon</small></h2></div>', unsafe_allow_html=True)

# Attackers (clickable)
link_att = st.page_link("pages/02_Attacker.py", label="", icon=None)
st.markdown(f'''
<div class="tile att">
  <a href="{link_att.href}"><h2>Attackers</h2></a>
</div>
''', unsafe_allow_html=True)

# Strikers (clickable)
link_str = st.page_link("pages/01_Striker.py", label="", icon=None)
st.markdown(f'''
<div class="tile str">
  <a href="{link_str.href}"><h2>Strikers</h2></a>
</div>
''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)











