# app.py — Scouting HQ (dark, premium tiles, real navigation)
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
<style>
:root{
  --bg:#0b0f1f;
  --text:#eef3fb;
  --muted:#9fb0c8;
  --card:#10162b;
  --stroke:rgba(255,255,255,.06);
}
.stApp{ background:var(--bg); }
.block-container{ max-width:1120px; padding-top:50px; padding-bottom:48px; }

/* hero */
.hq-title{ font-weight:900; color:var(--text); margin:0; letter-spacing:.2px;
  font-size:clamp(36px,4.8vw,54px); }
.hq-sub{ color:var(--muted); margin:6px 0 26px 0; }

/* grid */
.tile-grid{
  display:grid; grid-template-columns:repeat(2,1fr); gap:26px;
}
@media(max-width:980px){ .tile-grid{ grid-template-columns:1fr; } }

/* tile base (card) */
.tile{
  position:relative; height:190px; border-radius:22px; overflow:hidden;
  border:1px solid var(--stroke); background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
  box-shadow:0 24px 44px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
  display:flex; align-items:center; justify-content:center;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.tile:hover{ transform:translateY(-6px); box-shadow:0 34px 66px rgba(0,0,0,.48), inset 0 1px 0 rgba(255,255,255,.06); }

/* glow ring on hover */
.tile::after{
  content:""; position:absolute; inset:-2px; border-radius:24px; pointer-events:none;
  background:radial-gradient(1200px 220px at 30% -10%, rgba(255,255,255,.18), transparent 60%);
  opacity:0; transition:opacity .18s ease;
}
.tile:hover::after{ opacity:1; }

/* gradient skins */
.cb  { background:linear-gradient(135deg,#1d976c,#93f9b9); }
.fb  { background:linear-gradient(135deg,#c94b4b,#4b134f); }
.cm  { background:linear-gradient(135deg,#f7971e,#ffd200); }
.att { background:linear-gradient(135deg,#8e2de2,#4a00e0); }
.str { background:linear-gradient(135deg,#2980b9,#2c3e50); }

/* headings inside tiles */
.tile h2{
  color:#ffffff; text-shadow:0 2px 6px rgba(0,0,0,.45);
  font-weight:850; margin:0; text-align:center;
  font-size:clamp(22px,2.2vw,30px);
}
.tile .sub{ color:rgba(255,255,255,.85); font-weight:600; margin-top:6px; }

/* make Streamlit buttons invisible overlays that fill the card */
.tile .stButton>button{
  position:absolute; inset:0; border:none; background:transparent; color:transparent;
  cursor:pointer; box-shadow:none; padding:0; margin:0;
}
.tile .stButton>button:hover{ background:transparent; }

/* subtle separators above each row on large screens */
.row-sep{ height:2px; }
</style>
""", unsafe_allow_html=True)

# ---------------------- HERO ----------------------
st.markdown('<h1 class="hq-title">Scouting HQ</h1>', unsafe_allow_html=True)
st.markdown('<div class="hq-sub">Central scouting dashboard</div>', unsafe_allow_html=True)

# ---------------------- GRID ----------------------
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

# 1) Center Backs (placeholder)
st.markdown('<div class="tile cb"><h2>Center Backs</h2><div class="sub">Coming soon</div></div>', unsafe_allow_html=True)

# 2) Fullbacks (placeholder)
st.markdown('<div class="tile fb"><h2>Fullbacks</h2><div class="sub">Coming soon</div></div>', unsafe_allow_html=True)

# 3) Central Midfielders (placeholder)
st.markdown('<div class="tile cm"><h2>Central Midfielders</h2><div class="sub">Coming soon</div></div>', unsafe_allow_html=True)

# 4) Attackers (clickable)
st.markdown('<div class="tile att"><h2>Attackers</h2></div>', unsafe_allow_html=True)
col = st.container()
with col:
    # invisible full-card button; on click we switch page
    if st.button("go-attackers", key="btn_att", help=""):
        st.switch_page("pages/02_Attacker.py")

# 5) Strikers (clickable) — will wrap to next row automatically
st.markdown('<div class="tile str"><h2>Strikers</h2></div>', unsafe_allow_html=True)
col2 = st.container()
with col2:
    if st.button("go-strikers", key="btn_str", help=""):
        st.switch_page("pages/01_Striker.py")

st.markdown('</div>', unsafe_allow_html=True)
