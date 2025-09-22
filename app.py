# app.py — Scouting HQ (2x2x2 tiles; tiles are fully clickable)
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# --------- THEME & TILE CSS ----------
st.markdown("""
<style>
:root{
  --bg:#0b0f1f;
  --ink:#e9eef7;
  --muted:#9fb0c8;
  --radius:22px;
  --gap:32px;
  --height:200px;
}

.stApp{ background:var(--bg); }
.block-container{ max-width:1120px; padding-top:48px; padding-bottom:56px; }

/* Heading */
.hq-title{ margin:0 0 6px 0; color:var(--ink); font-weight:900;
  font-size:clamp(36px,4.6vw,56px); letter-spacing:.2px; }
.hq-sub{ color:var(--muted); margin:0 0 32px 0; }

/* 2-col grid */
.tile-grid{ display:grid; grid-template-columns:1fr 1fr; gap:var(--gap); }
@media (max-width: 980px){ .tile-grid{ grid-template-columns:1fr; } }

/* Base card look */
.card{
  height:var(--height); border-radius:var(--radius);
  box-shadow: 0 22px 44px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05);
  position:relative; overflow:hidden;
  display:flex; align-items:center; justify-content:center; text-align:center;
}
.card .title{ color:#fff; font-weight:800; font-size:clamp(22px,2.2vw,28px);
  text-shadow: 0 4px 16px rgba(0,0,0,.35); }
.card .sub{ color:rgba(255,255,255,.9); font-weight:600; margin-top:8px; }

/* Clickable link-as-card */
a.card{
  text-decoration:none; outline:none;
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
a.card:hover{ transform: translateY(-6px); box-shadow:0 34px 66px rgba(0,0,0,.55); filter:saturate(1.05); }
a.card:active{ transform: translateY(-2px); }

/* Non-click placeholder */
.card.soon::after{
  content:"Coming soon";
  position:absolute; top:14px; right:16px;
  color:rgba(255,255,255,.92); font-weight:700; font-size:12px;
  padding:6px 10px; border-radius:999px;
  background:rgba(0,0,0,.28); border:1px solid rgba(255,255,255,.18);
}

/* Gradients per tile */
.grad-cb  { background:linear-gradient(135deg,#1d976c 0%,#93f9b9 100%); }
.grad-fb  { background:linear-gradient(135deg,#c94b4b 0%,#4b134f 100%); }
.grad-cm  { background:linear-gradient(135deg,#f7971e 0%,#ffd200 100%); }
.grad-att { background:linear-gradient(135deg,#8e2de2 0%,#4a00e0 100%); }
.grad-str { background:linear-gradient(135deg,#2980b9 0%,#2c3e50 100%); }

/* Make Streamlit page links look like our cards */
a[data-testid="stPageLink"]{
  display:block; width:100%;
}
a[data-testid="stPageLink"] > div{ display:none; } /* hide default chip */
</style>
""", unsafe_allow_html=True)

# --------- HEADER ----------
st.markdown('<div class="hq-title">Scouting HQ</div>', unsafe_allow_html=True)
st.markdown('<div class="hq-sub">Central scouting dashboard</div>', unsafe_allow_html=True)

# --------- GRID ----------
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

# Row 1
st.markdown('<div class="card grad-cb soon"><div><div class="title">Center Backs</div><div class="sub">—</div></div></div>', unsafe_allow_html=True)
st.markdown('<div class="card grad-fb soon"><div><div class="title">Fullbacks</div><div class="sub">—</div></div></div>', unsafe_allow_html=True)

# Row 2 (Attackers link; Strikers link)
# Use st.page_link to guarantee proper internal navigation, then wrap it visually as a card.
col = st.columns(2)

with col[0]:
    # Attackers
    st.page_link("pages/02_Attacker.py", label="")  # actual link (hidden visually)
    st.markdown('<a class="card grad-att" href="pages/02_Attacker.py"><div><div class="title">Attackers</div></div></a>',
                unsafe_allow_html=True)

with col[1]:
    # Strikers
    st.page_link("pages/01_Striker.py", label="")  # actual link (hidden visually)
    st.markdown('<a class="card grad-str" href="pages/01_Striker.py"><div><div class="title">Strikers</div></div></a>',
                unsafe_allow_html=True)

# Row 3
st.markdown('<div class="card grad-cm soon"><div><div class="title">Central Midfielders</div><div class="sub">—</div></div></div>', unsafe_allow_html=True)
st.markdown('<div class="card grad-str soon"><div><div class="title">Wingers</div><div class="sub">—</div></div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

