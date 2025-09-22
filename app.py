# app.py — Scouting HQ (minimal, premium 2×2×2 tiles w/ reliable links)
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
:root{
  --bg:#0a0f1e;           /* deep navy */
  --panel:#0f1528;        /* tile base */
  --border:rgba(255,255,255,.12);
  --glass:rgba(255,255,255,.06);
  --text:#eaf0f6;
}

/* Page shell */
.stApp { background: var(--bg); }
.block-container { max-width: 1140px; padding-top: 2.6rem; padding-bottom: 2rem; }

/* Title */
.hq h1{
  margin: 0 0 8px 0;
  color: var(--text);
  font-weight: 900;
  letter-spacing:.2px;
  font-size: clamp(38px, 4.2vw, 56px);
  line-height: 1.06;
}

/* Grid */
.tile-grid{
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 26px;
  margin-top: 18px;
}

/* Tile base */
.tile{
  position: relative;
  border-radius: 18px;
  height: 220px;
  background: var(--panel);
  border: 1px solid var(--border);
  overflow: hidden;
  box-shadow: 0 22px 60px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
}

/* Gradient film (set per role with --grad) */
.tile::before{
  content:""; position:absolute; inset:0; z-index:0; opacity:.95; background: var(--grad);
}
.tile::after{
  content:""; position:absolute; inset:0; z-index:1;
  background: radial-gradient(1200px 500px at 8% -20%, rgba(255,255,255,.18), transparent 55%);
  mix-blend-mode: soft-light;
}

/* Make the whole tile clickable when it’s a link */
.tile > a{
  position: relative; z-index:2; display:flex;
  width:100%; height:100%;
  align-items:center; justify-content:center;
  text-decoration:none;
}

/* Non-clickable tiles use a div with same layout */
.tile > .inner{
  position: relative; z-index:2; display:flex;
  width:100%; height:100%;
  align-items:center; justify-content:center;
}

/* Text */
.tile h2{
  margin:0;
  color: #0e1426;                  /* dark text on bright gradients */
  text-shadow: 0 1px 0 rgba(255,255,255,.4);
  font-weight: 900;
  letter-spacing:.3px;
  font-size: clamp(22px, 2.4vw, 30px);
}

/* Hover only for linked tiles */
.tile.link a { transition: transform .16s ease, box-shadow .16s ease; }
.tile.link:hover a{
  transform: translateY(-6px);
}
.tile.link:hover{
  box-shadow: 0 30px 70px rgba(0,0,0,.45);
}

/* Role palettes (bright, modern, harmonized) */
.gk  { --grad: linear-gradient(135deg,#4e83ff 0%, #79b4ff 100%); }
.cb  { --grad: linear-gradient(135deg,#1fc48a 0%, #74e0b6 100%); }
.fb  { --grad: linear-gradient(135deg,#ee67aa 0%, #f3a7cd 100%); }
.cm  { --grad: linear-gradient(135deg,#f3c234 0%, #ffe27b 100%); }
.att { --grad: linear-gradient(135deg,#7a63ff 0%, #b39cff 100%); }
.str { --grad: linear-gradient(135deg,#f25555 0%, #ff9c9c 100%); }

/* Responsive */
@media (max-width: 950px){
  .tile-grid{ grid-template-columns: 1fr; }
  .tile{ height: 190px; }
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="hq"><h1>Scouting HQ</h1></div>', unsafe_allow_html=True)

# ---------- GRID ----------
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

def linked_tile(role_class: str, label: str, page_file: str):
    # Use Streamlit’s native page link (works on Cloud & locally).
    link = st.page_link(f"pages/{page_file}", label="", help=None, icon=None)
    # `st.page_link` renders an <a>. We wrap it inside our styled tile via HTML block.
    st.markdown(
        f"""
        <div class="tile {role_class} link">
          <a href="{link.href}"><h2>{label}</h2></a>
        </div>
        """,
        unsafe_allow_html=True,
    )

def static_tile(role_class: str, label: str):
    st.markdown(
        f"""
        <div class="tile {role_class}">
          <div class="inner"><h2>{label}</h2></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Row 1
static_tile("gk",  "Goalkeepers")
static_tile("cb",  "Center Backs")

# Row 2
static_tile("fb",  "Fullbacks")
static_tile("cm",  "Central Midfielders")

# Row 3 (live links)
linked_tile("att", "Attackers", "02_Attacker.py")
linked_tile("str", "Strikers",  "01_Striker.py")

st.markdown('</div>', unsafe_allow_html=True)










