# app.py — Scouting HQ (clean, premium 2×2×2 tiles — no links)
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ============= Styles =============
st.markdown("""
<style>
:root{
  --bg:#0a0f1e;             /* deep navy background */
  --panel:#0f1528;          /* card base (behind gradient) */
  --border:rgba(255,255,255,.10);
  --text:#eaf0f6;           /* main text */
  --muted:#9aa8bd;          /* secondary text */
}

/* page */
.stApp{ background: var(--bg); }
.block-container{ max-width: 1140px; padding-top: 48px; padding-bottom: 32px; }

/* heading */
.hq h1{
  margin: 0 0 10px 0;
  color: var(--text);
  font-weight: 900;
  letter-spacing:.2px;
  font-size: clamp(40px, 5vw, 62px);
  line-height: 1.04;
}
.hq .sub{
  color: var(--muted);
  font-size: 15px;
  letter-spacing:.2px;
  margin-bottom: 18px;
}

/* grid */
.tile-grid{
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 26px;
  margin-top: 12px;
}

/* tile */
.tile{
  position: relative;
  height: 220px;
  border-radius: 20px;
  overflow: hidden;
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow:
      0 22px 60px rgba(0,0,0,.35),
      inset 0 1px 0 rgba(255,255,255,.05);
}
.tile .inner{
  position: relative; z-index: 2;
  height: 100%; display:flex; align-items:center; justify-content:center;
}
.tile h2{
  margin:0;
  color:#0e1426;                   /* dark text so gradients pop */
  text-shadow:0 1px 0 rgba(255,255,255,.45);
  font-weight:900;
  letter-spacing:.3px;
  font-size: clamp(22px, 2.6vw, 32px);
}

/* gradient film (palette per role) */
.tile::before{ content:""; position:absolute; inset:0; z-index:1; opacity:.96; background: var(--grad); }
.tile::after{
  content:""; position:absolute; inset:0; z-index:1;
  background: radial-gradient(1200px 500px at 8% -20%, rgba(255,255,255,.18), transparent 55%);
  mix-blend-mode: soft-light;
}

/* subtle hover, even without links */
.tile{ transition: transform .18s ease, box-shadow .18s ease; }
.tile:hover{
  transform: translateY(-6px);
  box-shadow: 0 30px 70px rgba(0,0,0,.45);
}

/* role palettes (bright, professional, harmonious) */
.gk  { --grad: linear-gradient(135deg,#4e83ff 0%,  #79b4ff 100%); }
.cb  { --grad: linear-gradient(135deg,#1fc48a 0%,  #74e0b6 100%); }
.fb  { --grad: linear-gradient(135deg,#ee67aa 0%, #f3a7cd 100%); }
.cm  { --grad: linear-gradient(135deg,#f3c234 0%, #ffe27b 100%); }
.att { --grad: linear-gradient(135deg,#7a63ff 0%, #b39cff 100%); }
.str { --grad: linear-gradient(135deg,#f25555 0%, #ff9c9c 100%); }

/* responsive */
@media (max-width: 950px){
  .tile-grid{ grid-template-columns: 1fr; }
  .tile{ height: 190px; }
}
</style>
""", unsafe_allow_html=True)

# ============= Hero =============
st.markdown(
    """
    <div class="hq">
      <h1>Scouting HQ</h1>
      <div class="sub">Central scouting dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============= Grid =============
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

def tile(role_class: str, label: str):
    st.markdown(
        f"""
        <div class="tile {role_class}">
          <div class="inner"><h2>{label}</h2></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Order: Goalkeepers, Center Backs, Fullbacks, Central Midfielders, Attackers, Strikers
tile("gk",  "Goalkeepers")
tile("cb",  "Center Backs")
tile("fb",  "Fullbacks")
tile("cm",  "Central Midfielders")
tile("att", "Attackers")
tile("str", "Strikers")

st.markdown('</div>', unsafe_allow_html=True)











