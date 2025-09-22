# app.py — Scouting HQ (minimal, premium 2×2×2 tiles)
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
:root{
  --bg:#0b1222;           /* deep navy backdrop */
  --panel:#0d162b;        /* tile base */
  --glass:rgba(255,255,255,.06);
  --border:rgba(255,255,255,.14);
  --text:#eef2f7;         /* primary text */
}

/* Page */
.stApp { background: var(--bg); }
.block-container { padding-top: 2.8rem; padding-bottom: 2rem; max-width: 1140px; }

/* Hero */
.hq-hero h1{
  margin: 0 0 10px 0;
  color: var(--text);
  font-weight: 900;
  letter-spacing:.3px;
  font-size: clamp(38px, 4.4vw, 56px);
}

/* Grid */
.tile-grid{
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 28px;
  margin-top: 22px;
}

/* Tile base */
.tile{
  position: relative;
  border-radius: 18px;
  padding: 64px 40px;
  overflow: hidden;
  isolation: isolate;
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: 0 20px 50px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
}

/* Gradient film (per-role) */
.tile::before{
  content:""; position:absolute; inset:0; z-index:-1; opacity:.9; background: var(--grad);
}
.tile::after{
  content:""; position:absolute; inset:0; z-index:0;
  background: radial-gradient(1200px 500px at 10% -20%, rgba(255,255,255,.18), transparent 55%);
  mix-blend-mode: soft-light;
}

/* Content (only the title, centered) */
.inner{
  background: var(--glass);
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 14px;
  padding: 42px 26px;
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  display:flex; align-items:center; justify-content:center;
}
.tile h2{
  margin: 0;
  color: #0f1427;
  text-shadow: 0 1px 0 rgba(255,255,255,.35);
  font-weight: 900;
  letter-spacing:.3px;
  font-size: clamp(24px, 2.6vw, 32px);
}

/* Hover motion only for interactive tiles */
.clickable { cursor: pointer; transition: transform .18s ease, box-shadow .18s ease; }
.clickable:hover{
  transform: translateY(-6px);
  box-shadow: 0 26px 60px rgba(0,0,0,.45);
}

/* Role palettes (bright, refined) */
.gk   { --grad: linear-gradient(135deg,#5386ff 0%,#77b4ff 100%); }
.cb   { --grad: linear-gradient(135deg,#1fc48a 0%,#75e1b6 100%); }
.fb   { --grad: linear-gradient(135deg,#ef6dac 0%,#f3a7cd 100%); }
.cm   { --grad: linear-gradient(135deg,#f1c234 0%,#ffe078 100%); }
.att  { --grad: linear-gradient(135deg,#7c64ff 0%,#b59dff 100%); }
.str  { --grad: linear-gradient(135deg,#f15454 0%,#ff9a9a 100%); }

/* Responsive */
@media (max-width: 950px){
  .tile-grid{ grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ---------- JS nav helper (click tiles -> sidebar page) ----------
st.markdown("""
<script>
window.addEventListener("message", (e)=>{
  if (e.data && e.data.type==="streamlit_navigation"){
    const fname = e.data.page;
    const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
    if(!sidebar){ return; }
    const items = sidebar.querySelectorAll('a');
    for (const a of items){
      if (a.textContent.trim().toLowerCase().includes("striker") && fname.endsWith("01_Striker.py")) { a.click(); return; }
      if (a.textContent.trim().toLowerCase().includes("attacker") && fname.endsWith("02_Attacker.py")) { a.click(); return; }
    }
  }
});
</script>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="hq-hero"><h1>Scouting HQ</h1></div>', unsafe_allow_html=True)

# ---------- GRID ----------
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

def tile(role_class, label, clickable=False, target=None):
    onclick = ""
    extra = ""
    if clickable and target:
        onclick = f"onclick=\"window.parent.postMessage({{type:'streamlit_navigation',page:'{target}'}}, '*');\""
        extra = " clickable"
    st.markdown(
        f"""
        <div class="tile {role_class}{extra}" {onclick}>
          <div class="inner"><h2>{label}</h2></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Row 1
tile("gk",  "Goalkeepers")
tile("cb",  "Center Backs")

# Row 2
tile("fb",  "Fullbacks")
tile("cm",  "Central Midfielders")

# Row 3 (live links)
tile("att", "Attackers", clickable=True, target="02_Attacker.py")
tile("str", "Strikers",  clickable=True, target="01_Striker.py")

st.markdown('</div>', unsafe_allow_html=True)









