# app.py — Scouting HQ (premium 2×2×2 tile hub)
import streamlit as st

st.set_page_config(page_title="Scouting HQ", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
:root{
  --bg:#0b1222;           /* deep navy */
  --panel:#0e182e;        /* card base */
  --glass:rgba(255,255,255,.08);
  --border:rgba(255,255,255,.14);
  --text:#eef2f7;         /* primary text */
  --muted:#9aa4b2;        /* secondary text */
}

/* Page */
.stApp { background: var(--bg); }
.block-container { padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1150px; }

/* Hero */
.hq-hero h1{
  font-size: clamp(36px, 4.2vw, 52px);
  font-weight: 800;
  letter-spacing: .2px;
  margin: 0 0 .35rem 0;
  color: var(--text);
}
.hq-hero p{
  margin: 0;
  color: var(--muted);
  font-size: 15.5px;
}

/* Decorative underline */
.underline{
  width: 84px; height: 4px; border-radius: 4px; margin-top: 16px;
  background: linear-gradient(90deg,#60a5fa,#34d399,#facc15,#f87171,#a78bfa);
  filter: saturate(120%);
}

/* Grid */
.tile-grid{
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 28px;
  margin-top: 34px;
}

/* Tile base */
.tile{
  position: relative;
  border-radius: 18px;
  padding: 52px 40px;
  overflow: hidden;
  isolation: isolate;
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: 0 18px 50px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
  cursor: default;
  transform: translateZ(0);  /* enable GPU */
}

/* Gradient film (per-role) */
.tile::before{
  content:""; position:absolute; inset:0; z-index:-1;
  background: var(--grad);
  opacity:.82;
}
.tile::after{
  /* top sheen */
  content:""; position:absolute; inset:0; z-index:0;
  background: linear-gradient(to bottom, rgba(255,255,255,.12), transparent 25%);
  mix-blend-mode: soft-light;
}

/* Content */
.tile h2{
  margin: 0 0 6px 0;
  color:#0d1021;
  font-weight: 800;
  letter-spacing:.2px;
  font-size: clamp(22px, 2.2vw, 28px);
  text-shadow: 0 1px 0 rgba(255,255,255,.35);
}
.tile p{
  margin: 0;
  color: #122; 
  opacity: .9;
  font-weight: 600;
}

/* Glass frame so colours look premium */
.inner{
  background: var(--glass);
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 14px;
  padding: 34px 26px;
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.12);
}

/* Hover motion for interactive tiles */
.clickable { cursor: pointer; transition: transform .18s ease, box-shadow .18s ease; }
.clickable:hover{
  transform: translateY(-6px);
  box-shadow: 0 26px 60px rgba(0,0,0,.45);
}

/* Badge for placeholders */
.badge{
  position: absolute; top: 14px; right: 14px;
  background: rgba(15,22,38,.55);
  color: #dbe3ee; border: 1px solid rgba(255,255,255,.18);
  font-size: 12px; padding: 6px 10px; border-radius: 999px;
  letter-spacing:.2px;
}

/* Role palettes (bold but tasteful) */
.gk   { --grad: linear-gradient(135deg,#4f86f9 0%,#6cb8ff 100%); }
.cb   { --grad: linear-gradient(135deg,#22c68a 0%,#76e2b7 100%); }
.fb   { --grad: linear-gradient(135deg,#f06faf 0%,#f6a9cd 100%); }
.cm   { --grad: linear-gradient(135deg,#f3c63b 0%,#ffe17a 100%); }
.att  { --grad: linear-gradient(135deg,#7d67f6 0%,#b19cff 100%); }
.str  { --grad: linear-gradient(135deg,#ef5858 0%,#ff9b9b 100%); }

/* Responsive */
@media (max-width: 950px){
  .tile-grid{ grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ---------- JS nav helper (works in Streamlit Cloud) ----------
# Navigates to a page in /pages by filename.
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
st.markdown('<div class="hq-hero"><h1>Scouting HQ</h1><p>Central scouting dashboard</p><div class="underline"></div></div>', unsafe_allow_html=True)

# ---------- GRID ----------
st.markdown('<div class="tile-grid">', unsafe_allow_html=True)

def tile(role_class, title, subtitle, clickable=False, target=None, badge=None):
    """Render a single tile. If clickable, it will postMessage to navigate."""
    onclick = ""
    extra_cls = ""
    if clickable and target:
        onclick = f"onclick=\"window.parent.postMessage({{type:'streamlit_navigation',page:'{target}'}}, '*');\""
        extra_cls = " clickable"
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="tile {role_class}{extra_cls}" {onclick}>
          {badge_html}
          <div class="inner">
            <h2>{title}</h2>
            <p>{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Row 1
tile("gk",  "Goalkeepers",        "Shot-stopping, claims, distribution", badge="Coming soon")
tile("cb",  "Center Backs",       "Duels, line height, progression",     badge="Coming soon")

# Row 2
tile("fb",  "Fullbacks",          "Overlap/underlap, crossing, duels",   badge="Coming soon")
tile("cm",  "Central Midfielders","Build-up, ball-winning, chance creation", badge="Coming soon")

# Row 3 (live)
tile("att", "Attackers",          "Creative threat, dribbles, xA", clickable=True, target="02_Attacker.py")
tile("str", "Strikers",           "Finishing, xG, box presence",   clickable=True, target="01_Striker.py")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Internal scouting platform • © 2025")








