# app.py — Scouting Hub (polished landing)
from __future__ import annotations
import streamlit as st

# ---------- Page meta ----------
st.set_page_config(
    page_title="Scouting Hub",
    page_icon="🔎",
    layout="wide",
)

# ---------- Minimal CSS polish ----------
st.markdown("""
<style>
:root{
  --bg:#0b1220;          /* background */
  --card:#111a2b;        /* cards */
  --accent:#22c55e;      /* green */
  --muted:#94a3b8;       /* slate-400 */
  --text:#e5e7eb;        /* slate-200 */
}
.stApp { background:linear-gradient(180deg, var(--bg), #0f172a 60%); }
.block-container { padding-top:2.4rem; padding-bottom:3rem; }
h1,h2,h3,h4,p,li,div,span { color:var(--text); }
small, .helptext, .stCaption, .st-emotion-cache-16idsys p { color:var(--muted) !important; }

.hero{
  display:flex; gap:24px; align-items:center; padding:22px 22px;
  border-radius:18px; background:linear-gradient(135deg,#0e1726, #0b1324 60%);
  border:1px solid #1f2a44; box-shadow:0 10px 30px rgba(0,0,0,.25);
}
.badges{ display:flex; gap:10px; flex-wrap:wrap; }
.badge{
  font-size:.75rem; padding:4px 10px; border-radius:999px;
  border:1px solid #2a3551; color:#cbd5e1; background:#0f1a2d;
}
.kpis{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-top:10px; }
.kpi{
  background:var(--card); border:1px solid #223155; border-radius:14px; padding:16px;
}
.kpi h3{ margin:0; font-size:1.8rem; color:white; }
.kpi p{ margin:.2rem 0 0; color:var(--muted); }

.grid{ display:grid; gap:18px; grid-template-columns:repeat(2,minmax(0,1fr)); }
.card{
  background:var(--card); border:1px solid #223155; border-radius:16px;
  padding:22px; display:flex; flex-direction:column; gap:12px;
}
.card h3{ margin:.2rem 0; }
.card p{ color:var(--muted); margin:0; }
.btnrow{ display:flex; gap:10px; flex-wrap:wrap; }
.btn{
  text-decoration:none; padding:10px 14px; border-radius:10px; font-weight:600;
  border:1px solid #2a3a5f; color:white; background:#16223a;
}
.btn.primary{ background:var(--accent); color:#052e17; border-color:#16a34a; }
.btn:hover{ filter:brightness(1.05); }
.features{ display:grid; gap:16px; grid-template-columns:repeat(3,minmax(0,1fr)); }
.feature{ background:#0f182a; border:1px solid #223055; border-radius:14px; padding:16px; }
.footer{ color:var(--muted); text-align:center; margin-top:28px; font-size:.9rem; }
@media (max-width: 980px){
  .grid{ grid-template-columns:1fr; }
  .features{ grid-template-columns:1fr; }
  .kpis{ grid-template-columns:1fr; }
}
</style>
""", unsafe_allow_html=True)

# ---------- Header / hero ----------
st.markdown("""
<div class="hero">
  <div style="font-size:42px; line-height:1">🔎</div>
  <div>
    <h1 style="margin:0">Scouting Hub</h1>
    <p style="margin:.25rem 0 0; font-size:1.05rem; color:#cbd5e1">
      A single place to launch your analysis apps and share links with staff.
    </p>
    <div class="badges" style="margin-top:8px;">
      <span class="badge">Multipage</span>
      <span class="badge">Attacker & Striker tools</span>
      <span class="badge">Shareable URLs</span>
      <span class="badge">Streamlit</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ---------- Optional headline KPIs (static text, tweak as you like) ----------
st.markdown("""
<div class="kpis">
  <div class="kpi"><h3>2</h3><p>Apps in this hub</p></div>
  <div class="kpi"><h3>⚡ Instant</h3><p>Switch via sidebar or buttons</p></div>
  <div class="kpi"><h3>Link</h3><p>Share this page with staff</p></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ---------- Navigation helpers ----------
def go(page_filename: str):
    """
    Try to open a page programmatically (Streamlit 1.31+).
    Fallback: tell the user to use the sidebar Pages menu.
    """
    try:
        st.switch_page(f"pages/{page_filename}")
    except Exception:
        st.warning("Use the left sidebar → **Pages** to open this module.")

# ---------- App cards ----------
st.markdown("### Open a module")
st.markdown("""
<div class="grid">
  <div class="card">
    <div style="font-size:28px">🎯</div>
    <h3>Attacker App</h3>
    <p>Role tables, player profiles, scatter, similar players, radar, and club fit for creators/forwards.</p>
    <div class="btnrow">
      <a class="btn primary" href="#" onClick="window.parent.postMessage({type:'streamlit_navigation',page:'01_Attacker.py'},'*'); return false;">Open Attacker</a>
      <a class="btn" href="https://github.com" target="_blank">Docs</a>
    </div>
  </div>

  <div class="card">
    <div style="font-size:28px">🧭</div>
    <h3>Striker App</h3>
    <p>Goal threat focus: xG, shot quality, box activity, finishing proxies & shortlists.</p>
    <div class="btnrow">
      <a class="btn primary" href="#" onClick="window.parent.postMessage({type:'streamlit_navigation',page:'01_Striker.py'},'*'); return false;">Open Striker</a>
      <a class="btn" href="https://github.com" target="_blank">Docs</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Tiny JS bridge so the “Open …” buttons can call st.switch_page()
nav_target = st.empty()
msg = st.session_state.get("_nav_msg", "")
if msg:
    st.info(msg)

st.markdown("""
<script>
window.addEventListener("message", (e) => {
  if(!e.data || e.data.type !== "streamlit_navigation") return;
  const page = e.data.page || "";
  // Send a query flag Streamlit can read (fallback if st.switch_page not available)
  const qs = new URLSearchParams(window.location.search);
  qs.set("go", page);
  window.location.search = qs.toString();
});
</script>
""", unsafe_allow_html=True)

# Fallback: if ?go=… is present, try switch_page on the Python side
go_param = st.query_params.get("go")
if go_param:
    go(go_param)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---------- Feature highlights ----------
st.markdown("### Why this hub?")
st.markdown("""
<div class="features">
  <div class="feature">
    <strong>One link for staff</strong>
    <p>Share a single URL; staff choose the tool from the sidebar.</p>
  </div>
  <div class="feature">
    <strong>Clean separation</strong>
    <p>Each module lives in <code>pages/</code>, so you can iterate independently.</p>
  </div>
  <div class="feature">
    <strong>Ready for more</strong>
    <p>Add <code>03_Midfield.py</code>, <code>04_Wingers.py</code>, etc. They appear automatically.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built with ❤️ for scouting workflows. Customize text, colors, and cards in <code>app.py</code>.</div>", unsafe_allow_html=True)

